# localagg_tr/dual_backend_head.py
"""
Dual-backend voxelization for a GaussTR-style Gaussian head.

This module holds only the parts of the head that turn predicted 3D Gaussians
into a semantic occupancy grid. How the Gaussians themselves are predicted
(query decoding, depth sampling, scale / opacity / semantic heads, covariance
construction) and how the head is trained are NOT part of this repo.

Usage inside a head that already produces Gaussians:

    class MyHead(DualBackendVoxelHeadMixin, BaseModule):
        def __init__(self, ..., voxelizer, tau_quantile=0.90,
                     s_max_xyz=(0.26, 0.26, 0.26), prompt_denoising=True):
            super().__init__()
            ...
            self.prompt_denoising = prompt_denoising  # used by 'gaussvox' path
            self.init_dual_backend(voxelizer, tau_quantile, s_max_xyz)

        def forward(self, x, ..., mode='tensor'):
            means3d, opacities_raw, scales = ...          # your Gaussian head
            means3d, opacities, scales, debug = self.sanitize_gaussians(
                means3d, opacities_raw, scales)
            covariances, semantics, features = ...        # built from sanitized scales
            gaussian_info = self.make_gaussian_info(
                means3d, opacities, semantics, covariances, scales,
                opacities_raw, debug)
            if mode == 'predict':
                return self.predict_occupancy(
                    means3d, opacities, semantics, scales, covariances,
                    features, gaussian_info)
            ...

Backend is chosen by the config's `voxelizer=dict(type=...)`:
  - 'LocalAggWrapper'          -> "localagg"
  - 'GaussianVoxelizerCompat'  -> "gaussvox"
"""

import torch
import torch.nn.functional as F
from mmdet3d.registry import MODELS

# Upstream GaussTR (https://github.com/hustvl/GaussTR, MIT). Must be importable.
from gausstr.models.gausstr_head import prompt_denoising

from .gaussian_voxelizer_compat import GaussianVoxelizerCompat
from .localagg_wrapper import LocalAggWrapper


class DualBackendVoxelHeadMixin:
    """
    Adds to a Gaussian head:
      - Two voxelization backends switchable by config:
          * LocalAggWrapper ("localagg")
          * GaussianVoxelizerCompat ("gaussvox")
      - Unified "free" masking policy using density quantile (tau_quantile)
      - Geometry sanitization:
          * pc_range-aware clamping
          * scale softplus + [s_min, s_max_xyz] stabilization
          * opacity lower bound for stable gradients
    """

    def init_dual_backend(self,
                          voxelizer,
                          tau_quantile=0.90,
                          s_max_xyz=(0.26, 0.26, 0.26)):
        # Global density-quantile used to define "occupied vs free" in 3D
        self.tau_quantile = float(tau_quantile)

        # Per-axis upper cap for Gaussian scales σ_x, σ_y, σ_z.
        # Used after softplus to prevent overly large Gaussians that could
        # pollute many voxels in LocalAgg / GaussVoxelizer.
        self.register_buffer(
            "s_max_xyz",
            torch.tensor(s_max_xyz, dtype=torch.float32),
            persistent=True,
        )

        # Build voxelizer (can be LocalAggWrapper or GaussianVoxelizerCompat)
        self.voxelizer = MODELS.build(voxelizer)

        # Backend tag: used to choose prediction path
        if isinstance(self.voxelizer, LocalAggWrapper):
            self._voxel_backend = "localagg"
        elif isinstance(self.voxelizer, GaussianVoxelizerCompat):
            self._voxel_backend = "gaussvox"
        else:
            # Fallback; training should usually assert on this in configs
            self._voxel_backend = "unknown"

    # ---------------------------------------------------------------------- #
    # Geometry / scale / opacity sanitization before voxelization
    # ---------------------------------------------------------------------- #
    def _head_pc_max_open(self, means3d, pc_min, pc_max):
        dev = means3d.device

        # Use the same right-open bound policy as the voxelizer/Wrapper:
        #   if voxelizer implements _pc_max_open, delegate to it.
        if hasattr(self.voxelizer, "_pc_max_open"):
            pc_max_open = self.voxelizer._pc_max_open(
                device=dev,
                dtype=torch.float32,
            ).to(means3d.dtype)
        else:
            # Reconstruct an equivalent right-open bound policy from
            # use_nextafter / eps_base / eps_ratio + voxel_size
            use_next = getattr(self.voxelizer, "use_nextafter", True)
            if use_next:
                # "nextafter" policy:
                #   move pc_max towards pc_min by one ULP (float32),
                #   so that the max bound becomes right-open.
                pc_max_open = torch.nextafter(
                    pc_max.float(),
                    pc_min.float(),
                ).to(means3d.dtype)
            else:
                # Epsilon policy:
                #   pc_max_open = pc_max - max(eps_base, voxel_size * eps_ratio)
                vx_attr = getattr(self.voxelizer, "voxel_size", None)

                if isinstance(vx_attr, torch.Tensor):
                    if vx_attr.numel() == 1:
                        vx = float(vx_attr.item())
                    else:
                        # If voxel_size is shape [1] or [3], use the first element
                        vx = float(vx_attr.view(-1)[0].item())
                else:
                    # Already a float-like scalar
                    vx = float(vx_attr)

                eps_base = float(getattr(self.voxelizer, "eps_base", 1e-6))
                eps_ratio = float(getattr(self.voxelizer, "eps_ratio", 1e-3))
                eps = max(eps_base, vx * eps_ratio)
                pc_max_open = (pc_max - eps).to(means3d.dtype)

        return pc_max_open

    def sanitize_gaussians(self, means3d, opacities_raw, scales):
        """
        Args:
            means3d:       [B, N, G, 3] Gaussian centers in ego frame
            opacities_raw: [B, N, G, 1] raw α from the opacity head
            scales:        [B, N, G, 3] scales before softplus / cap

        Returns:
            means3d, opacities, scales (sanitized) and a dict of debug masks.
        """
        dev = means3d.device
        pc_min = self.voxelizer.pc_min.to(device=dev, dtype=means3d.dtype)
        pc_max = self.voxelizer.pc_max.to(device=dev, dtype=means3d.dtype)
        pc_max_open = self._head_pc_max_open(means3d, pc_min, pc_max)

        opacities = opacities_raw.clone()

        # 0) Record original numerical validity and range BEFORE any sanitization
        orig_finite = torch.isfinite(means3d).all(dim=-1)  # [B, N, G]

        inrange_before = (
            (means3d[..., 0] >= pc_min[0]) & (means3d[..., 0] < pc_max_open[0]) &
            (means3d[..., 1] >= pc_min[1]) & (means3d[..., 1] < pc_max_open[1]) &
            (means3d[..., 2] >= pc_min[2]) & (means3d[..., 2] < pc_max_open[2])
        )  # [B, N, G]
        out_of_range_before = ~inrange_before

        # "Geometrically valid" Gaussians at Head level:
        #   - finite coordinates
        #   - within pc_range (right-open)
        valid_geo_head = orig_finite & inrange_before       # [B, N, G]
        drop_geo_head = ~valid_geo_head                     # [B, N, G]

        # 3) Coordinate sanitization + clamping to pc_range
        #    Replace NaN/Inf with bounded values, then clamp.
        means3d = torch.nan_to_num(
            means3d,
            nan=pc_min.mean(),
            posinf=pc_max.max(),
            neginf=pc_min.min(),
        )

        # 4) Scale / σ stabilization and debug statistics
        #    - Lower bound via softplus
        #    - Upper bound per axis via s_max_xyz
        s_min = scales.new_tensor([5e-4, 5e-4, 5e-4])
        s_max = self.s_max_xyz.to(device=scales.device, dtype=scales.dtype)

        # Apply softplus + s_min (this is treated as "pre_cap" scale)
        scales_pre_cap = s_min + F.softplus(scales)  # [B, N, G, 3]
        # Apply upper cap for numerical stability and to avoid huge Gaussians
        scales = torch.minimum(scales_pre_cap, s_max)  # [B, N, G, 3]

        # Per-Gaussian max σ before/after cap (for debugging / hooks)
        with torch.no_grad():
            smax_pre = scales_pre_cap.max(dim=-1).values   # [B, N, G]
            smax_post = scales.max(dim=-1).values          # [B, N, G]

        # 5) Opacity stabilization:
        #    - Enforce a small lower bound to avoid vanishing gradients near 0
        alpha_min = 5e-4
        opacities = opacities.clamp_min(alpha_min)  # [B, N, G, 1]

        # Zero out opacities for geometrically invalid Gaussians
        zero_opa = opacities.new_zeros(opacities.shape)
        mask_geo = (
            valid_geo_head[..., None] if opacities.dim() == 4 else valid_geo_head
        )
        opacities = torch.where(mask_geo, opacities, zero_opa)

        debug = dict(
            smax_pre=smax_pre,
            smax_post=smax_post,
            orig_finite=orig_finite,
            out_of_range_before=out_of_range_before,
            drop_geo_head=drop_geo_head,
        )
        return means3d, opacities, scales, debug

    @staticmethod
    def make_gaussian_info(means3d, opacities, semantics, covariances, scales,
                           opacities_raw, debug):
        # Collect Gaussian-level info for hooks / debugging / visualization.
        # All fields are detached to avoid interfering with backprop.
        return dict(
            means3d=means3d.detach(),          # [B, N, G, 3]
            opacities=opacities.detach(),      # [B, N, G, 1]
            semantics=semantics.detach(),      # semantic logits
            covariances=covariances.detach(),
            scales=scales.detach(),
            alpha_pre=opacities_raw.detach(),  # raw α before floor
            # Scale debugging
            smax_pre=debug["smax_pre"].detach(),        # max σ before cap
            smax_post=debug["smax_post"].detach(),      # max σ after cap
            # Coordinate debugging
            orig_finite=debug["orig_finite"].detach(),              # NaN/Inf mask
            out_of_range_before=debug["out_of_range_before"].detach(),  # out of pc_range before clamp
            drop_geo_head=debug["drop_geo_head"].detach(),          # head-level geometric drop mask
        )

    # ---------------------------------------------------------------------- #
    # PREDICT MODE: dispatch to LocalAgg backend or GaussVoxelizer backend
    # ---------------------------------------------------------------------- #
    def predict_occupancy(self, means3d, opacities, semantics, scales,
                          covariances, features, gaussian_info):
        """Returns (preds_3d [B, H, W, D], gaussian_info)."""
        backend = getattr(self, "_voxel_backend", "unknown")

        if backend == "localagg":
            preds, gaussian_info = self._forward_predict_localagg(
                means3d,
                opacities,
                semantics,
                scales,
                covariances,
                gaussian_info,
            )

        elif backend == "gaussvox":
            preds, gaussian_info = self._forward_predict_gaussvox(
                means3d,
                opacities,
                semantics,
                scales,
                covariances,
                features,
                gaussian_info,
            )

        else:
            raise NotImplementedError(
                f"Unknown voxelizer backend: {backend}"
            )

        return preds, gaussian_info

    def _forward_predict_localagg(
        self,
        means3d,
        opacities,
        semantics,
        scales,
        covariances,
        gaussian_info,
    ):
        """
        Prediction path when voxelizer is LocalAggWrapper.

        Steps:
          1) Run LocalAgg once with semantic logits to get class logits_3d.
          2) Run LocalAgg again with all-ones semantics to get a density proxy.
          3) Normalize logits by density proxy.
          4) Apply tau-quantile-based free masking.
        """
        out = self.voxelizer(
            means3d=means3d.flatten(1, 2),
            opacities=opacities.flatten(1, 2).squeeze(-1),
            semantics=semantics.flatten(1, 2),
            scales=scales.flatten(1, 2),
            covariances=covariances.flatten(1, 2),
        )
        logits_3d = out["logits_3d"]  # [B, C, H, W, D]

        # Propagate gate information (effective α, hard/soft masks) into gaussian_info
        alpha_eff = out.get("alpha_eff", None)
        if alpha_eff is not None:
            gaussian_info["alpha_eff"] = alpha_eff.detach()
            gaussian_info["valid_hard"] = out["valid_hard"].detach()
            gaussian_info["valid_soft"] = out["valid_soft"].detach()
            gaussian_info["drop_mask_soft"] = out["drop_mask_soft"].detach()

        # Re-run LocalAgg with all-ones semantics to obtain a "density proxy"
        sem_ones = torch.ones_like(semantics)
        with torch.no_grad():
            out_den = self.voxelizer(
                means3d=means3d.flatten(1, 2),
                opacities=opacities.flatten(1, 2).squeeze(-1),
                semantics=sem_ones.flatten(1, 2),
                scales=scales.flatten(1, 2),
                covariances=covariances.flatten(1, 2),
            )
            den_logits = out_den["logits_3d"]
            density_proxy = den_logits[:, 0]  # [B, H, W, D]

        # Normalize logits by density proxy (avoid division by zero)
        eps = 1e-6
        logits_3d = logits_3d / (density_proxy.unsqueeze(1) + eps)

        gaussian_info.update(
            logits_localagg=logits_3d.detach(),
            density_proxy=density_proxy.detach(),
        )

        probs = logits_3d.softmax(dim=1)
        preds = probs.argmax(dim=1)

        # Quantile-based free masking:
        #   - Treat low-density voxels as FREE (class id 17)
        q = density_proxy.view(density_proxy.size(0), -1) \
            .quantile(self.tau_quantile, dim=1, keepdim=True)
        tau = q.view(-1, 1, 1, 1).to(density_proxy.dtype)
        preds = torch.where(
            density_proxy > tau,
            preds,
            preds.new_full(preds.shape, 17),
        )

        return preds, gaussian_info

    # ---------------------------------------------------------------------- #
    # GaussVoxelizer backend prediction
    # ---------------------------------------------------------------------- #
    def _forward_predict_gaussvox(
        self,
        means3d,
        opacities,
        semantics,
        scales,
        covariances,
        features,      # unused; kept for a uniform backend signature
        gaussian_info,
    ):
        """
        GaussVoxelizer path, kept identical to the original GaussTR predict path:
          - features: per-Gaussian semantics (softmax)
          - label mapping: no merge / shift
          - free/occupied: density > 4e-2
        means3d / opacities / covariances are the sanitized values from
        sanitize_gaussians(), same as the LocalAgg path.
        """

        # 1) Use semantics as features, as in the original
        density, grid_feats = self.voxelizer(
            means3d=means3d.flatten(1, 2),
            opacities=opacities.flatten(1, 2),
            features=semantics.flatten(1, 2).softmax(-1),
            covariances=covariances.flatten(1, 2),
        )
        # density:    [B, H, W, D, 1]
        # grid_feats: [B, H, W, D, C]

        # 2) prompt_denoising / softmax unchanged from the original
        if self.prompt_denoising:
            probs = prompt_denoising(grid_feats)
        else:
            probs = grid_feats.softmax(-1)

        # 3) Label mapping: same as the original baseline
        #    (no OCC3D_CATEGORIES merge / class shift)
        preds = probs.argmax(-1)    # [B, H, W, D]

        # 4) Free/occupied policy: density > 4e-2
        density = density.squeeze(-1)   # [B, H, W, D]
        thr = density.new_tensor(4e-2)

        preds = torch.where(
            density > thr,
            preds,
            preds.new_full(preds.shape, 17),   # free class id = 17
        )

        # 5) Keep tensors for debugging / hooks
        gaussian_info.update(
            density_gaussvox=density.detach(),
            logits_gaussvox=grid_feats.permute(0, 4, 1, 2, 3).detach(),  # [B, C, H, W, D]
        )

        return preds, gaussian_info


def cache_predict_outputs(model, preds, gaussian_info):
    """
    Call from the model's forward(mode='predict') after the head returns.
    The dump hooks in localagg_tr.hooks read these two attributes.
    """
    # Keep on CPU, not GPU
    model._last_gaussian_info = {
        k: v.detach().cpu() for k, v in gaussian_info.items()
    }
    model._last_preds_head = preds.detach().cpu()
