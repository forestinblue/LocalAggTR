# localagg_tr/localagg_wrapper.py

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet3d.registry import MODELS

# Python wrapper for the GaussianFormer Local Aggregation CUDA extension.
# Not part of this repo: install it from GaussianFormer (see third_party/README.md).
from local_aggregate import LocalAggregator as _GF_LocalAggregator


@MODELS.register_module()
class LocalAggWrapper(nn.Module):
    """
    Adapter between GaussTR and GaussianFormer Local Aggregation kernel.

    Responsibility separation:
      - This adapter (LocalAggWrapper) only produces **3D grid class logits**.
      - Final gating / label decisions (density / confidence thresholds, etc.)
        are handled exclusively in the **Head**.

    Conventions:
      - GaussTR: pc_range / voxel_size=0.4 / grid_shape=(200, 200, 16)
      - GaussianFormer: aggregator(
            pts[B=1, N, 3],
            means[B, G, 3],
            origi_opa[B, G],
            semantics[B, G, C],
            scales[B, G, 3],
            CovInv[B, G, 3, 3]
        )  -> logits [N, C]  (C=18 here)

    This wrapper:
      - Builds grid centers and flattens them to [1, N, 3]
      - Normalizes / reshapes inputs
      - Converts Σ(3×3) to Σ⁻¹(3×3)
      - Calls the CUDA kernel
      - Reshapes outputs back to [B, C, H, W, D]
      - **Does NOT** decide labels or apply final semantic gating

    Single source of truth for boundary policy:
      - The right-open bound policy for pc_range is defined and stored here.
        Head modules should only reference it, not re-implement it.

    This version introduces:
      - **Hard gate (safety / range)** + **Soft gate (opacity attenuation)**

        · Hard gate:
            - Strictly drop NaN / Inf / out-of-range Gaussians
        · Soft gate:
            - For Gaussians with too small α / scale, attenuate α instead of
              completely dropping them, so they still contribute weakly.

    Returned dict:
      - "logits_3d"      : [B, C, H, W, D]
      - "alpha_eff"      : [B, G] (effective α after all gating)
      - "valid_hard"     : [B, G] (hard gate pass mask)
      - "valid_soft"     : [B, G] (soft gate pass mask)
      - "drop_mask_soft" : [B, G] (soft-attenuated Gaussians)
    """

    def __init__(
        self,
        pc_range,                 # list/tuple, e.g. [-40, -40, -1, 40, 40, 5.4]
        voxel_size: float = 0.4,  # 0.4m (GaussTR default)
        grid_shape=(200, 200, 16),
        num_classes: int = 18,
        # Even when using AMP, it is recommended to keep Aggregator in FP32
        force_fp32: bool = True,
        # Radius factor inside GaussianFormer LocalAggregator
        # (mapping scale → search radius).
        scale_multiplier: float = 3.0,
        chunk_size: int | None = None,
        strict_check: bool = False,
        log_gate_ratio: bool = True,
        fail_soft_when_all_drop: bool = True,
        # === Boundary policy options (single source of truth) ===
        use_nextafter: bool = True,   # Default: use nextafter-based right-open bound
        eps_base: float = 1e-6,
        eps_ratio: float = 1e-3,
        # === Soft gate (attenuation) parameters ===
        soft_alpha_min: float = 5e-4,     # opacity threshold (relaxed from 1e-3 → 5e-4)
        soft_scale_min: float = 5e-5,     # scale threshold (relaxed from 1e-4 → 5e-5)
        soft_drop_atten: float = 0.8,     # attenuation factor for soft-dropped Gaussians
    ):
        super().__init__()
        self.chunk_size = int(chunk_size) if chunk_size else 0
        assert len(pc_range) == 6
        self.H, self.W, self.D = map(int, grid_shape)
        self.num_classes = int(num_classes)
        self.force_fp32 = bool(force_fp32)

        # Gating / monitoring options
        self.strict_check = strict_check
        self.log_gate_ratio = log_gate_ratio
        self.fail_soft_when_all_drop = fail_soft_when_all_drop
        self._ema_gate_hard = None
        self._ema_gate_soft = None

        # === Boundary policy config ===
        self.use_nextafter = bool(use_nextafter)
        self.eps_base = float(eps_base)
        self.eps_ratio = float(eps_ratio)

        # === Soft gate parameters ===
        self.soft_alpha_min = float(soft_alpha_min)
        self.soft_scale_min = float(soft_scale_min)
        self.soft_drop_atten = float(soft_drop_atten)

        # Constant buffers (fixed for training / inference)
        pc_min = torch.tensor(pc_range[:3], dtype=torch.float32)
        pc_max = torch.tensor(pc_range[3:], dtype=torch.float32)
        self.register_buffer("pc_min", pc_min, persistent=False)
        self.register_buffer("pc_max", pc_max, persistent=False)
        self.register_buffer(
            "voxel_size",
            torch.tensor([voxel_size], dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "grid_shape_buf",
            torch.tensor([self.H, self.W, self.D], dtype=torch.int32),
            persistent=False,
        )

        # GaussianFormer Aggregator instance (CUDA extension wrapper)
        self.aggregator = _GF_LocalAggregator(
            H=self.H,
            W=self.W,
            D=self.D,
            pc_min=self.pc_min,
            grid_size=float(voxel_size),
            scale_multiplier=scale_multiplier,
            inv_softmax=False,   # not used in this setup
        )

        # Lazily cached grid centers (used often)
        # Shape: [1, N, 3] (B=1 assumption), created on first use.
        self._centers_cache = None

    # --------------------------------------------------------------------- #
    # Internal utilities
    # --------------------------------------------------------------------- #
    @torch.no_grad()
    def _voxel_centers(self, device, dtype=torch.float32) -> torch.Tensor:
        """
        Generate 3D grid centers using:
            center = (index + 0.5) * voxel_size + pc_min

        Returns:
            centers: [1, N, 3] (N = H*W*D), float32, on the given device.
        """
        if (self._centers_cache is not None) and (self._centers_cache.device == device):
            return self._centers_cache

        H, W, D = self.H, self.W, self.D
        vx = float(self.voxel_size.item())
        pc0 = self.pc_min.to(device=device, dtype=dtype)  # [3]

        xs = torch.arange(H, device=device, dtype=dtype)
        ys = torch.arange(W, device=device, dtype=dtype)
        zs = torch.arange(D, device=device, dtype=dtype)
        X, Y, Z = torch.meshgrid(xs, ys, zs, indexing="ij")  # [H, W, D]

        centers = torch.stack([X, Y, Z], dim=-1)  # [H, W, D, 3]
        centers = (centers + 0.5) * vx + pc0      # cell centers
        centers = centers.reshape(1, H * W * D, 3).contiguous()  # [1, N, 3]

        self._centers_cache = centers  # cache for reuse
        return centers

    @staticmethod
    def _cov_to_inv33(cov3x3: torch.Tensor) -> torch.Tensor:
        """
        Convert Σ(...,3,3) to Σ⁻¹(...,3,3).

        The inverse is computed in double precision (FP64) for numerical
        stability, then cast back to FP32.
        """
        inv = torch.linalg.inv(cov3x3.double())
        return inv.to(torch.float32)  # [B, G, 3, 3]

    def _pc_max_open(self, device, dtype=torch.float32):
        """
        Helper to compute the 'right-open upper bound' for pc_range.

        Two modes:
          - nextafter mode:
              Use the representable float value just below pc_max in FP32,
              moving one ULP toward pc_min.
          - epsilon mode:
              pc_max - max(eps_base, voxel_size * eps_ratio).
        """
        pc_min = self.pc_min.to(device=device, dtype=dtype)
        pc_max = self.pc_max.to(device=device, dtype=dtype)
        if self.use_nextafter:
            # Move pc_max down by one ULP in FP32 toward pc_min
            return torch.nextafter(pc_max.float(), pc_min.float()).to(dtype)
        else:
            vx = float(self.voxel_size.item())
            eps = max(self.eps_base, vx * self.eps_ratio)
            return (pc_max - eps).to(dtype)

    # --------------------------------------------------------------------- #
    # Forward path
    # --------------------------------------------------------------------- #
    def forward(
        self,
        *,
        # Tensors produced by the GaussTR head (see dual_backend_head.py)
        means3d: torch.Tensor,        # [B, G, 3]
        opacities: torch.Tensor,      # [B, G] or [B, G, 1] (if None, will be replaced with ones)
        semantics: torch.Tensor,      # [B, G, C] (class logits preferred, pre-softmax)
        scales: torch.Tensor,         # [B, G, 3]
        covariances: torch.Tensor,    # [B, G, 3, 3] (Σ already includes rotation)
    ):
        """
        Returns:
            dict with:
              - logits_3d : [B, C, H, W, D]
              - alpha_eff : [B, G]  effective α after gating
              - valid_hard, valid_soft, drop_mask_soft : [B, G] masks

        NOTE:
            Final label decisions and masking policies are handled in the Head.
        """
        B, G, _ = means3d.shape
        assert B >= 1, "batch size must be >= 1"
        dev = means3d.device

        # 1) Generate grid centers (pts): [1, N, 3], fp32, same device
        pts = self._voxel_centers(device=dev, dtype=torch.float32)

        # 2) Prepare inputs (dtype/device/contiguous, FP32 recommended)
        def _prep(x):
            return x.to(device=dev, dtype=torch.float32).contiguous()

        means3d     = _prep(means3d)
        scales      = _prep(scales).detach()   # no gradients needed for radius computation
        semantics   = _prep(semantics)         # class logits
        covariances = _prep(covariances)

        if opacities is None:
            opacities = torch.ones((B, G), device=dev, dtype=torch.float32)
        else:
            opacities = _prep(opacities).view(B, G)  # normalize to [B, G]

        # ------------------------------------------------------------------ #
        # 2.5) Gating & monitoring
        # ------------------------------------------------------------------ #
        # (a) Safety / range: hard gate
        finite_mask = (
            torch.isfinite(means3d).all(dim=-1) &
            torch.isfinite(scales).all(dim=-1) &
            torch.isfinite(covariances).flatten(-2).all(dim=-1) &  # (3,3) → 9
            torch.isfinite(semantics).all(dim=-1)
        )

        pc_max_open = self._pc_max_open(device=dev, dtype=torch.float32)  # right-open upper bound

        inrange = (
            (means3d[..., 0] >= self.pc_min[0]) & (means3d[..., 0] < pc_max_open[0]) &
            (means3d[..., 1] >= self.pc_min[1]) & (means3d[..., 1] < pc_max_open[1]) &
            (means3d[..., 2] >= self.pc_min[2]) & (means3d[..., 2] < pc_max_open[2])
        )

        # Hard-valid = numerically safe AND within pc_range
        valid_hard = finite_mask & inrange

        # (b) Soft gate candidates: valid but "weak" Gaussians
        #     - We relax the old thresholds (1e-3 / 1e-4) to:
        #         α >= 5e-4, scale > 5e-5
        opa_ok = opacities >= self.soft_alpha_min           # [B, G]
        sc_ok  = (scales > self.soft_scale_min).all(dim=-1) # [B, G]
        valid_soft = opa_ok & sc_ok

        # (c) Effective "valid" for kernel input:
        #     - Hard gate decides which Gaussians can enter the kernel at all.
        #     - Soft gate only attenuates α; it does NOT fully drop them.
        valid = valid_hard
        drop_mask_soft = valid_hard & (~valid_soft)         # [B, G]

        # Logging / monitoring of gate ratios during training
        if self.log_gate_ratio and self.training:
            # Hard drop ratio: fraction of Gaussians that fail the hard gate
            hard_drop_ratio = 1.0 - valid_hard.float().mean().item()
            self._ema_gate_hard = (
                hard_drop_ratio
                if self._ema_gate_hard is None
                else 0.9 * self._ema_gate_hard + 0.1 * hard_drop_ratio
            )

            # Soft drop ratio: among hard-valid Gaussians, how many get soft-attenuated
            hard_keep = valid_hard.float().sum().item()
            soft_drop_ratio = (
                drop_mask_soft.float().sum().item() / max(hard_keep, 1.0)
            )
            self._ema_gate_soft = (
                soft_drop_ratio
                if self._ema_gate_soft is None
                else 0.9 * self._ema_gate_soft + 0.1 * soft_drop_ratio
            )

            # Simple warnings when gate behavior becomes suspicious
            if hard_drop_ratio > 0.2 or (self._ema_gate_hard is not None and self._ema_gate_hard > 0.15):
                print(f"[gate-hard] drop={hard_drop_ratio:.1%} (EMA {self._ema_gate_hard:.1%})")
            if soft_drop_ratio > 0.3 or (self._ema_gate_soft is not None and self._ema_gate_soft > 0.25):
                print(
                    f"[gate-soft] atten={soft_drop_ratio:.1%} "
                    f"(EMA {self._ema_gate_soft:.1%})  (alpha *= {self.soft_drop_atten})"
                )

        # (d) Critical case: no hard-valid Gaussians in the entire batch
        if not valid.any():
            msg = "[gate][WARN] all Gaussians invalid this batch (hard gate)"
            if self.strict_check:
                # Make it a hard failure in debug mode
                raise AssertionError(msg)
            else:
                print(msg)
                if self.fail_soft_when_all_drop:
                    # Return all-zero logits as a fail-soft fallback
                    zeros = means3d.new_zeros((B, self.num_classes, self.H, self.W, self.D))
                    return {"logits_3d": zeros}

        # (e) Soft drop: attenuate α for hard-valid but soft-invalid Gaussians
        if drop_mask_soft.any() and (self.soft_drop_atten is not None) and (self.soft_drop_atten < 1.0):
            opacities = torch.where(drop_mask_soft, opacities * self.soft_drop_atten, opacities)

        # (f) Hard invalid → α = 0 (fully excluded from the kernel)
        opacities = torch.where(valid_hard, opacities, opacities.new_zeros((B, G)))

        # At this point, opacities already incorporate:
        #   - Hard gate exclusion (α=0 for invalid)
        #   - Soft gate attenuation for weak Gaussians
        alpha_eff = opacities  # [B, G]

        # ------------------------------------------------------------------ #
        # 3) Σ → Σ⁻¹ conversion
        # ------------------------------------------------------------------ #
        covinv33 = self._cov_to_inv33(covariances)  # [B, G, 3, 3]

        # ------------------------------------------------------------------ #
        # 4) Call Aggregator: only pass hard-valid Gaussians into the kernel
        # ------------------------------------------------------------------ #
        logits_list = []
        N = self.H * self.W * self.D
        C = self.num_classes

        with torch.cuda.amp.autocast(enabled=(not self.force_fp32)):
            for b in range(B):
                vb = valid[b]  # [G]; hard gate mask for batch b
                if not vb.any():
                    # Entire batch element is invalid; use fail-soft fallback
                    if self.fail_soft_when_all_drop:
                        logits_bcN = means3d.new_zeros((1, C, N))
                        logits_list.append(logits_bcN)
                    continue

                # FP32 accumulation buffer over chunks:
                # we sum per-chunk [N, C] outputs into logits_accum_NC
                logits_accum_NC = means3d.new_zeros((N, C), dtype=torch.float32)

                idx_all = torch.nonzero(vb, as_tuple=False).squeeze(1)  # [G_valid]
                if self.chunk_size and idx_all.numel() > self.chunk_size:
                    # Chunked aggregator calls to avoid OOM when many Gaussians exist
                    for start in range(0, idx_all.numel(), self.chunk_size):
                        sl = idx_all[start:start + self.chunk_size]  # [G_chunk]
                        out_nc = self.aggregator(
                            pts,
                            means3d[b:b+1, sl],      # [1, Gc, 3]
                            opacities[b:b+1, sl],    # [1, Gc]
                            semantics[b:b+1, sl],    # [1, Gc, C]
                            scales[b:b+1, sl],       # [1, Gc, 3]
                            covinv33[b:b+1, sl],     # [1, Gc, 3, 3]
                        )  # [N, C]
                        logits_accum_NC = logits_accum_NC + out_nc.to(torch.float32)
                else:
                    # Single call (original path) without chunking
                    out_nc = self.aggregator(
                        pts,
                        means3d[b:b+1, vb],      # [1, G_valid, 3]
                        opacities[b:b+1, vb],    # [1, G_valid]
                        semantics[b:b+1, vb],    # [1, G_valid, C]
                        scales[b:b+1, vb],       # [1, G_valid, 3]
                        covinv33[b:b+1, vb],     # [1, G_valid, 3, 3]
                    )  # [N, C]
                    logits_accum_NC = logits_accum_NC + out_nc.to(torch.float32)

                # Convert [N, C] to [1, C, N] and append
                logits_bcN = logits_accum_NC.t().contiguous()[None, ...]  # [1, C, N]
                logits_list.append(logits_bcN)

        # Stack per-batch results: [B, C, N]
        logits_bCN = torch.cat(logits_list, dim=0)
        logits_3d  = logits_bCN.view(B, C, self.H, self.W, self.D)

        # Return both logits and gating information for downstream analysis / hooks
        return {
            "logits_3d": logits_3d,
            "alpha_eff": alpha_eff.detach(),            # [B, G]
            "valid_hard": valid_hard.detach(),          # [B, G]
            "valid_soft": valid_soft.detach(),          # [B, G]
            "drop_mask_soft": drop_mask_soft.detach(),  # [B, G]
        }