# localagg_tr/hooks/fair_compare.py

import os
import pickle
import torch
import numpy as np
from mmengine.hooks import Hook
from mmdet3d.registry import HOOKS


@HOOKS.register_module()
class DumpPredictFairCompareHook(Hook):
    """
    (New) Dump hook dedicated to LocalAgg outputs.

    Role:
      - During test (or val) phase, read the cached tensors that the model's
        forward(mode='predict') stored (see
        localagg_tr.dual_backend_head.cache_predict_outputs) in
          model._last_gaussian_info, model._last_preds_head
        and save **pure LocalAgg outputs and gating information** as a pkl file.

    This hook no longer touches GaussianVoxelizer at all.
    It records only the 3D logits / gate tensors produced by LocalAggWrapper.

    Per-sample pkl structure:

      Basic meta:
        - sample_idx: str  (NuScenes sample token)
        - HWD: [H, W, D]
        - num_classes: int
        - pc_min: (3,) float32
        - pc_max: (3,) float32
        - voxel_size: float
        - grid_shape: [H, W, D]

      LocalAgg 3D output:
        - logits_3d: [C, H, W, D], float32
        - preds_localagg: [H, W, D], int16  (argmax over C)
        - pred_head: [H, W, D], int32  (final prediction from the head, if available)

      Gaussian-level geometry & semantics:
        - means3d:   [G_flat, 3], float32  (originally [N, G, 3] → reshape(-1, 3))
        - scales:    [G_flat, 3], float32
        - covariances: [G_flat, 3, 3], float32
        - semantics: [G_flat, C], float32  (class logits per Gaussian)

      Head-side α/σ debug tensors:
        - alpha_pre:  [G_flat], float32  (raw opacity_head output, before floor/clamp)
        - alpha_post: [G_flat], float32  (after floor + NaN kill + clamp)
        - smax_pre:   [G_flat], float32  (max σ after softplus + lower bound only)
        - smax_post:  [G_flat], float32  (max σ after upper-bound clamp)
        - orig_finite:        [G_flat], bool   (whether original values were finite or NaN/Inf)
        - out_of_range_before:[G_flat], bool   (whether out of pc_range)
        - drop_geo_head:      [G_flat], bool   (Gaussians dropped by head-level geometry gating)

      LocalAgg gate information:
        - alpha_eff:    [G_flat], float32  (effective α used inside LocalAggWrapper)
        - valid_hard:   [G_flat], bool     (hard gate for NaN/Inf/out-of-range)
        - valid_soft:   [G_flat], bool     (soft gate for α/scale based filtering)
        - drop_soft:    [G_flat], bool     (mask of Gaussians attenuated by soft gating)
    """

    def __init__(
        self,
        out_dir: str = "outputs_compare",
        interval: int = 1,
        save_logits: bool = True,    # Always save logits; kept for API compatibility even if False.
        void_id: int = 17,           # No longer used for voxel masking; kept for compatibility.
        max_save=None,               # Upper limit on number of samples to dump.
    ):
        self.out_dir = out_dir
        self.interval = int(interval)
        self.save_logits = bool(save_logits)
        self.void_id = int(void_id)
        self.max_save = max_save
        self._saved = 0

        # Ensure output directory exists.
        os.makedirs(self.out_dir, exist_ok=True)

    @torch.no_grad()
    def after_test_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        # 0) Check whether we already reached the maximum number of dumps.
        if self.max_save is not None and self._saved >= self.max_save:
            return

        # Skip if this iteration does not match the dumping interval.
        if (batch_idx % self.interval) != 0:
            return

        # 1) Handle DDP / non-DDP: unwrap to the actual model.
        model = runner.model.module if hasattr(runner.model, "module") else runner.model

        # 2) Fetch cached tensors saved during forward(mode='predict') (already on CPU).
        ginfo = getattr(model, "_last_gaussian_info", None)
        preds_head = getattr(model, "_last_preds_head", None)

        if ginfo is None:
            return

        # LocalAgg must have written its logits into ginfo.
        if "logits_localagg" not in ginfo:
            # If Head code has not yet populated LocalAggWrapper logits, skip.
            return

        # 3) Get handles to head / LocalAggWrapper (for grid / pc_range metadata).
        head = None
        if hasattr(model, "gauss_heads") and len(model.gauss_heads) > 0:
            head = model.gauss_heads[-1]
        localagg = getattr(head, "voxelizer", None) if head is not None else None
        if localagg is None:
            return

        # Grid / range metadata from LocalAggWrapper.
        try:
            pc_min = localagg.pc_min.detach().cpu().numpy().astype("float32")
            pc_max = localagg.pc_max.detach().cpu().numpy().astype("float32")
            voxel_size = float(localagg.voxel_size.item())
            grid_shape = [int(localagg.H), int(localagg.W), int(localagg.D)]
        except Exception:
            pc_min = None
            pc_max = None
            voxel_size = None
            grid_shape = None

        B = ginfo["logits_localagg"].shape[0]
        data_samples = data_batch["data_samples"]

        for b in range(B):
            # 4) Fetch sample_idx for this element in the batch.
            sample_idx = getattr(
                data_samples[b],
                "sample_idx",
                f"iter{runner.iter}_b{b}",
            )

            # ===== (1) LocalAgg 3D logits and predictions =====
            # ginfo["logits_localagg"]: [B, C, H, W, D] (CPU)
            logits_3d = ginfo["logits_localagg"][b]  # [C, H, W, D], CPU tensor
            C, H, W, D = logits_3d.shape
            probs_3d = logits_3d.softmax(dim=0)      # [C, H, W, D]
            preds_3d = probs_3d.argmax(dim=0)        # [H, W, D]

            out = dict(
                sample_idx=sample_idx,
                HWD=[H, W, D],
                num_classes=int(C),
                logits_3d=logits_3d.numpy().astype("float32"),
                preds_localagg=preds_3d.numpy().astype("int16"),
            )

            # Optionally store ground-truth occupancy if provided.
            try:
                gt = data_samples[b].gt_pts_seg.semantic_seg  # [1, H, W, D] or [H, W, D]
                gt = gt.squeeze(0) if gt.dim() == 4 else gt   # Normalize to [H, W, D]
                out["gt_occ"] = gt.cpu().numpy().astype("int16")
            except AttributeError:
                # Dataloader may not provide GT in pure test mode; ignore if missing.
                pass

            # Optionally store density proxy from LocalAgg.
            if "density_proxy" in ginfo:
                try:
                    density_la = (
                        ginfo["density_proxy"][b]
                        .cpu()
                        .numpy()
                        .astype("float32")
                    )  # [H, W, D]
                    out["density_localagg"] = density_la
                except Exception:
                    pass

            # Attach grid / range metadata if available.
            if pc_min is not None:
                out["pc_min"] = pc_min
                out["pc_max"] = pc_max
            if voxel_size is not None:
                out["voxel_size"] = float(voxel_size)
            if grid_shape is not None:
                out["grid_shape"] = list(grid_shape)

            # Final head prediction (the one fed into OccMetric).
            if preds_head is not None:
                try:
                    out["pred_head"] = (
                        preds_head[b]
                        .cpu()
                        .numpy()
                        .astype("int32")
                    )
                except Exception:
                    pass

            # ===== (2) Gaussian-level semantics / geometry =====
            # semantics: [B, N, G, C] → [G_flat, C]
            try:
                sem = (
                    ginfo["semantics"][b]
                    .cpu()
                    .numpy()
                    .astype("float32")
                )
                sem = sem.reshape(-1, sem.shape[-1])  # [G_flat, C]
                out["semantics"] = sem
            except Exception:
                pass

            # means3d: [B, N, G, 3] → [G_flat, 3]
            try:
                means = (
                    ginfo["means3d"][b]
                    .cpu()
                    .numpy()
                    .astype("float32")
                )
                out["means3d"] = means.reshape(-1, 3)
            except Exception:
                pass

            # scales: [B, N, G, 3] → [G_flat, 3]
            try:
                scales = (
                    ginfo["scales"][b]
                    .cpu()
                    .numpy()
                    .astype("float32")
                )
                out["scales"] = scales.reshape(-1, 3)
            except Exception:
                pass

            # Optional raw / intermediate scale tensors from the head.
            for key in ["scale_raw", "scale_u"]:
                if key not in ginfo:
                    continue
                try:
                    arr = (
                        ginfo[key][b]
                        .cpu()
                        .numpy()
                        .astype("float32")
                    )  # [N, G, 3]
                    out[key] = arr.reshape(-1, 3)
                except Exception:
                    pass

            # covariances: [B, N, G, 3, 3] → [G_flat, 3, 3]
            try:
                covs = (
                    ginfo["covariances"][b]
                    .cpu()
                    .numpy()
                    .astype("float32")
                )
                out["covariances"] = covs.reshape(-1, 3, 3)
            except Exception:
                pass

            # ===== (3) Head-side α/σ debug tensors =====
            def _flatten_bng(key, astype="float32", to_1d=True):
                """
                Helper to extract [B, N, G, ...] tensors for a single batch index b
                and flatten over (N, G) dimensions.
                """
                if key not in ginfo:
                    return None
                arr = (
                    ginfo[key][b]
                    .cpu()
                    .numpy()
                    .astype(astype)
                )
                if to_1d:
                    return arr.reshape(-1)
                # For higher-rank tensors, keep the last dims intact.
                extra_shape = arr.shape[-(arr.ndim - 2):]
                return arr.reshape(-1, *extra_shape)

            alpha_pre = _flatten_bng("alpha_pre")
            if alpha_pre is not None:
                out["alpha_pre"] = alpha_pre

            # α after floor + NaN kill + clamp (i.e., final opacities used by the head).
            try:
                alpha_post = (
                    ginfo["opacities"][b]
                    .cpu()
                    .numpy()
                    .astype("float32")
                    .reshape(-1)
                )
                out["alpha_post"] = alpha_post
            except Exception:
                pass

            smax_pre = _flatten_bng("smax_pre")
            if smax_pre is not None:
                out["smax_pre"] = smax_pre

            smax_post = _flatten_bng("smax_post")
            if smax_post is not None:
                out["smax_post"] = smax_post

            orig_finite = _flatten_bng("orig_finite", astype="bool")
            if orig_finite is not None:
                out["orig_finite"] = orig_finite

            oor_before = _flatten_bng("out_of_range_before", astype="bool")
            if oor_before is not None:
                out["out_of_range_before"] = oor_before

            drop_geo_head = _flatten_bng("drop_geo_head", astype="bool")
            if drop_geo_head is not None:
                out["drop_geo_head"] = drop_geo_head

            # ===== (4) LocalAggWrapper gate information (alpha_eff / valid_* / drop_soft) =====
            def _flat_bg(key, astype, name_out):
                """
                Helper for LocalAggWrapper tensors [B, N, G] or [B, G_flat]:
                extract batch b, cast, and flatten to [G_flat].
                """
                if key not in ginfo:
                    return
                arr = (
                    ginfo[key][b]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(astype)
                )
                out[name_out] = arr.reshape(-1)

            _flat_bg("alpha_eff",   "float32", "alpha_eff")
            _flat_bg("valid_hard",  "bool",    "valid_hard")
            _flat_bg("valid_soft",  "bool",    "valid_soft")
            _flat_bg("drop_mask_soft", "bool", "drop_soft")

            # ===== (5) Save to pkl =====
            save_path = os.path.join(self.out_dir, f"{sample_idx}.pkl")
            with open(save_path, "wb") as f:
                pickle.dump(out, f)

            self._saved += 1
            if self.max_save is not None and self._saved >= self.max_save:
                # Stop dumping further samples after this batch.
                break