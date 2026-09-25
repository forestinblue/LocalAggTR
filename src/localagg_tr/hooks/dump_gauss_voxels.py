# localagg_tr/hooks/dump_gauss_voxels.py

import os
import pickle
from typing import Optional

import numpy as np
import torch
from mmengine.hooks import Hook
from mmdet3d.registry import HOOKS


@HOOKS.register_module()
class DumpGaussVoxelizerHook(Hook):
    """
    (New) Dump hook dedicated to GaussVoxelizer outputs only.

    Purpose:
      - During test/validation, dump ONLY the GaussVoxelizer results
        that the model cached in forward(mode='predict')
        (see localagg_tr.dual_backend_head.cache_predict_outputs) into:
            model._last_gaussian_info,
            model._last_preds_head
      - This hook does NOT touch LocalAggWrapper at all.
      - Results are saved into a separate output directory as pkl files.

    Expected per-sample pkl structure:

      Basic metadata:
        - sample_idx      : str
        - HWD             : [H, W, D]
        - num_classes     : int (based on logits_gaussvox)
        - voxel_size      : float (if readable from head.voxelizer)

      GaussVoxelizer outputs:
        - preds_gaussvox  : [H, W, D] int16      (final head prediction based on GaussVoxelizer)
        - density_gaussvox: [H, W, D] float32
        - logits_gaussvox : [C, H, W, D] float32  (only if save_logits=True)

      (Optional) Ground truth:
        - gt_occ          : [H, W, D] int16       (if provided by the test dataloader)
    """

    def __init__(
        self,
        out_dir: str = "outputs_gaussvox",
        interval: int = 1,
        save_logits: bool = True,
        max_save: Optional[int] = None,
    ):
        """
        Args:
            out_dir (str):
                Directory where pkl files will be saved.

            interval (int):
                Dump results every `interval` test iterations.
                (Only when batch_idx % interval == 0)

            save_logits (bool):
                Whether to also save logits_gaussvox.
                WARNING: This can consume large disk space.

            max_save (Optional[int]):
                Maximum number of samples to save.
                If None, no upper limit is applied.
        """
        self.out_dir = out_dir
        self.interval = int(interval)
        self.save_logits = bool(save_logits)
        self.max_save = max_save

        self._saved = 0
        os.makedirs(self.out_dir, exist_ok=True)

    @torch.no_grad()
    def after_test_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        # --------------------------------------------
        # (0) Check maximum save limit
        # --------------------------------------------
        if self.max_save is not None and self._saved >= self.max_save:
            return

        # --------------------------------------------
        # (1) Interval-based skipping
        # --------------------------------------------
        if (batch_idx % self.interval) != 0:
            return

        # --------------------------------------------
        # (2) Handle DDP / non-DDP model access
        # --------------------------------------------
        model = runner.model.module if hasattr(runner.model, "module") else runner.model

        # --------------------------------------------
        # (3) Fetch cached prediction tensors
        #     These must be populated inside forward(mode='predict')
        # --------------------------------------------
        ginfo = getattr(model, "_last_gaussian_info", None)
        preds_head = getattr(model, "_last_preds_head", None)

        if ginfo is None or preds_head is None:
            return

        # --------------------------------------------
        # (4) Ensure GaussVoxelizer path was actually used
        #     (skip LocalAgg-only experiments)
        # --------------------------------------------
        if ("density_gaussvox" not in ginfo) or ("logits_gaussvox" not in ginfo):
            return

        # --------------------------------------------
        # (5) Try to read voxel_size from head.voxelizer
        #     This is IMPORTANT for reproducibility.
        # --------------------------------------------
        voxel_size = None
        head = None
        if hasattr(model, "gauss_heads") and len(model.gauss_heads) > 0:
            head = model.gauss_heads[-1]

        if head is not None:
            vox = getattr(head, "voxelizer", None)
            if vox is not None and hasattr(vox, "voxel_size"):
                try:
                    voxel_size = float(vox.voxel_size.item())
                except Exception:
                    # voxel_size may already be a float
                    try:
                        voxel_size = float(vox.voxel_size)
                    except Exception:
                        voxel_size = None

        # --------------------------------------------
        # (6) Extract GaussVoxelizer tensors
        # --------------------------------------------
        density = ginfo["density_gaussvox"]   # [B, H, W, D]
        logits = ginfo["logits_gaussvox"]     # [B, C, H, W, D]
        preds = preds_head                   # [B, H, W, D]

        B = preds.shape[0]

        data_samples = data_batch.get("data_samples", None)
        if data_samples is None:
            # MMDet3D standard dataloader should always provide this.
            return

        for b in range(B):
            # --------------------------------------------
            # (7) Sample identifier (e.g., NuScenes token)
            # --------------------------------------------
            sample_idx = getattr(
                data_samples[b],
                "sample_idx",
                f"iter{runner.iter}_b{b}",
            )

            # --------------------------------------------
            # (8) Move tensors to CPU and convert to numpy
            # --------------------------------------------
            preds_b = (
                preds[b]
                .detach()
                .cpu()
                .numpy()
                .astype("int16")
            )  # [H, W, D]

            dens_b = (
                density[b]
                .detach()
                .cpu()
                .numpy()
                .astype("float32")
            )  # [H, W, D]

            logits_b = (
                logits[b]
                .detach()
                .cpu()
                .numpy()
                .astype("float32")
            )  # [C, H, W, D]

            C, H, W, D = logits_b.shape

            # --------------------------------------------
            # (9) Construct output dictionary
            # --------------------------------------------
            out = dict(
                sample_idx=sample_idx,
                HWD=[H, W, D],
                num_classes=int(C),
                preds_gaussvox=preds_b,
                density_gaussvox=dens_b,
            )

            if self.save_logits:
                out["logits_gaussvox"] = logits_b

            if voxel_size is not None:
                out["voxel_size"] = float(voxel_size)

            # --------------------------------------------
            # (10) Optional: save GT occupancy if available
            # --------------------------------------------
            gt_occ = None
            try:
                gt = data_samples[b].gt_pts_seg.semantic_seg  # [1, H, W, D] or [H, W, D]
                if gt is not None:
                    gt = gt.squeeze(0) if gt.dim() == 4 else gt
                    gt_occ = (
                        gt
                        .detach()
                        .cpu()
                        .numpy()
                        .astype("int16")
                    )
            except AttributeError:
                gt_occ = None

            if gt_occ is not None:
                out["gt_occ"] = gt_occ

            # --------------------------------------------
            # (11) Save pkl file
            # --------------------------------------------
            save_path = os.path.join(self.out_dir, f"{sample_idx}.pkl")
            with open(save_path, "wb") as f:
                pickle.dump(out, f)

            self._saved += 1
            if self.max_save is not None and self._saved >= self.max_save:
                break