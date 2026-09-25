# localagg_tr/gaussian_voxelizer_compat.py

import torch
from mmdet3d.registry import MODELS

# Upstream GaussTR (https://github.com/hustvl/GaussTR, MIT). Must be importable.
from gausstr.models.gaussian_voxelizer import GaussianVoxelizer


@MODELS.register_module()
class GaussianVoxelizerCompat(GaussianVoxelizer):
    """
    Upstream GaussTR GaussianVoxelizer with the same boundary interface as
    LocalAggWrapper, so the head can treat both backends the same way.

    Voxelization itself is unchanged; only these members are added / changed:
      - voxel_size: float attribute -> float32 buffer
      - pc_min / pc_max buffers derived from vol_range
      - use_nextafter / eps_base / eps_ratio (boundary policy, epsilon mode)
    """

    def __init__(self, vol_range, voxel_size, **kwargs):
        # 1) Build the upstream voxelizer (grid_shape / grid_coords / filters).
        voxel_size = float(voxel_size)
        super().__init__(vol_range, voxel_size, **kwargs)

        # 2) Store voxel_size as a buffer (float32), same style as LocalAggWrapper.
        #    This makes it easy for Heads / hooks to query a unified "voxel_size"
        #    field across both GaussVoxelizer and LocalAggWrapper.
        del self.voxel_size
        self.register_buffer(
            "voxel_size",
            torch.tensor(voxel_size, dtype=torch.float32),
            persistent=False,
        )

        # 3) Store vol_range as float32 and derive pc_min / pc_max buffers.
        #    - vol_range: [x_min, y_min, z_min, x_max, y_max, z_max]
        #    - pc_min / pc_max are kept explicitly to mirror LocalAggWrapper
        #      and to allow shared boundary / pc_range logic in the Head.
        vol_range = torch.tensor(vol_range, dtype=torch.float32)
        self.vol_range = vol_range
        self.register_buffer("pc_min", vol_range[:3])
        self.register_buffer("pc_max", vol_range[3:])

        # 4) Default boundary-policy-related members that the head expects.
        #    These fields are added so that GaussVoxelizer exposes an interface
        #    compatible with LocalAggWrapper (use_nextafter / eps_base / eps_ratio),
        #    even though GaussVoxelizer itself does not actively use them yet.
        self.use_nextafter = False   # Heads treat this as "epsilon mode".
        self.eps_base = 1e-6
        self.eps_ratio = 1e-3
