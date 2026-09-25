# localagg_tr/__init__.py
from .localagg_wrapper import LocalAggWrapper
from .gaussian_voxelizer_compat import GaussianVoxelizerCompat
from .dual_backend_head import DualBackendVoxelHeadMixin, cache_predict_outputs
from . import hooks  # registers dump hooks

__all__ = [
    "LocalAggWrapper",
    "GaussianVoxelizerCompat",
    "DualBackendVoxelHeadMixin",
    "cache_predict_outputs",
]
