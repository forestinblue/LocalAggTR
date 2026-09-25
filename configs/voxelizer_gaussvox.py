# Voxelizer config fragment: GaussTR Gaussian voxelizer backend (baseline).
# Merge into the head config as `voxelizer=voxelizer` and add
# custom_imports = dict(imports=['gausstr', 'localagg_tr']).
# Values are the ones used in the experiments (from the original training config).
# GaussianVoxelizerCompat = upstream GaussianVoxelizer + pc_min/pc_max interface.

volume_range = [-40, -40, -1, 40, 40, 5.4]

voxelizer = dict(
    type='GaussianVoxelizerCompat',
    vol_range=volume_range,
    voxel_size=0.4,
    filter_gaussians=True,
    opacity_thresh=0.6,
    covariance_thresh=1.5e-2)
