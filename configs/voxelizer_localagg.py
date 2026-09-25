# Voxelizer config fragment: LocalAgg backend.
# Merge into the head config as `voxelizer=voxelizer` and add
# custom_imports = dict(imports=['gausstr', 'localagg_tr']).
# Values are the ones used in the experiments (from the original training config).

volume_range = [-40, -40, -1, 40, 40, 5.4]
voxel_size = 0.4
H = int((volume_range[3] - volume_range[0]) / voxel_size)  # 200
W = int((volume_range[4] - volume_range[1]) / voxel_size)  # 200
D = int((volume_range[5] - volume_range[2]) / voxel_size)  # 16
num_class_ = 18

voxelizer = dict(
    type='LocalAggWrapper',   # Only this line needs to be changed for the core swap
    pc_range=volume_range,    # = volume_range
    voxel_size=voxel_size,    # = 0.4
    grid_shape=(H, W, D),     # Precomputed (200,200,16)
    num_classes=num_class_,
    force_fp32=True,          # FP32 is recommended for the CUDA kernel even with AMP
    scale_multiplier=3.0,
    chunk_size=512,
    strict_check=False,       # Default monitoring mode
    log_gate_ratio=True,      # Log drop ratio
    fail_soft_when_all_drop=True,  # Fail-soft with zero-logits if all invalid
    use_nextafter=True,       # Default: enforce < upper-bound using ULP-safe nextafter
    eps_base=1e-6,            # Used only if nextafter is unavailable
    eps_ratio=1e-3)           # Margin proportional to voxel_size
