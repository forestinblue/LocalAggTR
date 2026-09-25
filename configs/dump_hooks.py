# Test-time dump hooks (pick the one matching the voxelizer backend).
# The model must call localagg_tr.cache_predict_outputs() in forward(mode='predict').

# LocalAgg backend -> outputs_compare/*.pkl
# (read by scripts/visualize_localagg_voxels.py, scripts/inspect_outputs_compare.py)
custom_hooks_localagg = [
    dict(
        type='DumpPredictFairCompareHook',
        out_dir='outputs_compare',
        interval=600,
        max_save=10,          # save at most 10 files
        save_logits=True,     # save logits/feats for comparison/debugging
        void_id=17),
]

# GaussVoxelizer backend -> outputs_gaussvox/*.pkl
# (read by scripts/visualize_gauss_voxels.py)
custom_hooks_gaussvox = [
    dict(
        type='DumpGaussVoxelizerHook',
        out_dir='outputs_gaussvox',   # must match visualization script
        interval=600,
        max_save=10,
        save_logits=False),
]
