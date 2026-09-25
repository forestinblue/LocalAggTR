# RESULTS.md — CV-ready bullets

Every number below is copied from [`results/NUMBERS.md`](results/NUMBERS.md). The conditions are stated in the same bullet.
The headline is speed at comparable accuracy (owner decision 2026-09-25, option (b)).

## Bullets

1. Integrated GaussianFormer's Local Aggregation CUDA op into GaussTR as a config-switchable voxelization backend (adapter, Gaussian sanitization, density-quantile free-space rule). End-to-end test time went from 4.0872 to 0.5680 s/sample (**7.2× faster**) at mIoU 0.1107 → 0.1142 (+3.2%). *(nuScenes val, same checkpoint and head, only the voxelizer swapped; 1 run each, RTX 2080 Ti, batch 1)*

2. Measured the accuracy cost of the head's test-time Gaussian sanitization. Without retraining, it lowered the original GaussTR voxelizer from 0.4563 to 0.2521 IoU (−44.8%); against that original setting, Local Aggregation is 6.1× faster at 0.3164 IoU. *(nuScenes val, same checkpoint, 1 run each, RTX 2080 Ti)*

## Traceability

| Number in bullets | Row in `results/NUMBERS.md` |
|---|---|
| 4.0872 s, 0.5680 s, 7.2× (7.20×) | Inference time table: `dec_gaussvox`, `dec_localagg`; Relative changes: controlled pair, Time (median) |
| mIoU 0.1107, 0.1142, +3.2% | Accuracy table: `dec_gaussvox`, `dec_localagg`; Relative changes: controlled pair |
| IoU 0.4563, 0.2521, −44.8% | Accuracy table: `aug_gaussvox_orighead`, `dec_gaussvox`; Relative changes: third row |
| 6.1× (6.11×), IoU 0.3164 | Relative changes: second row; Accuracy table: `dec_localagg` |

## Rules when copying to a CV

- Keep the condition in brackets. It can be shortened to "(nuScenes val, same checkpoint, 1× RTX 2080 Ti)", but do not drop it.
- "7.2× faster" is the time of the whole test loop, not of the voxelizer alone. Do not write "7× faster voxelization".
- **Do not use "+25% IoU"** (0.3164 vs 0.2521). It is only true against the modified-head baseline. Against the original GaussTR voxelizer, IoU is −30.7%, and the final presentation (slide 8) uses the original as its baseline.
- Do not claim training or fine-tuning results. All runs are test-time only.
- Contribution scope: the head module. The lab pipeline, training and checkpoint are Zhang Chi's project, and the CUDA op belongs to GaussianFormer / Inria.
