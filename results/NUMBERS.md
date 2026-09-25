# NUMBERS.md — every number, with its source

All numbers below are copied from `results/logs/*.txt`. Those files are verbatim line excerpts of the
raw mmengine test logs, made by `scripts/extract_log_excerpts.py`. The only edit is that server paths are replaced by `<server>/`. Each excerpt line carries its
line number in the raw log (`L<n>:`).
The raw logs are kept offline in `Zhang Chi_Project/`. They are not published because they contain the full lab training config.

## Common conditions (all 5 runs)

| Condition | Value | Source |
|---|---|---|
| Split | nuScenes val, 6019 samples, `ann_file='nuscenes_infos_val.pkl'` | every excerpt, `ann_file` and `Epoch(test) [6019/6019]` lines |
| Metric | `OccMetric`, 18 classes, `use_image_mask=True` (camera-visible voxels) | every excerpt, `use_image_mask=` line |
| `iou` | binary occupancy IoU (geometry); `miou` = mean over semantic classes | final table in each excerpt |
| Checkpoint | same for all runs: `work_dirs/baseline/iter_56272.pth` (lab weak-supervision model; not released) | `Load checkpoint` line |
| Training | **none of these runs trains anything**: each is test-time only, with a swapped voxelizer/head path on the same checkpoint | `Testing command` line |
| Hardware | 1× NVIDIA GeForce RTX 2080 Ti, PyTorch 2.1.2+cu118, batch size 1 | `GPU 0:` / `PyTorch:` lines |
| #runs | 1 run per setting (no repeats, no variance estimate) | — |

## Runs

| Run id | Date | Voxelizer | Head code version | Key head settings | Excerpt |
|---|---|---|---|---|---|
| `aug_gaussvox_orighead` | 2025-08-26 | `GaussianVoxelizer` | as received from the lab (before owner's changes) | — | `results/logs/aug_gaussvox_orighead.txt` |
| `oct_localagg_v1` | 2025-10-19 | `LocalAggWrapper` | owner's, early | no `tau_quantile` in config | `results/logs/oct_localagg_v1.txt` |
| `oct_localagg_v2` | 2025-10-20 | `LocalAggWrapper` | owner's, early | `tau_quantile=0.2` | `results/logs/oct_localagg_v2.txt` |
| `dec_localagg` | 2025-12-04 | `LocalAggWrapper` | owner's, final (= released code) | `tau_quantile=0.9`, `s_max_xyz=(0.26,0.26,0.26)` | `results/logs/dec_localagg.txt` |
| `dec_gaussvox` | 2025-12-10 | `GaussianVoxelizer` | owner's, final (= released code) | `tau_quantile=0.9`, `s_max_xyz=(0.26,0.26,0.26)` | `results/logs/dec_gaussvox.txt` |

`dec_localagg` and `dec_gaussvox` are the controlled pair: same checkpoint, same head code, same config except the
voxelizer block and the dump hook (config dumps diffed 2026-09-25).

## Accuracy (absolute, from the final table of each excerpt)

| Run id | IoU | mIoU | Source line |
|---|---|---|---|
| `aug_gaussvox_orighead` | 0.4563 | 0.1352 | `aug_gaussvox_orighead.txt` L712 |
| `oct_localagg_v1` | 0.2261 | 0.0652 | `oct_localagg_v1.txt` L728 |
| `oct_localagg_v2` | 0.2255 | 0.0659 | `oct_localagg_v2.txt` L6748 |
| `dec_localagg` | 0.3164 | 0.1142 | `dec_localagg.txt` L789 |
| `dec_gaussvox` | 0.2521 | 0.1107 | `dec_gaussvox.txt` L768 |

Per-class IoU for every run is in the same table row.

## Inference time

Measured over the whole test loop: backbone, decoder, head, voxelizer, metric and dump hooks, at batch size 1.
**This is not the voxelizer's time alone.** It is also not a dedicated benchmark: one run each, on a shared server, with no warm-up control.

| Run id | Median `time:` per iter (s) | Wall clock, checkpoint load → last iter (s) | Wall / 6019 (s/sample) |
|---|---|---|---|
| `aug_gaussvox_orighead` | 3.4687 | 20797 | 3.455 |
| `oct_localagg_v1` | 0.5583 | 3363 | 0.559 |
| `oct_localagg_v2` | 0.6116 | 3683 | 0.612 |
| `dec_localagg` | 0.5680 | 3559 | 0.591 |
| `dec_gaussvox` | 4.0872 | 25148 | 4.178 |

- The median is taken over the 30 `Epoch(test) [k/6019]` progress lines, k = 200…6000.
- Wall clock is the timestamp of the `Load checkpoint` line to the timestamp of the `Epoch(test) [6019/6019]` line.
- These values come from `python scripts/extract_log_excerpts.py`.

## Relative changes (derived from the tables above)

| Comparison | IoU | mIoU | Time (median) | Time (wall) |
|---|---|---|---|---|
| `dec_localagg` vs `dec_gaussvox` (controlled pair) | 0.3164 vs 0.2521 = **+25.5%** | 0.1142 vs 0.1107 = **+3.2%** | 4.0872 / 0.5680 = **7.20× faster** | 25148 / 3559 = **7.07× faster** |
| `dec_localagg` vs `aug_gaussvox_orighead` (original GaussTR voxelizer, head as received) | 0.3164 vs 0.4563 = **−30.7%** | 0.1142 vs 0.1352 = **−15.5%** | 3.4687 / 0.5680 = **6.11× faster** | 20797 / 3559 = **5.84× faster** |
| `dec_gaussvox` vs `aug_gaussvox_orighead` (effect of the owner's head changes on the GaussVoxelizer path) | 0.2521 vs 0.4563 = **−44.8%** | 0.1107 vs 0.1352 = **−18.1%** | — | — |

## CV claim check

| CV claim | Matches a log? | Verdict |
|---|---|---|
| "+25% IoU" | Yes: `dec_localagg` vs `dec_gaussvox` = +25.5% (0.3164 vs 0.2521) | ⚠️ **Only true against the December GaussVoxelizer run, which uses the owner's modified head.** Against the original GaussTR voxelizer with the head as received (`aug_gaussvox_orighead`, 0.4563), LocalAgg is **−30.7% IoU**. The final presentation (`ppt_final_capstone.pptx`, slide 8) itself uses the August numbers (0.1352 / 0.4563) as the "GaussTR baseline". Stating "+25% IoU" without the baseline condition would be misleading. |
| "~6–7× faster" | Yes: 5.84×–7.20× depending on baseline and measure | ✅ Supported, but it is end-to-end test time per sample (whole model, batch 1, 1 run, RTX 2080 Ti), not isolated voxelizer time. |

**Decision (owner, 2026-09-25): option (b).** The CV/README headline is speed at comparable accuracy, not "+25% IoU".
Wording to use: ~7× faster end-to-end inference (7.20× median per-sample, controlled pair) with mIoU 0.1142 vs 0.1107 (+3.2%).
The IoU of both baselines (0.4563 original, 0.2521 same-head) must be reported next to it.

## Observations to keep in mind (not claims)

- In the two December runs, the checkpoint lacks `s_max_xyz` (logged as `missing keys ... s_max_xyz`). The checkpoint was trained before the owner's scale cap/softplus existed. Applying them only at test time changes the Gaussian scales the model was trained with. This is a plausible cause of the −44.8% IoU on the GaussVoxelizer path; it has not been verified.
- Every run logs `OccMetric got empty self.results`. This is the normal behaviour of GaussTR's `OccMetric`, which accumulates a confusion histogram instead of `self.results`; the tables are still computed.
- The October → December LocalAgg changes (0.2255 → 0.3164 IoU) mix config (`tau_quantile` 0.2 → 0.9, `s_max_xyz` added) and code changes, so they are **not** a controlled ablation.
- Presentation slide 8 lists rows for "3D fine-tuning with 30% / 50% labeled frames" with no numbers. That is the semi-supervised follow-up and is out of release scope (CLAUDE.md).
