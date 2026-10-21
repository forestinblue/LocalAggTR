# Figure 1 (hero) — spec

## What the figure claims
The same GaussTR checkpoint produces 3D Gaussians. One voxelization block is swapped by a config flag: GaussTR's heuristic voxelizer (baseline, non-differentiable) or GaussianFormer's Local Aggregation CUDA op (ours, differentiable). This changes nothing upstream. On nuScenes val, the full test loop drops from 4.09 to 0.57 s/sample (7.2×) at similar mIoU (0.111 → 0.114). The swap also opens a gradient path from a 3D occupancy loss back to the Gaussians. **That path was never used for training in this work**, so the loss box and return arrow show only what is *possible*.

## Numbers used (must equal README "Key result", column 2 vs column 3, rounded)
| Figure | README | Source |
|---|---|---|
| mIoU 0.111 → 0.114 | 0.1107 → 0.1142 | `results/NUMBERS.md` (`dec_gaussvox`, `dec_localagg`) |
| 4.09 → 0.57 s/sample | 4.0872 → 0.5680 | `results/NUMBERS.md`, median over 30 progress lines |
| 7.2× | 7.20× | README key-result row |

The speed appears only in the results box, as "end-to-end inference" (README L14: whole test loop, not the voxelizer alone). The voxelizer boxes carry no timing. The figure has no footnote, so the Figure 1 caption in README must state the conditions: nuScenes val 6019, same checkpoint and head, 1 run each, 1× RTX 2080 Ti.

## Provenance of the existing occupancy renders (checked 2026-09-26)
- `600_sem_localagg.png` has the in-image title "600 | Head (student)". In `Modified_GaussTR/tools/visualize_localagg_voxels.py:383`, that title is produced only by `mode == "head"`. That mode draws `pred_head` and saves `{sid}_head.png`. No code writes `*_sem_localagg.png`, so the file was renamed by hand. → **The render is `pred_head`, not `preds_localagg`.** (Confirmed from the code.)
- `pred_head` = `model._last_preds_head` (`gausstr/models/weak_gausstr.py:298`), which is what the head returns in predict mode and what the metric evaluates. With `type='LocalAggWrapper'` (`local_aggregation_test.log:221`), the head dispatches to `_forward_predict_localagg` (`weak_gausstr_head.py:341`). There it takes the argmax of the density-normalized LocalAgg logits, then applies τ-quantile free masking (`tau_quantile=0.9`, log L196). `preds_localagg` is the same argmax **without** the free mask.
- The baseline `preds_gaussvox` is also `_last_preds_head` (`dump_gauss_voxels.py:138`). → Both renders are the head's final prediction under each backend: a like-for-like comparison, and the output behind the reported mIoU.
- Caveat (LocalAgg side, **unresolved**): the local copies of `dump_localagg_result.py` (Dec 4 20:24) and `weak_gausstr_head.py` (Dec 4 19:27) were modified after the LocalAgg run started (2025-12-04 07:43, time zone unknown). "`pred_head` = LocalAgg + τ free mask" holds for the local code only. The LocalAgg dump (`outputs_compare/600.pkl`) no longer exists on the server (checked 2026-09-26), so this cannot be verified, and the user decided not to re-run.

### Checks on the baseline dump (2026-09-26, `scripts/render_fig_assets.py check`)
Source: `labserver:/data/zhangchi/gausstr_shared/outputs_gaussvox/600.pkl` (5,120,408 B, Dec 10 2025, same date as `gauss_voxel.log`) → `raw/dumps/600_gaussvox.pkl`.
- Keys: `HWD, density_gaussvox, gt_occ, num_classes (18), preds_gaussvox, sample_idx (600), voxel_size (0.4)`. **No Gaussian tensors.**
- `gt_occ` == Occ3D `labels.npz['semantics']` of token `fd534bf0c87d4d4cb891f0fc6f41a6a9` (scene-0095): **identical, 0 / 640,000 voxels differ.** So the camera images (looked up by sample_idx 600 in `/data/zhangchi/GaussTR/nuscenes_infos_val.pkl`: 6019 entries, list index 600) are the same sample as the dump.
- Baseline free mask (`_forward_predict_gaussvox`: density > 4e-2, else 17): all 610,017 voxels with density ≤ 4e-2 are 17. Of the 29,983 above the threshold, 29,673 are non-free (the remaining 310 have argmax 17). This matches the local head code, so `preds_gaussvox` is the head's final baseline prediction.
- Re-render: `Modified_GaussTR/tools/visualize_gauss_voxels.py --mode all` on this pkl (matplotlib 3.11.2; plot function identical to `scripts/visualize_gauss_voxels.py`). By eye it shows the same structure, class layout and outliers as `600_sem_gaussvox.png`. It is **not pixel-identical**: 1 px height difference and different marker rasterization. Even GT vs GT, which come from identical data, reaches only 0.33 on the class-pixel overlap measure. The new render's best match among the old renders is `600_sem_gaussvox.png` (0.55, vs 0.45 `600_sem_localagg.png`, 0.19 `600_gt.png`). → Consistent with the baseline image coming from this dump; not a byte-level proof.

## Filled (2026-09-26)
| Cell id | Content | Made from |
|---|---|---|
| `cam_*` (6) | `cam_CAM_*.jpg`, 480×270 (embedded) | nuScenes `samples/CAM_*/n008-2018-08-01-15-16-36-0400__CAM_*__15331512501*.jpg`, sample_idx 600; downscaled only (`render_fig_assets.py cams`) |
| `gaussians` | plain box "3D Gaussians" | **no image**: no Gaussian tensors in any surviving dump (user decision 2026-09-26) |
| `occ_base` | `occ_baseline.png` (embedded) | `Zhang Chi_Project/Comparison_LocalAgg_and_GaussVoxel/600_sem_gaussvox.png` = `preds_gaussvox` |
| `occ_ours` | `occ_ours.png` (embedded) | `.../600_sem_localagg.png` = `pred_head` (renamed `600_head.png`) |

Occupancy thumbnails come from `scripts/render_fig_assets.py occ`: **crop only, no re-render**. One shared box for both images, in original-render pixels (x0, y0, x1, y1) = (276, 310, 1019, 992). It is the union of both images' coloured scene pixels plus 12 px padding, grown to the 230:211 cell aspect and slid up to stay above the black axis line (first at y=996). Title, tick labels, axis lines and outer whitespace are outside the box. The grey 3D panes and grid behind the scene remain, since they can't be separated without re-rendering. Output is 720×661 each, filling the 230×211 cell exactly.

**License note:** the camera thumbnails are nuScenes images (CC BY-NC-SA 4.0). The README caption must credit nuScenes, and the figure file inherits NC-SA terms.

## Fetch raw inputs (updated 2026-09-26; original `/data/jun/Modified_GaussTR` is deleted)
Server `labserver` (~/.ssh/config). Sources are in Zhang Chi's copy, **read and copy only**: nothing on the server is written, moved or deleted. Remote Python runs from `~` with `PYTHONDONTWRITEBYTECODE=1`, reading files only.
Run from the repo root, block by block.

```bash
SERVER=labserver
Z=/data/zhangchi/gausstr_shared
mkdir -p assets/fig_src/raw/{dumps,cams,gt}

# A) look first (read-only)
ssh $SERVER bash -s <<'EOF'
Z=/data/zhangchi/gausstr_shared
echo "## vis3d_600"; ls -la $Z/work_dirs/outputs_compare/vis3d_600/; du -sh $Z/work_dirs/outputs_compare/vis3d_600
echo "## dumps";     ls -la $Z/outputs_gaussvox/ | head -20; ls -la $Z/work_dirs/outputs_compare/ | head -30
echo "## ckpt";      find $Z/work_dirs -maxdepth 3 -name 'iter_56272.pth' -exec ls -la --time-style=full-iso {} \;
echo "## infos";     ls -la /data/zhangchi/GaussTR/nuscenes_infos_val.pkl
echo "## nuscenes";  ls /data/nuscenes | head -30
EOF

# B) which Python envs exist (need numpy for infos; torch + mmengine for the checkpoint)
ssh $SERVER bash -s <<'EOF'
for p in /data/jun/miniconda3/envs/*/bin/python /data/zhangchi/miniconda3/envs/*/bin/python ~/miniconda3/envs/*/bin/python ~/anaconda3/envs/*/bin/python $(which -a python3 python 2>/dev/null); do
  [ -x "$p" ] || continue
  echo "$p  $("$p" -c 'import sys,importlib.util as u; print(sys.version.split()[0], [m for m in ("numpy","torch","mmengine") if u.find_spec(m)])' 2>&1 | tail -1)"
done
EOF
PY=<path from B with numpy, torch, mmengine>

# C) baseline dump + vis3d_600 (check the size from A first)
scp "$SERVER:$Z/outputs_gaussvox/600.pkl" assets/fig_src/raw/dumps/600_gaussvox.pkl
scp -r "$SERVER:$Z/work_dirs/outputs_compare/vis3d_600" assets/fig_src/raw/

# D) sample_idx 600 -> token, scene, 6 camera files (read-only)
ssh $SERVER "PY=$PY bash -s" > assets/fig_src/raw/cams/sample600.txt <<'EOF'
cd ~ && PYTHONDONTWRITEBYTECODE=1 "$PY" - <<'PYEOF'
import pickle, sys
dl = pickle.load(open('/data/zhangchi/GaussTR/nuscenes_infos_val.pkl', 'rb'))['data_list']
e = next((x for x in dl if str(x.get('sample_idx')) == '600'), None)
if e is None: sys.exit('sample_idx 600 not found')
print('#', 'len', len(dl), 'list_index', dl.index(e), 'token', e['token'], 'scene_idx', e.get('scene_idx'))
for cam, v in e['images'].items():
    print(cam, '/data/nuscenes/samples/%s/%s' % (cam, v['img_path'].split('/')[-1]))
PYEOF
EOF
cat assets/fig_src/raw/cams/sample600.txt

# E) 6 camera images
grep -v '^#' assets/fig_src/raw/cams/sample600.txt | while read -r cam p; do scp "$SERVER:$p" "assets/fig_src/raw/cams/$cam.jpg" < /dev/null; done

# F) GT of that sample, to prove cameras == dump sample (gt_occ in 600.pkl). Adjust if gts/ is elsewhere (see A).
read TOKEN SCENE <<< "$(awk '/^#/{print $7, $9}' assets/fig_src/raw/cams/sample600.txt)"
scp "$SERVER:/data/nuscenes/gts/$SCENE/$TOKEN/labels.npz" assets/fig_src/raw/gt/

# G) checkpoint fingerprint (read-only; prints only)
ssh $SERVER "PY=$PY bash -s" > assets/fig_src/raw/ckpt_check.txt <<'EOF'
find /data/zhangchi/gausstr_shared/work_dirs -maxdepth 3 -name 'iter_56272.pth' -exec sha256sum {} \;
cd ~ && PYTHONDONTWRITEBYTECODE=1 "$PY" - <<'PYEOF'
import torch
c = torch.load('/data/zhangchi/gausstr_shared/work_dirs/baseline/iter_56272.pth', map_location='cpu')
m = c.get('meta', {})
for k in ('epoch', 'iter', 'time', 'experiment_name', 'mmengine_version', 'seed'):
    print('meta', k, m.get(k))
cfg = str(m.get('cfg', ''))
print('cfg', [l.strip() for l in cfg.splitlines() if any(s in l for s in ('work_dir', 'load_from', '/data/'))][:10])
sd = c['state_dict']
print('n_keys', len(sd))
for k in ('gauss_heads.0.voxelizer.vol_range', 'gauss_heads.0.voxelizer.grid_coords',
          'gauss_heads.0.s_max_xyz', 'gauss_heads.0.voxelizer.pc_min', 'gauss_heads.0.voxelizer.aggregator.pc_min'):
    print('has', k, k in sd)
PYEOF
EOF
cat assets/fig_src/raw/ckpt_check.txt
```

## Layout for README legibility (rebuilt 2026-09-26)
GitHub shows the README image at about 880 px wide. The canvas is **1400 × 640** (content about 1376 px), so it appears at **0.627×**.
- Fonts (Helvetica only): box titles 18 px (→ 11.3 px), sub-labels, legend, camera caption and arrow labels 15 px (→ **9.4 px, the smallest**), occupancy labels and results box 16 px (→ 10.0 px), "×" 30 px. The requested 13–14 px minimum would appear at 8.2–8.8 px (< 9), so it was raised to 15 px.
- Lines: box borders 2 px; arrows 2.5 px, arrowhead `endSize=12`; dashed baseline `6 4`.
- Boxes: main boxes 64–84 px high, `spacing=10`. The one exception is the `config flag` tag (110×34), a label-sized switch rather than a pipeline box.
- Size order: occupancy images 230×211 (largest) > voxelizer 210×64 > loss 160×84 > encoder 140×70 > Gaussians 130×70 > cameras 68×38 each in a 2×3 grid.
- Preview: `drawio -x -f png -s 2 -b 20` (draw.io desktop 31.5.3), downscaled to 880 px → `preview_880.png` (880×428).
- Layout generator (not committed): scratchpad `build_hero_1400.py`. From now on, edit `hero.drawio` by hand.

## Text moved out of the figure → must be in the README caption
- baseline = GaussTR's heuristic voxelizer; ours = GaussianFormer's Local Aggregation CUDA op (third-party, not mine), selected by a config flag (`voxelizer = LocalAggWrapper`).
- Same checkpoint and same head for both paths; only the voxelizer differs. nuScenes val sample_idx 600.
- Numbers: nuScenes val (6019), 1 run each, 1× RTX 2080 Ti; time = end-to-end test loop per sample (median).
- The gradient path is enabled only: no 3D-loss training was run.
- Camera images: nuScenes, CC BY-NC-SA 4.0.

## Adjust by hand in draw.io
- The empty lower-left area (below the cameras, left of the legend) can be tightened by moving the legend up or narrowing the canvas.
- The occupancy cells show the scene edge to edge (grey panes behind it are part of the render). If you resize a cell, keep the 230:211 aspect (`aspect=fixed`).
- After any move, re-export and check nothing falls below 15 px in draw.io (9.4 px at README width).

## Not claimed / not drawn
- No training with a 3D loss (loss box says "not trained here").
- Speed is end-to-end (whole test loop), shown only in the results box, never attributed to the voxelizer.
- README column 1 (head as received, mIoU 0.1352) is higher than ours. The figure compares only the same-head setting (column 2 vs 3), and the caption must say "same head".
