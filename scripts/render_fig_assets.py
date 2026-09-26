"""
Thumbnails for the hero figure (assets/fig_src/hero.drawio).

  occ    assets/fig_src/occ_baseline.png, occ_ours.png
         From the existing 3D renders of val sample 600 (same axes, view and palette):
           600_sem_gaussvox.png  = preds_gaussvox  (visualize_gauss_voxels.py, mode semantic)
           600_sem_localagg.png  = pred_head       (visualize_localagg_voxels.py, mode head;
                                                    renamed by hand from 600_head.png)
         Both are the head's final prediction with each backend. Crop only (no re-render): both
         images get the SAME box around the union of their coloured scene pixels, grown to the
         230:211 hero cell, so viewpoint and scale stay matched. Title, tick labels and outer
         whitespace fall outside the box; the grey 3D panes behind the scene remain.

  cams   assets/fig_src/cam_<CAM>.jpg
         The 6 nuScenes camera images of val sample_idx 600 (token fd534bf0c87d4d4cb891f0fc6f41a6a9,
         scene-0095), copied into assets/fig_src/raw/cams/ (git-ignored) and downscaled only.

  check  Provenance checks on assets/fig_src/raw/ (prints only):
         gt_occ in the baseline dump == Occ3D labels.npz of that token (cameras = dump sample),
         and the baseline free mask (density <= 4e-2 -> class 17, weak_gausstr_head.py).

No Gaussian thumbnail: the baseline dump has no Gaussian tensors and no LocalAgg dump survives,
so the figure uses a plain "3D Gaussians" box.

Usage:
    python scripts/render_fig_assets.py occ --render-dir "<dir with 600_sem_*.png>"
    python scripts/render_fig_assets.py cams
    python scripts/render_fig_assets.py check
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

TITLE_BAND_PX = 60   # height of the matplotlib title band in the original renders
THUMB_W = 720
PAD = 12


def load_rgb(path):
    img = Image.open(path).convert('RGBA')
    bg = Image.new('RGBA', img.size, (255, 255, 255, 255))
    return Image.alpha_composite(bg, img).convert('RGB')


def content_box(img, thr=250):
    a = np.asarray(img)
    ys, xs = np.where((a < thr).any(axis=2))
    return xs.min(), ys.min(), xs.max() + 1, ys.max() + 1


def make_occ(render_dir, sample, out_dir, aspect=230 / 211):
    pairs = [
        (f'{sample}_sem_gaussvox.png', 'occ_baseline.png'),
        (f'{sample}_sem_localagg.png', 'occ_ours.png'),
    ]
    imgs = [load_rgb(Path(render_dir) / src) for src, _ in pairs]
    assert imgs[0].size == imgs[1].size, 'renders differ in size; view may not match'
    W, H = imgs[0].size

    # Scene = coloured voxel pixels (grey panes, grid lines and black tick text are unsaturated).
    # One shared box = union of both scenes, padded, then grown on the short side to the
    # hero-cell aspect so the scene fills the cell without distortion. Crop only, no re-render.
    boxes = []
    for img in imgs:
        a = np.asarray(img).astype(np.int16)[TITLE_BAND_PX:]
        ys, xs = np.where((a.max(-1) - a.min(-1)) > 40)
        boxes.append((xs.min(), ys.min() + TITLE_BAND_PX, xs.max() + 1, ys.max() + 1 + TITLE_BAND_PX))
    x0, y0 = min(b[0] for b in boxes) - PAD, min(b[1] for b in boxes) - PAD
    x1, y1 = max(b[2] for b in boxes) + PAD, max(b[3] for b in boxes) + PAD
    w, h = x1 - x0, y1 - y0
    if w / h > aspect:
        grow = round(w / aspect) - h
        y0, y1 = y0 - grow // 2, y1 + grow - grow // 2
    else:
        grow = round(h * aspect) - w
        x0, x1 = x0 - grow // 2, x1 + grow - grow // 2
    # Keep the black axis lines out: if the box reaches one below the scene, slide the box up.
    for img in imgs:
        a = np.asarray(img).astype(np.int16)[:, x0:x1]
        dark = (a.max(-1) < 90) & ((a.max(-1) - a.min(-1)) < 30)
        rows = np.where(dark[max(b[3] for b in boxes):].any(1))[0]
        if len(rows):
            limit = max(b[3] for b in boxes) + rows.min() - 4
            if y1 > limit:
                y0, y1 = y0 - (y1 - limit), limit
    box = (max(0, x0), max(TITLE_BAND_PX, y0), min(W, x1), min(H, y1))
    print('shared crop box (x0, y0, x1, y1):', box)

    out_dir.mkdir(parents=True, exist_ok=True)
    for img, (src, dst) in zip(imgs, pairs):
        c = img.crop(box)
        c = c.resize((THUMB_W, round(c.height * THUMB_W / c.width)), Image.LANCZOS)
        c.save(out_dir / dst, optimize=True)
        print(f'{src} -> {out_dir / dst}  {c.size}')


CAMS = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
        'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
CAM_W = 480


def make_cams(raw_dir, out_dir):
    for cam in CAMS:
        img = Image.open(raw_dir / 'cams' / f'{cam}.jpg').convert('RGB')
        img = img.resize((CAM_W, round(img.height * CAM_W / img.width)), Image.LANCZOS)
        img.save(out_dir / f'cam_{cam}.jpg', quality=88)
        print(f'{cam}.jpg -> {out_dir / f"cam_{cam}.jpg"}  {img.size}')


def check(raw_dir, sample):
    import pickle
    with open(raw_dir / 'dumps' / f'{sample}_gaussvox.pkl', 'rb') as f:
        d = pickle.load(f)
    print('dump keys:', sorted(d))
    print('has Gaussians:', any(k in d for k in ('means3d', 'scales', 'covariances')))

    gt = np.load(raw_dir / 'gt' / 'labels.npz')['semantics'].astype(np.int64)
    occ = d['gt_occ'].astype(np.int64)
    print(f'gt_occ == labels.npz semantics: {np.array_equal(occ, gt)} '
          f'({int((occ != gt).sum())} / {gt.size} voxels differ)')

    pred, den = d['preds_gaussvox'], d['density_gaussvox']
    low = den <= 4e-2
    print(f'density <= 4e-2: {int(low.sum())} voxels, pred == 17 in {int((pred[low] == 17).sum())}')
    print(f'density >  4e-2: {int((~low).sum())} voxels, pred != 17 in {int((pred[~low] != 17).sum())}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('what', choices=['occ', 'cams', 'check'])
    ap.add_argument('--render-dir')
    ap.add_argument('--sample', default='600')
    ap.add_argument('--raw-dir', default='assets/fig_src/raw')
    ap.add_argument('--out-dir', default='assets/fig_src')
    args = ap.parse_args()
    if args.what == 'occ':
        make_occ(args.render_dir, args.sample, Path(args.out_dir))
    elif args.what == 'cams':
        make_cams(Path(args.raw_dir), Path(args.out_dir))
    else:
        check(Path(args.raw_dir), args.sample)
