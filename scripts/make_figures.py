"""
Build the two README figures.

  assets/hero.png      GT | GaussTR Gaussian voxelizer | Local Aggregation, one val sample.
                       Composed from the 3D renders made by scripts/visualize_gauss_voxels.py and
                       scripts/visualize_localagg_voxels.py on the dec_gaussvox / dec_localagg
                       dumps (see results/NUMBERS.md). The original per-panel titles are cropped
                       and replaced; the renders themselves are not modified.
  assets/pipeline.png  Block diagram: which parts are upstream / lab / this repo.

Usage:
    python scripts/make_figures.py --render-dir "<dir with 600_gt.png, 600_sem_gaussvox.png, 600_sem_localagg.png>"
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch
from PIL import Image

# Same palette as scripts/visualize_*_voxels.py (OccMetric class order, 17 = free / not drawn)
CLASS_NAMES = [
    'others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
    'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade',
    'vegetation',
]
COLORS = np.array([
    [0, 0, 0], [112, 128, 144], [220, 20, 60], [255, 127, 80],
    [255, 158, 0], [233, 150, 70], [255, 61, 99], [0, 0, 230],
    [47, 79, 79], [255, 140, 0], [255, 98, 70], [0, 207, 191],
    [175, 0, 75], [75, 0, 75], [112, 180, 60], [222, 184, 135],
    [0, 175, 0],
]) / 255.0

TITLE_BAND_PX = 60   # height of the original matplotlib title band in the renders


def make_hero(render_dir, sample, out_path):
    panels = [
        (f'{sample}_gt.png', 'Ground truth'),
        (f'{sample}_sem_gaussvox.png', 'GaussTR Gaussian voxelizer'),
        (f'{sample}_sem_localagg.png', 'Local Aggregation backend (this repo)'),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.6), dpi=150)
    for ax, (name, title) in zip(axes, panels):
        img = Image.open(Path(render_dir) / name).convert('RGB')
        img = img.crop((0, TITLE_BAND_PX, img.width, img.height))
        ax.imshow(img)
        ax.set_title(title, fontsize=13)
        ax.axis('off')

    handles = [Patch(color=COLORS[i], label=n) for i, n in enumerate(CLASS_NAMES) if i > 0]
    fig.legend(handles=handles, loc='lower center', ncol=8, fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, 0.0))
    fig.text(0.5, 0.965,
             f'nuScenes val, dumped sample #{sample}. Same checkpoint and head code for both '
             'predictions; only the voxelizer differs (runs dec_gaussvox / dec_localagg).',
             ha='center', fontsize=9, color='0.35')
    fig.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.12, wspace=0.02)
    fig.savefig(out_path, facecolor='white')
    plt.close(fig)


def _box(ax, xy, w, h, text, fc, ec, ls='-'):
    ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle='round,pad=0.02,rounding_size=0.08',
                                fc=fc, ec=ec, lw=1.4, ls=ls))
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha='center', va='center', fontsize=9.5,
            linespacing=1.35)


def _arrow(ax, p, q):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle='-|>', mutation_scale=12, lw=1.2,
                                 color='0.25'))


def make_pipeline(out_path):
    UP = ('#eef1f5', '#8a94a6')     # upstream GaussTR / lab (not in this repo)
    MINE = ('#e6f4ea', '#2e7d4f')   # this repo
    EXT = ('#fdf1e3', '#b7791f')    # third-party op, installed separately

    fig, ax = plt.subplots(figsize=(14.8, 4.5), dpi=150)
    ax.set_xlim(0, 14.8)
    ax.set_ylim(-0.1, 4.4)
    ax.axis('off')

    _box(ax, (0.1, 1.6), 1.7, 1.0, 'Surround-view\nimages (6 cams)', *UP)
    _box(ax, (2.2, 1.6), 2.4, 1.0, 'GaussTR backbone\n+ Gaussian decoder\n(lab model, not released)', *UP, ls='--')
    _box(ax, (5.0, 1.6), 2.0, 1.0, 'Per-Gaussian means,\nscales, opacities,\nsemantic logits', *UP, ls='--')
    _box(ax, (7.4, 1.6), 1.9, 1.0, 'Geometry / scale /\nopacity\nsanitization', *MINE)
    _box(ax, (10.4, 2.6), 2.5, 0.9, 'GaussianVoxelizerCompat\n(GaussTR voxelizer subclass)', *MINE)
    _box(ax, (10.4, 0.7), 2.5, 0.9, 'LocalAggWrapper\n+ density-quantile free mask', *MINE)
    _box(ax, (13.3, 1.6), 1.45, 1.0, 'Semantic\noccupancy\n200×200×16', *UP)
    _box(ax, (10.9, 0.0), 1.5, 0.45, 'local_aggregate\n(third_party)', *EXT)

    _arrow(ax, (1.8, 2.1), (2.2, 2.1))
    _arrow(ax, (4.6, 2.1), (5.0, 2.1))
    _arrow(ax, (7.0, 2.1), (7.4, 2.1))
    _arrow(ax, (9.3, 2.3), (10.4, 3.05))
    _arrow(ax, (9.3, 1.9), (10.4, 1.15))
    _arrow(ax, (12.9, 3.05), (13.3, 2.35))
    _arrow(ax, (12.9, 1.15), (13.3, 1.85))
    _arrow(ax, (11.65, 0.7), (11.65, 0.45))
    ax.text(9.85, 2.1, 'config:\nvoxelizer=\ndict(type=...)', ha='center', va='center',
            fontsize=7, color='0.3')

    handles = [
        Patch(fc=MINE[0], ec=MINE[1], label='this repo (src/localagg_tr)'),
        Patch(fc=UP[0], ec=UP[1], label='upstream GaussTR / lab pipeline (not in this repo)'),
        Patch(fc=EXT[0], ec=EXT[1], label='Local Aggregation CUDA op: GaussianFormer, installed separately'),
    ]
    ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=3,
              fontsize=8.5, frameon=False)
    fig.savefig(out_path, facecolor='white', bbox_inches='tight')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--render-dir', required=True)
    ap.add_argument('--sample', default='600')
    ap.add_argument('--out-dir', default='assets')
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    make_hero(args.render_dir, args.sample, out / 'hero.png')
    make_pipeline(out / 'pipeline.png')


if __name__ == '__main__':
    main()
