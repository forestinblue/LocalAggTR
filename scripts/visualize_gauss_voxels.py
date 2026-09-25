# tools/visualize_gauss_voxels.py
"""
Simple script to visualize 3D voxel predictions saved by DumpGaussVoxelizerHook
as a 3D scatter plot.

Input: per-sample .pkl files saved by DumpGaussVoxelizerHook in
       gausstr/hooks/dump_gauss_voxels.py, e.g. outputs_gaussvox/*.pkl

Required / optional keys in each per-sample pkl:

  Required:
    - preds_gaussvox   : [H, W, D] int16  (final prediction based on GaussVoxelizer)
    - density_gaussvox : [H, W, D] float32

  Optional:
    - logits_gaussvox  : [C, H, W, D] float32 (not used in this script)
    - voxel_size       : float (if missing, use the --voxel_size argument)
    - gt_occ           : [H, W, D] int16 (3D occupancy GT, visualized only if present)
"""

import os
import argparse
import pickle
from glob import glob

import numpy as np

# Use Agg backend so it also works on headless servers
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (import for projection='3d')


# Semantic palette you used before (classes 0–16, 17 = void)
COLORS = np.array([
    [0, 0, 0, 255],
    [112, 128, 144, 255],
    [220, 20, 60, 255],
    [255, 127, 80, 255],
    [255, 158, 0, 255],
    [233, 150, 70, 255],
    [255, 61, 99, 255],
    [0, 0, 230, 255],
    [47, 79, 79, 255],
    [255, 140, 0, 255],
    [255, 98, 70, 255],
    [0, 207, 191, 255],
    [175, 0, 75, 255],
    [75, 0, 75, 255],
    [112, 180, 60, 255],
    [222, 184, 135, 255],
    [0, 175, 0, 255],
], dtype=np.float32)


def _make_grid_coords(shape, voxel_size):
    """
    shape: (H, W, D)
    voxel_size: float (world-coordinate scale in meters)

    Return: pts [N, 3] float (x, y, z in world coordinates, voxel-center aligned)
    """
    H, W, D = shape
    xs, ys, zs = np.meshgrid(
        np.arange(H), np.arange(W), np.arange(D), indexing="ij"
    )
    pts = np.stack([xs, ys, zs], axis=-1).reshape(-1, 3).astype(np.float32)
    # Shift to voxel centers and scale by voxel_size
    pts = (pts + 0.5) * float(voxel_size)
    return pts  # [N, 3]


def _set_equal_aspect_3d(ax, pts):
    """Roughly enforce a 1:1:1 aspect ratio in a 3D scatter plot."""
    if pts.shape[0] == 0:
        return
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    max_range = np.array([
        x.max() - x.min(),
        y.max() - y.min(),
        z.max() - z.min()
    ]).max()
    mid_x = (x.max() + x.min()) * 0.5
    mid_y = (y.max() + y.min()) * 0.5
    mid_z = (z.max() + z.min()) * 0.5
    ax.set_xlim(mid_x - max_range / 2, mid_x + max_range / 2)
    ax.set_ylim(mid_y - max_range / 2, mid_y + max_range / 2)
    ax.set_zlim(mid_z - max_range / 2, mid_z + max_range / 2)


def plot_vox_semantic(
    vox,
    title,
    voxel_size=0.4,
    ignore=(17, 255),
    save=None,
    max_points=300_000,
):
    """
    Visualize a semantic volume as a 3D scatter plot (matplotlib).

    vox:    [H, W, D] int (semantic id)
    title:  figure title
    ignore: class ids to ignore (e.g., void=17, padding=255)
    save:   output path (.png). If None, use plt.show() (no effect on headless)
    """
    H, W, D = vox.shape
    vals = vox.reshape(-1)

    mask = np.ones_like(vals, dtype=bool)
    for ig in ignore:
        mask &= (vals != ig)

    if not np.any(mask):
        print(f"[WARN] plot_vox_semantic: no valid voxels to plot for title='{title}'")
        return

    # Create coordinates
    pts = _make_grid_coords((H, W, D), voxel_size)
    pts = pts[mask]
    vals = vals[mask]

    # Subsample if there are too many points
    N = pts.shape[0]
    if N > max_points:
        idx = np.random.choice(N, size=max_points, replace=False)
        pts = pts[idx]
        vals = vals[idx]

    # Clamp to valid COLOR LUT index range
    vals_clamped = np.clip(vals, 0, COLORS.shape[0] - 1).astype(np.int64)
    colors_rgba = COLORS[vals_clamped] / 255.0  # [N,4], range 0–1

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Swap x and y for more intuitive viewing
    xs = pts[:, 1]
    ys = pts[:, 0]
    zs = pts[:, 2]

    ax.scatter(xs, ys, zs, c=colors_rgba, marker="s", s=2, linewidths=0)

    _set_equal_aspect_3d(ax, np.stack([xs, ys, zs], axis=-1))
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    if save is not None:
        os.makedirs(os.path.dirname(save), exist_ok=True)
        fig.savefig(save, dpi=200, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_vox_scalar(
    scalar_vol,
    title,
    voxel_size=0.4,
    save=None,
    max_points=300_000,
):
    """
    Visualize a scalar volume (e.g., density) as a 3D scatter plot.

    scalar_vol: [H, W, D] float
    """
    H, W, D = scalar_vol.shape
    vals = scalar_vol.reshape(-1).astype(np.float32)

    # Remove NaN/Inf and filter out near-zero values
    m = np.isfinite(vals) & (vals > 1e-8)
    if not np.any(m):
        print(f"[WARN] plot_vox_scalar: no valid points for title='{title}'")
        return

    pts = _make_grid_coords((H, W, D), voxel_size)
    pts = pts[m]
    vals = vals[m]

    N = pts.shape[0]
    if N > max_points:
        idx = np.random.choice(N, size=max_points, replace=False)
        pts = pts[idx]
        vals = vals[idx]

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    xs = pts[:, 1]
    ys = pts[:, 0]
    zs = pts[:, 2]

    # Normalize for colormap
    vmin = float(vals.min())
    vmax = float(vals.max())
    if vmin == vmax:
        vmax = vmin + 1e-6
    norm = (vals - vmin) / (vmax - vmin)
    colors = plt.cm.jet(norm)

    ax.scatter(xs, ys, zs, c=colors, marker="s", s=2, linewidths=0)

    _set_equal_aspect_3d(ax, np.stack([xs, ys, zs], axis=-1))
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    if save is not None:
        os.makedirs(os.path.dirname(save), exist_ok=True)
        fig.savefig(save, dpi=200, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def main():
    ap = argparse.ArgumentParser(
        description="Visualize GaussVoxelizer voxel predictions (matplotlib 3D)."
    )
    ap.add_argument(
        "pattern",
        help="Glob pattern for DumpGaussVoxelizerHook pkl files "
             "(e.g., 'outputs_gaussvox/*.pkl').",
    )
    ap.add_argument(
        "--voxel_size",
        type=float,
        default=0.4,
        help=(
            "Voxel edge length in meters (default: 0.4). "
            "If the pkl has 'voxel_size', that value takes precedence."
        ),
    )
    ap.add_argument(
        "--save_dir",
        default="vis_gaussvox",
        help="Output directory for saved PNGs.",
    )
    ap.add_argument(
        "--mode",
        choices=["sem", "dens", "gt", "all"],
        default="sem",
        help=(
            "Which volume to visualize:\n"
            "  sem  : preds_gaussvox semantic volume\n"
            "  dens : density_gaussvox scalar volume\n"
            "  gt   : 3D GT (gt_occ) volume\n"
            "  all  : visualize all three"
        ),
    )
    ap.add_argument(
        "--void_id",
        type=int,
        default=17,
        help="Void/free class id (default: 17, used for ignoring in sem/gt visualization).",
    )
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    files = sorted(glob(args.pattern))
    if len(files) == 0:
        print(f"[WARN] No files matched pattern: {args.pattern}")
        return

    print(f"[INFO] Found {len(files)} files.")

    if args.mode == "all":
        modes = ["sem", "dens", "gt"]
    else:
        modes = [args.mode]

    for fpath in files:
        with open(fpath, "rb") as fh:
            data = pickle.load(fh)

        sid = data.get("sample_idx", os.path.basename(fpath).split(".")[0])

        preds_gaussvox = data.get("preds_gaussvox", None)
        dens_gaussvox = data.get("density_gaussvox", None)
        gt_occ = data.get("gt_occ", None)

        if preds_gaussvox is None:
            print(f"[WARN] {fpath} has no 'preds_gaussvox'. Skip file.")
            continue

        # voxel_size: if present in pkl, use that first
        voxel_size = float(data.get("voxel_size", args.voxel_size))

        for mode in modes:
            if mode == "sem":
                vox = preds_gaussvox.astype(np.int32)
                title = f"{sid} | GaussVoxelizer semantic"
                out_name = f"{sid}_sem_gaussvox.png"
                save_path = os.path.join(args.save_dir, out_name)
                print(f"[INFO] Rendering {sid} / mode=sem → {save_path}")
                plot_vox_semantic(
                    vox,
                    title=title,
                    voxel_size=voxel_size,
                    ignore=(args.void_id, 255),
                    save=save_path,
                )

            elif mode == "dens":
                if dens_gaussvox is None:
                    print(f"[WARN] {fpath} has no 'density_gaussvox'. Skip dens.")
                    continue
                vol = dens_gaussvox.astype(np.float32)
                title = f"{sid} | GaussVoxelizer density"
                out_name = f"{sid}_dens_gaussvox.png"
                save_path = os.path.join(args.save_dir, out_name)
                print(f"[INFO] Rendering {sid} / mode=dens → {save_path}")
                plot_vox_scalar(
                    vol,
                    title=title,
                    voxel_size=voxel_size,
                    save=save_path,
                )

            elif mode == "gt":
                if gt_occ is None:
                    print(f"[WARN] {fpath} has no 'gt_occ'. Skip gt.")
                    continue
                if gt_occ.shape != preds_gaussvox.shape:
                    print(
                        f"[WARN] Shape mismatch for GT in {fpath}: "
                        f"gt_occ {gt_occ.shape} vs preds_gaussvox {preds_gaussvox.shape}"
                    )
                    continue
                vox = gt_occ.astype(np.int32)
                title = f"{sid} | GT occupancy"
                out_name = f"{sid}_gt.png"
                save_path = os.path.join(args.save_dir, out_name)
                print(f"[INFO] Rendering {sid} / mode=gt → {save_path}")
                plot_vox_semantic(
                    vox,
                    title=title,
                    voxel_size=voxel_size,
                    ignore=(args.void_id, 255),
                    save=save_path,
                )

            else:
                print(f"[WARN] Unknown mode='{mode}' (skip).")


if __name__ == "__main__":
    main()