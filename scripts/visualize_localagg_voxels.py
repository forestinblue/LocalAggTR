# tools/visualize_localagg_voxels.py
"""
Script to visualize 3D voxel predictions saved by DumpPredictFairCompareHook
(matplotlib 3D version).

Input: per-sample .pkl files saved by DumpPredictFairCompareHook in
       gausstr/hooks/dump_localagg_result.py, e.g. outputs_compare/*.pkl

Required keys (per-sample pkl):
  - preds_localagg   : [H, W, D] int (LocalAgg argmax)

Optional keys:
  - pred_head        : [H, W, D] int (Head final prediction)
  - density_localagg : [H, W, D] float (LocalAgg density proxy)
  - voxel_size       : float (if missing, use --voxel_size)
  - gt_occ           : [H, W, D] int (Occ3D 3D occupancy GT)
"""

import os
import argparse
import pickle
from glob import glob

import numpy as np

# matplotlib: use Agg backend so it also works on headless servers
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (import for projection='3d')


# Reuse your visualize.py palette (0–16 semantic classes, 17 = void)
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
    voxel_size: float (world coordinate scale in meters)

    Return: pts [N, 3] float (x, y, z in world coordinates)
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

    vox:       [H, W, D] int (semantic id)
    title:     figure title
    ignore:    class ids to ignore (e.g., void=17, padding=255)
    save:      output path (.png). If None, fig.show() (no effect on headless)
    max_points: if there are too many voxels, subsample
    """
    H, W, D = vox.shape
    vals = vox.reshape(-1)

    mask = np.ones_like(vals, dtype=bool)
    for ig in ignore:
        mask &= (vals != ig)

    if not np.any(mask):
        print(f"[WARN] plot_vox_semantic: no valid voxels to plot for title='{title}'")
        return

    # Generate coordinates
    pts = _make_grid_coords((H, W, D), voxel_size)
    pts = pts[mask]
    vals = vals[mask]

    # Subsample if there are too many points
    N = pts.shape[0]
    if N > max_points:
        idx = np.random.choice(N, size=max_points, replace=False)
        pts = pts[idx]
        vals = vals[idx]

    # Clamp to valid LUT index range
    vals_clamped = np.clip(vals, 0, COLORS.shape[0] - 1).astype(np.int64)
    colors_rgba = COLORS[vals_clamped] / 255.0  # [N, 4], range 0–1

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Swap x and y for a more intuitive view
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
    Visualize a scalar volume such as density as a 3D scatter plot (matplotlib).

    scalar_vol: [H, W, D] float
    title:      figure title
    save:       output path (.png)
    max_points: if there are too many voxels, subsample
    """
    H, W, D = scalar_vol.shape
    vals = scalar_vol.reshape(-1).astype(np.float32)

    # Remove NaN/Inf and drop near-zero values
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


def make_agree_diff_volumes(pred_a, pred_b, void_id=17):
    """
    Build agree / diff semantic volumes from two predictions (a = LocalAgg, b = head or GT).

    Returns:
      agree_vol: voxels where a == b and both are non-void keep class,
                 all others are 255
      diff_vol:  voxels where a != b and both are non-void keep class,
                 all others are 255
    """
    assert pred_a.shape == pred_b.shape
    pt = pred_a.astype(np.int32)
    ps = pred_b.astype(np.int32)

    valid = (pt != void_id) & (ps != void_id)
    same = (pt == ps) & valid
    diff = (pt != ps) & valid

    agree_vol = np.full_like(pt, 255, dtype=np.int32)
    diff_vol = np.full_like(pt, 255, dtype=np.int32)

    agree_vol[same] = pt[same]
    diff_vol[diff] = pt[diff]  # color using classes of a

    return agree_vol, diff_vol


def make_la_dens_mask_volume(pred_a, dens_la, thr, void_id=17):
    """
    Keep LocalAgg semantics only at voxels whose density is above the threshold,
    and set all other voxels to ignore (255).
    """
    pt = pred_a.astype(np.int32)
    mask = np.isfinite(dens_la) & (dens_la >= thr) & (pt != void_id)
    out = np.full_like(pt, 255, dtype=np.int32)
    out[mask] = pt[mask]
    return out


def main():
    ap = argparse.ArgumentParser(
        description="Visualize LocalAgg / Head voxel predictions (matplotlib 3D)."
    )
    ap.add_argument(
        "pattern",
        help=(
            "Glob pattern for DumpPredictFairCompareHook pkl files. "
            "e.g., 'outputs_compare/*.pkl'"
        ),
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
        default="visualizations_compare",
        help="Output directory for saved PNGs.",
    )
    ap.add_argument(
        "--mode",
        choices=[
            "localagg", "head",
            "agree", "diff",
            "dens", "la_dens_mask",
            "gt",
            "la_gt_agree",
            "la_gt_diff",
            "all"
        ],
        default="localagg",
        help=(
            "Which volume to visualize:\n"
            "  localagg     : LocalAgg preds_localagg only\n"
            "  head         : pred_head (only if present)\n"
            "  agree        : voxels where LocalAgg == Head only\n"
            "  diff         : voxels where LocalAgg != Head only\n"
            "  dens         : density_localagg scalar volume\n"
            "  la_dens_mask : voxels with density_localagg >= thr keep LocalAgg class\n"
            "  gt           : 3D GT (gt_occ) only\n"
            "  la_gt_agree  : voxels where LocalAgg == GT only\n"
            "  la_gt_diff   : voxels where LocalAgg != GT only\n"
            "  all          : all available modes above"
        ),
    )
    ap.add_argument(
        "--void_id",
        type=int,
        default=17,
        help="Void/free class id (default: 17). Used for agree/diff and mask generation.",
    )
    ap.add_argument(
        "--dens_thr",
        type=float,
        default=1e-7,
        help="Density threshold used in la_dens_mask mode (default: 1e-7).",
    )
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    files = sorted(glob(args.pattern))
    if len(files) == 0:
        print(f"[WARN] No files matched pattern: {args.pattern}")
        return

    print(f"[INFO] Found {len(files)} files.")

    # Decide modes to run
    if args.mode == "all":
        modes = [
            "localagg", "head",
            "agree", "diff",
            "dens", "la_dens_mask",
            "gt", "la_gt_agree", "la_gt_diff",
        ]
    else:
        modes = [args.mode]

    for fpath in files:
        with open(fpath, "rb") as fh:
            data = pickle.load(fh)

        sid = data.get("sample_idx", os.path.basename(fpath).split(".")[0])
        preds_localagg = data.get("preds_localagg", None)
        pred_head = data.get("pred_head", None)
        dens_la = data.get("density_localagg", None)
        gt_occ = data.get("gt_occ", None)

        if preds_localagg is None:
            print(f"[WARN] {fpath} has no 'preds_localagg'. Skip.")
            continue

        # voxel_size: if present in pkl metadata, use that; otherwise use args.voxel_size
        voxel_size = float(data.get("voxel_size", args.voxel_size))

        for mode in modes:
            if mode == "localagg":
                title = f"{sid} | LocalAgg"
                out_name = f"{sid}_localagg.png"
                save_path = os.path.join(args.save_dir, out_name)
                print(f"[INFO] Rendering {sid} / mode=localagg → {save_path}")
                vox = preds_localagg.astype(np.int32)
                plot_vox_semantic(
                    vox,
                    title=title,
                    voxel_size=voxel_size,
                    ignore=(args.void_id, 255),
                    save=save_path,
                )

            elif mode == "head":
                if pred_head is None:
                    print(f"[WARN] {fpath} has no 'pred_head'. Skip mode=head.")
                    continue
                title = f"{sid} | Head"
                out_name = f"{sid}_head.png"
                save_path = os.path.join(args.save_dir, out_name)
                print(f"[INFO] Rendering {sid} / mode=head → {save_path}")
                vox = pred_head.astype(np.int32)
                plot_vox_semantic(
                    vox,
                    title=title,
                    voxel_size=voxel_size,
                    ignore=(args.void_id, 255),
                    save=save_path,
                )

            elif mode in ("agree", "diff"):
                if pred_head is None:
                    print(f"[WARN] {fpath} has no 'pred_head'. Skip mode={mode}.")
                    continue
                agree_vol, diff_vol = make_agree_diff_volumes(
                    preds_localagg, pred_head, void_id=args.void_id
                )
                if mode == "agree":
                    vox = agree_vol
                    suffix = "agree"
                    title = f"{sid} | LocalAgg vs Head (agree only)"
                else:
                    vox = diff_vol
                    suffix = "diff"
                    title = f"{sid} | LocalAgg vs Head (diff only)"

                out_name = f"{sid}_{suffix}.png"
                save_path = os.path.join(args.save_dir, out_name)
                print(f"[INFO] Rendering {sid} / mode={mode} → {save_path}")
                plot_vox_semantic(
                    vox,
                    title=title,
                    voxel_size=voxel_size,
                    ignore=(255,),
                    save=save_path,
                )

            elif mode == "gt":
                if gt_occ is None:
                    print(f"[WARN] {fpath} has no 'gt_occ'. Skip mode=gt.")
                    continue
                if gt_occ.shape != preds_localagg.shape:
                    print(
                        f"[WARN] Shape mismatch for GT in {fpath}: "
                        f"gt_occ {gt_occ.shape} vs preds_localagg {preds_localagg.shape}"
                    )
                    continue

                title = f"{sid} | GT occupancy"
                out_name = f"{sid}_gt.png"
                save_path = os.path.join(args.save_dir, out_name)
                print(f"[INFO] Rendering {sid} / mode=gt → {save_path}")

                vox = gt_occ.astype(np.int32)
                plot_vox_semantic(
                    vox,
                    title=title,
                    voxel_size=voxel_size,
                    ignore=(args.void_id, 255),
                    save=save_path,
                )

            elif mode in ("la_gt_agree", "la_gt_diff"):
                if gt_occ is None:
                    print(f"[WARN] {fpath} has no 'gt_occ'. Skip mode={mode}.")
                    continue
                if gt_occ.shape != preds_localagg.shape:
                    print(
                        f"[WARN] Shape mismatch for GT in {fpath}: "
                        f"gt_occ {gt_occ.shape} vs preds_localagg {preds_localagg.shape}"
                    )
                    continue

                agree_vol, diff_vol = make_agree_diff_volumes(
                    preds_localagg, gt_occ, void_id=args.void_id
                )

                if mode == "la_gt_agree":
                    vox = agree_vol
                    suffix = "la_gt_agree"
                    title = f"{sid} | LocalAgg vs GT (agree only)"
                else:
                    vox = diff_vol
                    suffix = "la_gt_diff"
                    title = f"{sid} | LocalAgg vs GT (diff only)"

                out_name = f"{sid}_{suffix}.png"
                save_path = os.path.join(args.save_dir, out_name)
                print(f"[INFO] Rendering {sid} / mode={mode} → {save_path}")

                plot_vox_semantic(
                    vox,
                    title=title,
                    voxel_size=voxel_size,
                    ignore=(255,),
                    save=save_path,
                )

            elif mode == "dens":
                if dens_la is None:
                    print(f"[WARN] {fpath} has no 'density_localagg'. Skip mode=dens.")
                    continue
                title = f"{sid} | density_localagg"
                out_name = f"{sid}_dens.png"
                save_path = os.path.join(args.save_dir, out_name)
                print(f"[INFO] Rendering {sid} / mode=dens → {save_path}")
                plot_vox_scalar(
                    dens_la.astype(np.float32),
                    title=title,
                    voxel_size=voxel_size,
                    save=save_path,
                )

            elif mode == "la_dens_mask":
                if dens_la is None:
                    print(f"[WARN] {fpath} has no 'density_localagg'. Skip mode=la_dens_mask.")
                    continue
                vox = make_la_dens_mask_volume(
                    preds_localagg, dens_la, thr=args.dens_thr, void_id=args.void_id
                )
                title = f"{sid} | LocalAgg (density >= {args.dens_thr:g})"
                out_name = f"{sid}_la_dens_mask.png"
                save_path = os.path.join(args.save_dir, out_name)
                print(f"[INFO] Rendering {sid} / mode=la_dens_mask → {save_path}")
                plot_vox_semantic(
                    vox,
                    title=title,
                    voxel_size=voxel_size,
                    ignore=(255,),
                    save=save_path,
                )

            else:
                print(f"[WARN] Unknown mode='{mode}' (skip).")


if __name__ == "__main__":
    main()