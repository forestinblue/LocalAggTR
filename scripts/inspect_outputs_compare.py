#!/usr/bin/env python

from pathlib import Path
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt


# Whether to visualize voxel-level comparison between LocalAgg and Head (default False)
COMPARE_HEAD_DEBUG = True


def load_pkl(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def _stat_1d(name, arr: np.ndarray):
    arr = arr.astype(np.float64).ravel()
    if arr.size == 0:
        print(f"{name:12s}: (empty)")
        return
    q25, q50, q75 = np.quantile(arr, [0.25, 0.5, 0.75])
    print(
        f"{name:12s}: mean={arr.mean():.4g} std={arr.std():.4g} "
        f"min={arr.min():.4g} q25={q25:.4g} med={q50:.4g} q75={q75:.4g} max={arr.max():.4g}"
    )


def _get_preds_from_sample(data, void_id: int = 17):
    """
    Expected new DumpPredictFairCompareHook format:

      - preds_localagg : [H,W,D] (LocalAgg argmax)
      - pred_head      : [H,W,D] (head final prediction, may be absent)

    Returns
    -------
    pred_la, pred_head
      - pred_la   : LocalAgg prediction (preds_localagg)
      - pred_head : Head final prediction (None if missing)
    """
    if "preds_localagg" not in data:
        raise KeyError("`preds_localagg` not found in pkl (DumpPredictFairCompareHook output expected).")

    pred_la = data["preds_localagg"].astype(np.int64)
    pred_head = data.get("pred_head", None)
    if pred_head is not None:
        pred_head = pred_head.astype(np.int64)

    return pred_la, pred_head


def _gaussians_to_voxel_groups(data, void_id=17):
    means = data.get("means3d", None)        # [G,3]
    smax = data.get("smax_post", None)       # [G]
    gt = data.get("gt_occ", None)           # [H,W,D]
    pred = data.get("preds_localagg", None)  # [H,W,D]
    pc_min = data.get("pc_min", None)        # (3,)
    voxel_size = data.get("voxel_size", None)
    grid_shape = data.get("grid_shape", None)  # [H,W,D]

    if any(v is None for v in [means, smax, gt, pred, pc_min, voxel_size, grid_shape]):
        return None

    H, W, D = grid_shape
    means = means.astype(np.float64)

    # world → voxel index (axis ordering may need to follow the actual LocalAgg definition)
    ix = np.floor((means[:, 0] - pc_min[0]) / voxel_size).astype(int)
    iy = np.floor((means[:, 1] - pc_min[1]) / voxel_size).astype(int)
    iz = np.floor((means[:, 2] - pc_min[2]) / voxel_size).astype(int)

    inrange = (
        (ix >= 0) & (ix < H) &
        (iy >= 0) & (iy < W) &
        (iz >= 0) & (iz < D)
    )
    if not inrange.any():
        return None

    ix, iy, iz = ix[inrange], iy[inrange], iz[inrange]
    smax = smax[inrange].astype(np.float64)

    gt_vox = gt[ix, iy, iz].astype(np.int64)
    pred_vox = pred[ix, iy, iz].astype(np.int64)

    gt_occ_mask = gt_vox != void_id
    pred_occ_mask = pred_vox != void_id

    # Four groups
    tp_g = gt_occ_mask & pred_occ_mask        # Gaussian centers whose voxel is occupied in both GT and prediction
    fn_g = gt_occ_mask & (~pred_occ_mask)     # GT occupied, prediction free
    fp_g = (~gt_occ_mask) & pred_occ_mask     # GT free, prediction occupied
    tn_g = (~gt_occ_mask) & (~pred_occ_mask)  # free in both

    return dict(
        smax=smax,
        tp_g=tp_g,
        fn_g=fn_g,
        fp_g=fp_g,
        tn_g=tn_g,
        gt_vox=gt_vox,
        pred_vox=pred_vox,
    )


def _band_metrics(gt, pred, pc_min, voxel_size, grid_shape, void_id=17):
    """
    Parameters
    ----------
    gt, pred : np.ndarray
        [H,W,D] class id volumes (including void_id).
    pc_min : array-like
        (3,) minimum point cloud coordinate.
    voxel_size : float
        Voxel size.
    grid_shape : tuple
        (H,W,D) grid shape.
    """
    H, W, D = grid_shape

    # Voxel center coordinates
    xs = pc_min[0] + (np.arange(H) + 0.5) * voxel_size
    ys = pc_min[1] + (np.arange(W) + 0.5) * voxel_size
    zs = pc_min[2] + (np.arange(D) + 0.5) * voxel_size
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")  # [H,W,D]

    # Distance on the ego-plane (can be changed to full 3D distance if needed)
    dist = np.sqrt(X**2 + Y**2)

    bands = [
        ("near", 0.0, 15.0),
        ("mid", 15.0, 30.0),
        ("far", 30.0, 60.0),
    ]
    res = {}
    gt_occ = (gt != void_id)
    pred_occ = (pred != void_id)

    for name, dmin, dmax in bands:
        mask_band = (dist >= dmin) & (dist < dmax)
        if not mask_band.any():
            continue

        g = gt_occ & mask_band
        p = pred_occ & mask_band

        tp = int((g & p).sum())
        fp = int((~g & p).sum())
        fn = int((g & ~p).sum())
        denom = tp + fp + fn
        iou = tp / denom if denom > 0 else 0.0
        res[name] = (tp, fp, fn, iou)

    return res


def summarize_sample(
    path: Path,
    void_id: int = 17,
    num_classes: int = 18,
    detailed: bool = True,
):
    data = load_pkl(path)
    print(f"\n=== {path.name} ===")
    print("keys:", list(data.keys()))

    # Print type / shape per key
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            print(f"  {k:16s} shape={v.shape}, dtype={v.dtype}")
        else:
            print(f"  {k:16s} type={type(v)} value={v}")

    # ------------------------------------------------------------
    # 0) Base predictions: LocalAgg, Head (optional)
    # ------------------------------------------------------------
    pred_la, pred_head = _get_preds_from_sample(data, void_id=void_id)

    H, W, D = pred_la.shape
    print(f"\n[Pred shapes] LocalAgg={pred_la.shape}, Head={getattr(pred_head, 'shape', None)}")

    # --- 0.1) LocalAgg prediction occupancy ratio (per-sample, including void) ---
    total_vox = pred_la.size
    la_labeled_mask = pred_la != void_id
    la_void_mask = ~la_labeled_mask
    num_la_labeled = int(la_labeled_mask.sum())
    num_la_void = int(la_void_mask.sum())

    print("\n[LocalAgg occupancy ratio (pred space, per-sample)]")
    print(f"  total voxels                 = {total_vox}")
    print(f"  labeled (pred != {void_id:2d}) = {num_la_labeled:9d} ({num_la_labeled / total_vox:6.2%})")
    print(f"  void    (pred == {void_id:2d}) = {num_la_void:9d} ({num_la_void / total_vox:6.2%})")

    # LocalAgg class histogram
    print("\n[Class histogram: LocalAgg preds (excluding void)]")
    mask_la = pred_la != void_id
    counts = np.bincount(pred_la[mask_la].ravel(), minlength=num_classes)
    total = int(mask_la.sum())
    for c in range(num_classes):
        if c == void_id:
            continue
        cnt = int(counts[c])
        if cnt == 0:
            continue
        print(f"  cls {c:2d}: {cnt:9d} ({cnt / total:6.2%})")

    # Optional: LocalAgg vs Head agreement (debug)
    if COMPARE_HEAD_DEBUG and pred_head is not None:
        print("\n[LocalAgg vs Head agreement (excluding void, DEBUG)]")
        valid = (pred_la != void_id) & (pred_head != void_id)
        same = (pred_la == pred_head) & valid
        total_valid = int(valid.sum())
        same_valid = int(same.sum())
        acc = same_valid / total_valid if total_valid > 0 else 0.0
        print(f"valid voxels={total_valid}, same={same_valid}, acc={acc:.4%}")

        if detailed and total_valid > 0:
            print("\n[Per-class agreement (LocalAgg as reference, excluding void, DEBUG)]")
            header = f"{'cls':>3s} | {'support':>9s} | {'match':>9s} | {'acc':>6s}"
            print(header)
            print("-" * len(header))
            for c in range(num_classes):
                if c == void_id:
                    continue
                mask_c = (pred_la == c) & valid
                supp = int(mask_c.sum())
                if supp == 0:
                    continue
                match = int((pred_head == pred_la)[mask_c].sum())
                acc_c = match / supp
                print(f"{c:3d} | {supp:9d} | {match:9d} | {acc_c:6.2%}")
        if detailed:
            print("\n[LocalAgg vs Head confusion (row=LocalAgg, col=Head, excluding void)]")
            la_valid = (pred_la != void_id) & (pred_head != void_id)
            la_flat = pred_la[la_valid].ravel()
            hd_flat = pred_head[la_valid].ravel()
            conf_la_hd = np.zeros((num_classes, num_classes), dtype=np.int64)
            for a, b in zip(la_flat, hd_flat):
                if 0 <= a < num_classes and 0 <= b < num_classes:
                    conf_la_hd[a, b] += 1

            # Print top off-diagonal pairs (e.g., top 10)
            pairs = []
            for a in range(num_classes):
                if a == void_id:
                    continue
                for b in range(num_classes):
                    if b == void_id or a == b:
                        continue
                    cnt = conf_la_hd[a, b]
                    if cnt > 0:
                        pairs.append((cnt, a, b))
            pairs.sort(reverse=True)

            print("Top LocalAgg→Head confusion pairs (this sample):")
            for cnt, a, b in pairs[:10]:
                print(f"  LocalAgg={a:2d} → Head={b:2d}: {cnt} voxels")
    elif pred_head is None:
        print("\n[No `pred_head` in pkl] -> Head comparison is skipped.")

    # ------------------------------------------------------------
    # 0.5) Comparison with 3D GT (occ) for LocalAgg / Head
    # ------------------------------------------------------------
    gt_occ = data.get("gt_occ", None)
    if gt_occ is not None:
        gt = gt_occ.astype(np.int64)
        if gt.shape != pred_la.shape:
            print(
                f"\n[WARN] gt_occ shape mismatch: gt_occ{gt.shape} "
                f"vs preds_localagg{pred_la.shape} -> GT comparison skipped"
            )
        else:
            # --- 0.4) GT occupancy ratio (per-sample) ---
            total_vox = gt.size
            gt_labeled_mask = gt != void_id
            gt_void_mask = ~gt_labeled_mask
            num_gt_labeled = int(gt_labeled_mask.sum())
            num_gt_void = int(gt_void_mask.sum())

            print("\n[GT occupancy ratio (per-sample)]")
            print(f"  total voxels               = {total_vox}")
            print(f"  labeled (gt != {void_id:2d}) = {num_gt_labeled:9d} ({num_gt_labeled / total_vox:6.2%})")
            print(f"  void    (gt == {void_id:2d}) = {num_gt_void:9d} ({num_gt_void / total_vox:6.2%})")

            # --- 0.5) Per-class volume ratio: GT vs LocalAgg (as % of all voxels) ---
            gt_counts_all = np.bincount(gt.ravel(), minlength=num_classes)
            la_counts_all = np.bincount(pred_la.ravel(), minlength=num_classes)

            print("\n[Per-class volume ratio (GT vs LocalAgg, % of all voxels)]")
            header = f"{'cls':>3s} | {'GT_cnt':>9s} | {'GT_%':>6s} | {'LA_cnt':>9s} | {'LA_%':>6s}"
            print(header)
            print("-" * len(header))
            for c in range(num_classes):
                # If you want to exclude void, remove `if c == void_id: continue`.
                # Keeping void in the table can be informative, so we show it.
                cnt_gt_c = int(gt_counts_all[c])
                cnt_la_c = int(la_counts_all[c])
                if (cnt_gt_c + cnt_la_c) == 0:
                    continue
                ratio_gt = cnt_gt_c / total_vox
                ratio_la = cnt_la_c / total_vox
                print(
                    f"{c:3d} | {cnt_gt_c:9d} | {ratio_gt:6.2%} | "
                    f"{cnt_la_c:9d} | {ratio_la:6.2%}"
                )

            print("\n[LocalAgg / Head vs 3D GT (excluding void)]")
            valid_gt = gt != void_id

            # --- 0.6) Binary occupancy metrics (per-sample, occupied = class != void_id) ---
            gt_occ_mask = gt != void_id
            la_occ_mask = pred_la != void_id

            tp_la = int((gt_occ_mask & la_occ_mask).sum())
            fp_la = int((~gt_occ_mask & la_occ_mask).sum())
            fn_la = int((gt_occ_mask & ~la_occ_mask).sum())
            tn_la = int((~gt_occ_mask & ~la_occ_mask).sum())
            tot_la = tp_la + fp_la + fn_la + tn_la

            def _print_occ_metrics_single(tag, tp, fp, fn, tn):
                total = tp + fp + fn + tn
                if total == 0:
                    print(f"  {tag}: no voxels")
                    return
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
                acc = (tp + tn) / total
                print(
                    f"  {tag}: TP={tp} FP={fp} FN={fn} TN={tn} | "
                    f"prec={prec:.4f} rec={rec:.4f} IoU={iou:.4f} acc={acc:.4f}"
                )

            print("\n[Binary occupancy (per-sample, GT vs LocalAgg)]")
            _print_occ_metrics_single("LocalAgg", tp_la, fp_la, fn_la, tn_la)

            pc_min = data.get("pc_min", None)
            voxel_size = data.get("voxel_size", None)
            grid_shape = data.get("grid_shape", None)
            if (pc_min is not None) and (voxel_size is not None) and (grid_shape is not None):
                band_res_la = _band_metrics(
                    gt, pred_la, pc_min, voxel_size, grid_shape, void_id=void_id
                )
                print("\n[Binary occupancy IoU by distance band (LocalAgg)]")
                for name, (tp_b, fp_b, fn_b, iou_b) in band_res_la.items():
                    print(f"  {name:4s}: IoU={iou_b:.4f} (TP={tp_b}, FP={fp_b}, FN={fn_b})")

            if pred_head is not None:
                head_occ_mask = pred_head != void_id
                tp_h = int((gt_occ_mask & head_occ_mask).sum())
                fp_h = int((~gt_occ_mask & head_occ_mask).sum())
                fn_h = int((gt_occ_mask & ~head_occ_mask).sum())
                tn_h = int((~gt_occ_mask & ~head_occ_mask).sum())
                print("[Binary occupancy (per-sample, GT vs Head)]")
                _print_occ_metrics_single("Head", tp_h, fp_h, fn_h, tn_h)

            # LocalAgg vs GT (multi-class, excluding void)
            same_la = (pred_la == gt) & valid_gt
            total_gt_valid = int(valid_gt.sum())
            same_la_cnt = int(same_la.sum())
            acc_la = same_la_cnt / total_gt_valid if total_gt_valid > 0 else 0.0
            print(
                f"  LocalAgg vs GT: valid={total_gt_valid}, "
                f"correct={same_la_cnt}, acc={acc_la:.4%}"
            )

            # Head vs GT (only if present)
            same_head_cnt = None
            acc_head = None

            if pred_head is not None:
                valid_head = valid_gt & (pred_head != void_id)
                same_head = (pred_head == gt) & valid_head
                total_head_valid = int(valid_head.sum())
                same_head_cnt = int(same_head.sum())
                acc_head = (
                    same_head_cnt / total_head_valid if total_head_valid > 0 else 0.0
                )
                print(
                    f"  Head    vs GT: valid={total_head_valid}, "
                    f"correct={same_head_cnt}, acc={acc_head:.4%}"
                )
            else:
                print("  (no pred_head) -> Head vs GT comparison is skipped")

            # Per-class recall (GT as reference)
            if detailed and total_gt_valid > 0:
                print("\n[Per-class vs GT (recall, GT as reference / excluding void)]")
                header = f"{'cls':>3s} | {'GT voxels':>9s} | {'LA correct':>10s} | {'LA_rec':>7s}"
                has_head = pred_head is not None
                if has_head:
                    header += f" | {'Head correct':>12s} | {'Head_rec':>8s}"
                print(header)
                print("-" * len(header))

                for c in range(num_classes):
                    if c == void_id:
                        continue
                    mask_gt_c = (gt == c) & valid_gt
                    supp_c = int(mask_gt_c.sum())
                    if supp_c == 0:
                        continue

                    la_corr_c = int((same_la & mask_gt_c).sum())
                    la_rec = la_corr_c / supp_c
                    line = f"{c:3d} | {supp_c:9d} | {la_corr_c:10d} | {la_rec:7.2%}"

                    if has_head:
                        head_corr_c = int(((pred_head == gt) & valid_gt & mask_gt_c).sum())
                        head_rec = head_corr_c / supp_c
                        line += f" | {head_corr_c:12d} | {head_rec:8.2%}"

                    print(line)

            # LocalAgg vs Head vs GT triple comparison (debug)
            if COMPARE_HEAD_DEBUG and pred_head is not None:
                print("\n[LocalAgg vs Head vs GT (voxel counts, excluding void, DEBUG)]")
                same_la = (pred_la == gt) & valid_gt
                same_head = (pred_head == gt) & valid_gt

                both_correct = same_la & same_head
                la_only = same_la & (~same_head)
                head_only = same_head & (~same_la)
                both_wrong = valid_gt & (~same_la) & (~same_head)

                print(f"  both_correct     : {int(both_correct.sum())}")
                print(f"  la_only_correct  : {int(la_only.sum())}")
                print(f"  head_only_correct: {int(head_only.sum())}")
                print(f"  both_wrong       : {int(both_wrong.sum())}")

                print("\n[Per-class triple stats (GT as reference, excluding void)]")
                header = (
                    f"{'cls':>3s} | {'GT vox':>7s} | {'both_ok':>7s} | "
                    f"{'la_only':>7s} | {'head_only':>9s} | {'both_wrong':>11s}"
                )
                print(header)
                print("-" * len(header))

                for c in range(num_classes):
                    if c == void_id:
                        continue
                    mask_gt_c = (gt == c) & valid_gt
                    supp = int(mask_gt_c.sum())
                    if supp == 0:
                        continue

                    bc = int((both_correct & mask_gt_c).sum())
                    la_o = int((la_only & mask_gt_c).sum())
                    hd_o = int((head_only & mask_gt_c).sum())
                    bw = int((both_wrong & mask_gt_c).sum())

                    print(
                        f"{c:3d} | {supp:7d} | {bc:7d} | "
                        f"{la_o:7d} | {hd_o:9d} | {bw:11d}"
                    )
    else:
        print("\n[No `gt_occ` in pkl] -> GT-based analysis is skipped.")

    # ------------------------------------------------------------
    # 1) Density-based analysis (LocalAgg density_proxy)
    # ------------------------------------------------------------
    dens_la = data.get("density_localagg", None)
    if dens_la is not None:
        print("\n[LocalAgg density_localagg stats]")
        print(
            f"{'dens_la':8s} min={dens_la.min():.4g} max={dens_la.max():.4g} "
            f"nan={np.isnan(dens_la).any()} inf={np.isinf(dens_la).any()}"
        )

    if not detailed:
        return data

    # ------------------------------------------------------------
    # 2) LocalAgg confidence (softmax max prob) vs Head agreement
    # ------------------------------------------------------------
    logits_3d = data.get("logits_3d", None)
    maxprob_la = None
    same_mask = None
    diff_mask = None

    if COMPARE_HEAD_DEBUG and (logits_3d is not None) and (pred_head is not None):
        print("\n[LocalAgg confidence (softmax max prob, DEBUG: conditioned on Head agreement)]")
        logits = logits_3d.astype(np.float64)  # [C,H,W,D]
        x = logits - logits.max(axis=0, keepdims=True)
        exp_x = np.exp(x)
        prob_la = exp_x / exp_x.sum(axis=0, keepdims=True)  # [C,H,W,D]
        maxprob_la = prob_la.max(axis=0)  # [H,W,D]

        same_mask = (pred_la == pred_head) & (pred_la != void_id) & (pred_head != void_id)
        diff_mask = (pred_la != pred_head) & (pred_la != void_id) & (pred_head != void_id)

        _stat_1d("la_p_same", maxprob_la[same_mask])
        _stat_1d("la_p_diff", maxprob_la[diff_mask])

    # Density vs confidence correlation (on voxels where LocalAgg and Head differ)
    if (dens_la is not None) and (maxprob_la is not None) and (diff_mask is not None):
        diff = diff_mask
        x = dens_la[diff].astype(np.float64).ravel()
        y = maxprob_la[diff].astype(np.float64).ravel()
        m = np.isfinite(x) & np.isfinite(y)
        x, y = x[m], y[m]
        if x.size > 1 and y.size > 1:
            corr = np.corrcoef(x, y)[0, 1]
            print(f"\n[corr(density_localagg, LocalAgg max prob) on diff voxels] = {corr:.3f}")
        else:
            print("\n[corr(density_localagg, LocalAgg max prob) on diff voxels] = (insufficient data)")

    # ------------------------------------------------------------
    # 2.5) LocalAgg density vs Head agreement (same / diff)
    # ------------------------------------------------------------
    if COMPARE_HEAD_DEBUG and (dens_la is not None) and (pred_head is not None):
        print("\n[LocalAgg density vs Head agreement (excluding void, DEBUG)]")
        same = (pred_la == pred_head) & (pred_la != void_id) & (pred_head != void_id)
        diff = (pred_la != pred_head) & (pred_la != void_id) & (pred_head != void_id)

        if same.any():
            _stat_1d("dens_la_same", dens_la[same])
        else:
            print("dens_la_same  : (no same voxels)")

        if diff.any():
            _stat_1d("dens_la_diff", dens_la[diff])

            vals = dens_la[diff]
            zero_like = (vals <= 1e-7)
            ratio_zero = float(zero_like.mean())
            print(
                f"dens_la_diff zero-ish ratio (<=1e-7): "
                f"{ratio_zero:.2%} (count={zero_like.sum()}/{vals.size})"
            )
        else:
            print("dens_la_diff  : (no diff voxels)")
            print("dens_la_diff zero-ish ratio (<=1e-7): (no diff voxels)")

    # ------------------------------------------------------------
    # 2.6) LocalAgg confidence / density vs GT correctness
    # ------------------------------------------------------------
    if gt_occ is not None and gt_occ.shape == pred_la.shape:
        gt = gt_occ.astype(np.int64)
        valid_gt = gt != void_id

        # LocalAgg correct / wrong voxels w.r.t GT
        correct_la = (pred_la == gt) & valid_gt
        wrong_la = (pred_la != gt) & valid_gt

        # (1) confidence vs GT
        if logits_3d is not None:
            # Reuse maxprob_la if already computed above
            if maxprob_la is None:
                logits = logits_3d.astype(np.float64)  # [C,H,W,D]
                x = logits - logits.max(axis=0, keepdims=True)
                exp_x = np.exp(x)
                prob_la = exp_x / exp_x.sum(axis=0, keepdims=True)
                maxprob_la = prob_la.max(axis=0)  # [H,W,D]

            print("\n[LocalAgg max prob vs GT (excluding void)]")
            if correct_la.any():
                _stat_1d("la_p_gt_correct", maxprob_la[correct_la])
            else:
                print("la_p_gt_correct: (no correct voxels)")

            if wrong_la.any():
                _stat_1d("la_p_gt_wrong", maxprob_la[wrong_la])
            else:
                print("la_p_gt_wrong  : (no wrong voxels)")

        # (2) density vs GT
        if dens_la is not None:
            print("\n[LocalAgg density vs GT (excluding void)]")
            if correct_la.any():
                _stat_1d("dens_gt_correct", dens_la[correct_la])
            else:
                print("dens_gt_correct: (no correct voxels)")

            if wrong_la.any():
                _stat_1d("dens_gt_wrong", dens_la[wrong_la])
                vals = dens_la[wrong_la]
                zero_like = (vals <= 1e-7)
                ratio_zero = float(zero_like.mean())
                print(
                    f"dens_gt_wrong zero-ish ratio (<=1e-7): "
                    f"{ratio_zero:.2%} (count={zero_like.sum()}/{vals.size})"
                )
            else:
                print("dens_gt_wrong  : (no wrong voxels)")

    # ------------------------------------------------------------
    # 3) Head-side clamp/floor/coord statistics (per-sample)
    # ------------------------------------------------------------
    alpha_pre = data.get("alpha_pre", None)
    alpha_post = data.get("alpha_post", None)
    smax_pre = data.get("smax_pre", None)
    smax_post = data.get("smax_post", None)
    orig_finite = data.get("orig_finite", None)
    oor_before = data.get("out_of_range_before", None)

    # σ cap statistics: smax_pre -> smax_post
    if (smax_pre is not None) and (smax_post is not None):
        s_pre = smax_pre.astype(np.float64).ravel()
        s_post = smax_post.astype(np.float64).ravel()
        diff_s = s_pre - s_post
        hit_mask = diff_s > 1e-6

        hit_ratio = hit_mask.mean() if hit_mask.size > 0 else 0.0
        print("\n[scale cap stats (per-sample)]")
        print(f"  hit_ratio = {hit_ratio:.2%} ( {hit_mask.sum()} / {hit_mask.size} )")
        if hit_mask.any():
            _stat_1d("smax_overshoot", diff_s[hit_mask])

    # α floor statistics: alpha_pre -> alpha_post
    if (alpha_pre is not None) and (alpha_post is not None):
        a_pre = alpha_pre.astype(np.float64).ravel()
        a_post = alpha_post.astype(np.float64).ravel()
        diff_a = a_post - a_pre
        floor_mask = diff_a > 1e-8
        floor_ratio = floor_mask.mean() if floor_mask.size > 0 else 0.0
        print("\n[alpha floor stats (per-sample)]")
        print(f"  floor_ratio = {floor_ratio:.2%} ( {floor_mask.sum()} / {floor_mask.size} )")
        if floor_mask.any():
            _stat_1d("alpha_raise", diff_a[floor_mask])

    # Coordinate NaN/Inf / out-of-pc_range ratio
    if orig_finite is not None or oor_before is not None:
        print("\n[coord sanitization stats (per-sample)]")
    if orig_finite is not None:
        finite = orig_finite.astype(bool).ravel()
        finite_ratio = finite.mean() if finite.size > 0 else 0.0
        print(
            f"  finite_ratio (before nan_to_num) = {finite_ratio:.2%} "
            f"( {finite.sum()} / {finite.size} )"
        )
    if oor_before is not None:
        oor = oor_before.astype(bool).ravel()
        ratio_oor = oor.mean() if oor.size > 0 else 0.0
        print(
            f"  out_of_range_before clamp = {ratio_oor:.2%} "
            f"( {oor.sum()} / {oor.size} )"
        )

    # ------------------------------------------------------------
    # 4) Gaussian-level α/σ statistics (raw vs effective, per gate)
    # ------------------------------------------------------------
    alpha_eff = data.get("alpha_eff", None)
    valid_hard = data.get("valid_hard", None)
    valid_soft = data.get("valid_soft", None)
    drop_soft = data.get("drop_soft", None)

    def _alpha_bucket_stats(name, arr, weak_thr=5e-4, mid_thr=5e-3):
        arr = arr.astype(np.float64).ravel()
        if arr.size == 0:
            print(f"{name:16s}: (empty)")
            return
        print(f"\n[{name}]")
        _stat_1d(name, arr)
        total = arr.size
        n_weak = int((arr < weak_thr).sum())
        n_mid = int(((arr >= weak_thr) & (arr < mid_thr)).sum())
        n_strong = int((arr >= mid_thr).sum())
        print(
            f"  weak(<{weak_thr:g}):   {n_weak:9d} ({n_weak / total:6.2%})\n"
            f"  mid([{weak_thr:g},{mid_thr:g})): {n_mid:9d} ({n_mid / total:6.2%})\n"
            f"  strong(≥{mid_thr:g}):  {n_strong:9d} ({n_strong / total:6.2%})"
        )

    # α distribution
    if alpha_pre is not None:
        _alpha_bucket_stats("alpha_pre_all", alpha_pre)
    if alpha_post is not None:
        _alpha_bucket_stats("alpha_post_all", alpha_post)
    if alpha_eff is not None:
        _alpha_bucket_stats("alpha_eff_all", alpha_eff)

    # α_eff per gate
    if (alpha_eff is not None) and (valid_hard is not None):
        vh = valid_hard.astype(bool).ravel()
        if vh.any():
            _alpha_bucket_stats("alpha_eff[valid_hard]", alpha_eff[vh])

    if (alpha_eff is not None) and (drop_soft is not None):
        ds = drop_soft.astype(bool).ravel()
        if ds.any():
            _alpha_bucket_stats("alpha_eff[drop_soft]", alpha_eff[ds])

    if (alpha_eff is not None) and (valid_soft is not None):
        vs = valid_soft.astype(bool).ravel()
        if vs.any():
            _alpha_bucket_stats("alpha_eff[valid_soft]", alpha_eff[vs])

    # σ statistics
    smax = smax_post if smax_post is not None else smax_pre
    if smax is not None:
        print("\n[Gaussian-level s_max stats]")
        smax = smax.astype(np.float64)
        _stat_1d("smax_all", smax)

        small_thr = 0.02
        mid_thr = 0.08

        total_s = smax.size
        n_small = int((smax < small_thr).sum())
        n_mid_s = int(((smax >= small_thr) & (smax < mid_thr)).sum())
        n_large = int((smax >= mid_thr).sum())

        print(
            f"  small(<{small_thr:g}):   {n_small:7d} ({n_small / total_s:6.2%})\n"
            f"  mid([{small_thr:g},{mid_thr:g})): {n_mid_s:7d} ({n_mid_s / total_s:6.2%})\n"
            f"  large(≥{mid_thr:g}):     {n_large:7d} ({n_large / total_s:6.2%})"
        )

    scale_raw = data.get("scale_raw", None)
    if scale_raw is not None:
        print("\n[scale_raw stats (flattened)]")
        _stat_1d("scale_raw", scale_raw)

    scale_u = data.get("scale_u", None)
    if scale_u is not None:
        print("\n[scale_u stats (0~1, flattened)]")
        _stat_1d("scale_u", scale_u)

    scales_arr = data.get("scales", None)
    if scales_arr is not None:
        print("\n[scales (after mapping) stats (flattened)]")
        _stat_1d("scales", scales_arr)

    # Gaussian center → voxel occupancy grouping (s_max distribution for TP/FP/FN/TN)
    g_stats = _gaussians_to_voxel_groups(data, void_id=void_id)
    if g_stats is not None:
        s = g_stats["smax"]
        print("\n[Gaussian s_max vs GT/pred occupancy (center-based)]")
        for name in ["tp_g", "fn_g", "fp_g", "tn_g"]:
            mask = g_stats[name]
            if mask.any():
                _stat_1d(f"smax[{name}]", s[mask])

    # ------------------------------------------------------------
    # 5) Gaussian gate alignment: Head vs Wrapper vs legacy gate heuristic
    # ------------------------------------------------------------
    drop_geo_head = data.get("drop_geo_head", None)
    if (drop_geo_head is not None) and (valid_hard is not None):
        print("\n[Gaussian gate alignment (drop masks per Gaussian)]")

        dg = drop_geo_head.astype(bool).ravel()  # Head drop mask
        vh = valid_hard.astype(bool).ravel()     # Wrapper valid_hard mask
        n = min(dg.size, vh.size)
        dg = dg[:n]
        vh = vh[:n]
        dw = ~vh                                  # Wrapper drop mask

        print(f"  #Gaussians used for gate comparison = {n}")

        # NOTE:
        #   Here, legacy_drop follows the "legacy heuristic gate":
        #   keep if (finite & in-range & alpha_pre >= 1e-3 & smax_pre > 1e-4),
        #   otherwise drop.
        legacy_drop = None
        if (alpha_pre is not None) and (smax_pre is not None) and (orig_finite is not None) and (oor_before is not None):
            ap = alpha_pre.astype(np.float64).ravel()[:n]
            sp = smax_pre.astype(np.float64).ravel()[:n]
            finite = orig_finite.astype(bool).ravel()[:n]
            oor = oor_before.astype(bool).ravel()[:n]

            keep_legacy = finite & (~oor) & (ap >= 1e-3) & (sp > 1e-4)
            legacy_drop = ~keep_legacy

        def _set_stats(name, m):
            if m is None:
                print(f"  {name:12s}: (None)")
                return
            m = m.astype(bool).ravel()[:n]
            cnt = int(m.sum())
            ratio = cnt / n if n > 0 else 0.0
            print(f"  {name:12s}: drop={cnt:7d} ({ratio:6.2%})")

        def _iou(name_ab, a, b):
            if a is None or b is None:
                print(f"  {name_ab:28s}: (None)")
                return
            a = a.astype(bool).ravel()[:n]
            b = b.astype(bool).ravel()[:n]
            inter = int((a & b).sum())
            union = int((a | b).sum())
            iou = inter / union if union > 0 else 0.0
            print(f"  {name_ab:28s}: inter={inter:7d}, union={union:7d}, IoU={iou:6.2%}")

        _set_stats("drop_head", dg)
        _set_stats("drop_wrap", dw)
        _set_stats("drop_legacy", legacy_drop)

        _iou("IoU(drop_head, drop_wrap)", dg, dw)
        if legacy_drop is not None:
            _iou("IoU(drop_head, drop_legacy)", dg, legacy_drop)
            _iou("IoU(drop_wrap, drop_legacy)", dw, legacy_drop)

            a = dg
            b = dw
            c = legacy_drop.astype(bool).ravel()[:n]
            inter3 = int((a & b & c).sum())
            union3 = int((a | b | c).sum())
            iou3 = inter3 / union3 if union3 > 0 else 0.0
            print(
                f"  triple-drop(head∧wrap∧legacy): "
                f"inter={inter3:7d}, union={union3:7d}, IoU={iou3:6.2%}"
            )

    return data


def save_slice_figure(
    data,
    out_path: Path,
    z=None,
    void_id: int = 17,
):
    """
    Visualize a single z-slice of LocalAgg(preds_localagg) vs Head(pred_head) (debug use).

    Panels:
      1: LocalAgg prediction
      2: Head prediction (if available)
      3: agree / diff (LocalAgg vs Head)
      4: density_localagg (if available)
      5: gt_occ (if available)
    """
    pred_la, pred_head = _get_preds_from_sample(data, void_id=void_id)
    dens_la = data.get("density_localagg", None)
    gt_occ = data.get("gt_occ", None)

    H, W, D = pred_la.shape
    if z is None:
        z = D // 2
    z = max(0, min(int(z), D - 1))

    has_gt = (gt_occ is not None) and (gt_occ.shape == pred_la.shape)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Compute the number of panels to be displayed
    n_panels = 1  # LocalAgg
    if pred_head is not None:
        n_panels += 1
    n_panels += 1  # agree_diff or LocalAgg-only
    if dens_la is not None:
        n_panels += 1
    if has_gt:
        n_panels += 1

    fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 4))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    ax_idx = 0

    # (1) LocalAgg prediction
    axes[ax_idx].imshow(pred_la[:, :, z])
    axes[ax_idx].set_title(f"LocalAgg pred (z={z})")
    axes[ax_idx].axis("off")
    ax_idx += 1

    # (2) Head prediction
    if pred_head is not None:
        axes[ax_idx].imshow(pred_head[:, :, z])
        axes[ax_idx].set_title("Head pred")
        axes[ax_idx].axis("off")
        ax_idx += 1

    # (3) agree / diff mask
    if pred_head is not None:
        same = (pred_la == pred_head) & (pred_la != void_id) & (pred_head != void_id)
        diff = (pred_la != pred_head) & (pred_la != void_id) & (pred_head != void_id)
        vis = np.zeros((*same[:, :, z].shape, 3), dtype=np.float32)
        vis[same[:, :, z]] = np.array([0.0, 1.0, 0.0])  # green = agree
        vis[diff[:, :, z]] = np.array([1.0, 0.0, 0.0])  # red   = different
        axes[ax_idx].imshow(vis)
        axes[ax_idx].set_title("agree (G) / diff (R)")
        axes[ax_idx].axis("off")
        ax_idx += 1
    else:
        # If Head is missing, show LocalAgg again instead
        axes[ax_idx].imshow(pred_la[:, :, z])
        axes[ax_idx].set_title("LocalAgg (no Head)")
        axes[ax_idx].axis("off")
        ax_idx += 1

    # (4) density_localagg
    if dens_la is not None:
        im = axes[ax_idx].imshow(dens_la[:, :, z])
        axes[ax_idx].set_title("density_localagg (z slice)")
        axes[ax_idx].axis("off")
        fig.colorbar(im, ax=axes[ax_idx], fraction=0.046)
        ax_idx += 1

    # (5) gt_occ
    if has_gt:
        axes[ax_idx].imshow(gt_occ[:, :, z])
        axes[ax_idx].set_title(f"GT occupancy (z={z})")
        axes[ax_idx].axis("off")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"saved slice fig -> {out_path}")

def aggregate_stats(
    pkls,
    void_id: int = 17,
    num_classes: int = 18,
):
    """
    Aggregate statistics over multiple .pkl files.

    - Only aggregate GT-based confusion / recall:
      * LocalAgg vs GT (required)
      * Head vs GT (only if pred_head is present)
    - Global Gaussian α/σ distribution, clamp/floor, coord, and gate stats are aggregated as well.
    """

    # GT-based confusion (row = GT, col = pred)
    hist_la_gt = np.zeros((num_classes, num_classes), dtype=np.int64)
    hist_head_gt = np.zeros((num_classes, num_classes), dtype=np.int64)

    total_gt_valid = 0
    total_la_gt_same = 0
    total_head_gt_same = 0

    # --- Global voxel / class count / occupancy confusion (for GT vs Pred) ---
    total_voxels_all = 0

    # Global per-class counts (GT / LocalAgg / Head, including void)
    total_gt_by_class = np.zeros(num_classes, dtype=np.int64)
    total_pred_la_by_class = np.zeros(num_classes, dtype=np.int64)
    total_pred_head_by_class = np.zeros(num_classes, dtype=np.int64)
    both_correct_cls = np.zeros(num_classes, dtype=np.int64)
    la_only_cls      = np.zeros(num_classes, dtype=np.int64)
    head_only_cls    = np.zeros(num_classes, dtype=np.int64)
    both_wrong_cls   = np.zeros(num_classes, dtype=np.int64)

    # Binary occupancy confusion (occupied = class != void_id)
    tp_occ_la = fp_occ_la = fn_occ_la = tn_occ_la = 0
    tp_occ_head = fp_occ_head = fn_occ_head = tn_occ_head = 0

    # Global Gaussian α/σ / gate aggregation
    alphas_pre_all = []
    alphas_post_all = []
    alphas_eff_all = []
    smax_pre_all = []
    smax_post_all = []
    orig_finite_all = []
    oor_before_all = []
    scale_raw_all = []
    scale_u_all = []
    scales_all = []
    maxprob_correct_all = []
    maxprob_wrong_all = []
    dens_correct_all = []
    dens_wrong_all = []

    # Gate stats per “top-class”
    interesting_classes = [3, 4, 10, 11, 14, 15, 16]
    gate_stats = {
        c: {"total": 0, "valid_hard": 0, "drop_soft": 0}
        for c in interesting_classes
    }

    percls_s_pre = {c: [] for c in interesting_classes}
    percls_s_post = {c: [] for c in interesting_classes}

    for p in pkls:
        data = load_pkl(p)
        pred_la, pred_head = _get_preds_from_sample(data, void_id=void_id)
        pred_la = pred_la.astype(np.int64)
        if pred_head is not None:
            pred_head = pred_head.astype(np.int64)

        # ---- LocalAgg / Head vs GT confusion ----
        gt_occ = data.get("gt_occ", None)
        if gt_occ is not None:
            gt = gt_occ.astype(np.int64)
            if gt.shape == pred_la.shape:
                valid_gt = gt != void_id

                # --- (A) Global voxel and class counts ---
                total_voxels_all += gt.size

                # GT class distribution
                gt_counts_all = np.bincount(gt.ravel(), minlength=num_classes)
                total_gt_by_class += gt_counts_all

                # LocalAgg prediction class distribution (over full grid)
                la_counts_all = np.bincount(pred_la.ravel(), minlength=num_classes)
                total_pred_la_by_class += la_counts_all

                # Head prediction class distribution (if available)
                if pred_head is not None:
                    head_counts_all = np.bincount(pred_head.ravel(), minlength=num_classes)
                    total_pred_head_by_class += head_counts_all

                # --- (B) Binary occupancy confusion (GT vs Pred) ---
                gt_occ_mask = gt != void_id
                la_occ_mask = pred_la != void_id

                tp_occ_la += int((gt_occ_mask & la_occ_mask).sum())
                fp_occ_la += int((~gt_occ_mask & la_occ_mask).sum())
                fn_occ_la += int((gt_occ_mask & ~la_occ_mask).sum())
                tn_occ_la += int((~gt_occ_mask & ~la_occ_mask).sum())

                if pred_head is not None:
                    head_occ_mask = pred_head != void_id
                    tp_occ_head += int((gt_occ_mask & head_occ_mask).sum())
                    fp_occ_head += int((~gt_occ_mask & head_occ_mask).sum())
                    fn_occ_head += int((gt_occ_mask & ~head_occ_mask).sum())
                    tn_occ_head += int((~gt_occ_mask & ~head_occ_mask).sum())

                # LocalAgg vs GT (multi-class)
                la_flat = pred_la[valid_gt].ravel()
                gt_flat = gt[valid_gt].ravel()
                for g, l in zip(gt_flat, la_flat):
                    if 0 <= g < num_classes and 0 <= l < num_classes:
                        hist_la_gt[g, l] += 1
                total_gt_valid += int(valid_gt.sum())
                total_la_gt_same += int((la_flat == gt_flat).sum())

                # Head vs GT (if available)
                if pred_head is not None:
                    head_flat = pred_head[valid_gt].ravel()
                    for g, h in zip(gt_flat, head_flat):
                        if 0 <= g < num_classes and 0 <= h < num_classes:
                            hist_head_gt[g, h] += 1
                    total_head_gt_same += int((head_flat == gt_flat).sum())

                    same_la   = (la_flat   == gt_flat)
                    same_head = (head_flat == gt_flat)

                    both_correct_mask = same_la & same_head
                    la_only_mask      = same_la & (~same_head)
                    head_only_mask    = same_head & (~same_la)
                    both_wrong_mask   = (~same_la) & (~same_head)

                    # Accumulate per-class counts based on GT (GT as reference)
                    if both_correct_mask.any():
                        cnt = np.bincount(
                            gt_flat[both_correct_mask],
                            minlength=num_classes
                        )
                        both_correct_cls += cnt
                    if la_only_mask.any():
                        cnt = np.bincount(
                            gt_flat[la_only_mask],
                            minlength=num_classes
                        )
                        la_only_cls += cnt
                    if head_only_mask.any():
                        cnt = np.bincount(
                            gt_flat[head_only_mask],
                            minlength=num_classes
                        )
                        head_only_cls += cnt
                    if both_wrong_mask.any():
                        cnt = np.bincount(
                            gt_flat[both_wrong_mask],
                            minlength=num_classes
                        )
                        both_wrong_cls += cnt

                logits_3d = data.get("logits_3d", None)
                dens_la = data.get("density_localagg", None)
                if (logits_3d is not None) and (dens_la is not None):
                    gt_valid = gt != void_id
                    la_correct = (pred_la == gt) & gt_valid
                    la_wrong   = (pred_la != gt) & gt_valid

                    if la_correct.any() or la_wrong.any():
                        logits = logits_3d.astype(np.float64)  # [C,H,W,D]
                        x = logits - logits.max(axis=0, keepdims=True)
                        exp_x = np.exp(x)
                        prob_la = exp_x / exp_x.sum(axis=0, keepdims=True)
                        maxprob = prob_la.max(axis=0)  # [H,W,D]

                        if la_correct.any():
                            maxprob_correct_all.append(maxprob[la_correct].ravel())
                            dens_correct_all.append(
                                dens_la[la_correct].astype(np.float64).ravel()
                            )
                        if la_wrong.any():
                            maxprob_wrong_all.append(maxprob[la_wrong].ravel())
                            dens_wrong_all.append(
                                dens_la[la_wrong].astype(np.float64).ravel()
                            )

            else:
                print(
                    f"[WARN] aggregate: gt_occ shape mismatch in {p.name}: "
                    f"gt_occ{gt.shape} vs preds_localagg{pred_la.shape} -> GT confusion skipped"
                )

        # ---- Gaussian-level aggregation ----
        alpha_pre = data.get("alpha_pre", None)
        alpha_post = data.get("alpha_post", None)
        alpha_eff = data.get("alpha_eff", None)
        smax_pre = data.get("smax_pre", None)
        smax_post = data.get("smax_post", None)
        orig_finite = data.get("orig_finite", None)
        oor_before = data.get("out_of_range_before", None)
        valid_hard = data.get("valid_hard", None)
        drop_soft = data.get("drop_soft", None)
        semantics = data.get("semantics", None)

        if alpha_pre is not None:
            alphas_pre_all.append(alpha_pre.astype(np.float64).ravel())
        if alpha_post is not None:
            alphas_post_all.append(alpha_post.astype(np.float64).ravel())
        if alpha_eff is not None:
            alphas_eff_all.append(alpha_eff.astype(np.float64).ravel())
        if smax_pre is not None:
            smax_pre_all.append(smax_pre.astype(np.float64).ravel())
        if smax_post is not None:
            smax_post_all.append(smax_post.astype(np.float64).ravel())
        if orig_finite is not None:
            orig_finite_all.append(orig_finite.astype(bool).ravel())
        if oor_before is not None:
            oor_before_all.append(oor_before.astype(bool).ravel())

        scale_raw = data.get("scale_raw", None)
        if scale_raw is not None:
            scale_raw_all.append(scale_raw.astype(np.float64).ravel())

        scale_u = data.get("scale_u", None)
        if scale_u is not None:
            scale_u_all.append(scale_u.astype(np.float64).ravel())

        scales_arr = data.get("scales", None)
        if scales_arr is not None:
            scales_all.append(scales_arr.astype(np.float64).ravel())

        # semantics / gate-based top-class stats
        if (semantics is not None) and (valid_hard is not None) and (drop_soft is not None):
            sem = semantics.astype(np.float64)
            if sem.ndim == 3:
                G_flat, C = sem.shape[0] * sem.shape[1], sem.shape[2]
                sem_flat = sem.reshape(G_flat, C)
            elif sem.ndim == 2:
                sem_flat = sem
            else:
                sem_flat = None

            if sem_flat is not None:
                top_cls = sem_flat.argmax(-1)  # [G_flat]
                vh = valid_hard.astype(bool).ravel()
                ds = drop_soft.astype(bool).ravel()

                n = min(top_cls.shape[0], vh.shape[0], ds.shape[0])
                top_cls = top_cls[:n]
                vh = vh[:n]
                ds = ds[:n]

                # gate_stats per class
                for c in interesting_classes:
                    mask_c = (top_cls == c)
                    total_c = int(mask_c.sum())
                    if total_c == 0:
                        continue
                    gate_stats[c]["total"] += total_c
                    gate_stats[c]["valid_hard"] += int((mask_c & vh).sum())
                    gate_stats[c]["drop_soft"] += int((mask_c & ds).sum())

                # per-class smax_pre/post (if available)
                if (smax_pre is not None) and (smax_post is not None):
                    s_pre_arr = smax_pre.astype(np.float64).ravel()
                    s_post_arr = smax_post.astype(np.float64).ravel()
                    m2 = min(n, s_pre_arr.shape[0], s_post_arr.shape[0])
                    if m2 > 0:
                        top_cls_s = top_cls[:m2]
                        s_pre_s = s_pre_arr[:m2]
                        s_post_s = s_post_arr[:m2]
                        for c in interesting_classes:
                            m = (top_cls_s == c)
                            if m.any():
                                percls_s_pre[c].append(s_pre_s[m])
                                percls_s_post[c].append(s_post_s[m])

    # ======================
    # LocalAgg / Head vs GT (aggregate)
    # ======================
    if total_gt_valid > 0:
        acc_la_gt = total_la_gt_same / total_gt_valid
        print(
            f"\n[Aggregated LocalAgg vs GT (excluding void)] "
            f"valid={total_gt_valid}, correct={total_la_gt_same}, acc={acc_la_gt:.4%}"
        )

        if total_head_gt_same > 0:
            acc_head_gt = total_head_gt_same / total_gt_valid
            print(
                f"[Aggregated Head    vs GT (excluding void)] "
                f"valid={total_gt_valid}, correct={total_head_gt_same}, acc={acc_head_gt:.4%}"
            )
        else:
            print("[Aggregated Head vs GT] Either there are no samples with pred_head or the aggregate is zero.")

        print("\n[Per-class vs GT (recall, GT as reference / excluding void)]")
        header = f"{'cls':>3s} | {'GT voxels':>9s} | {'LA correct':>10s} | {'LA_rec':>7s}"
        has_head_gt = hist_head_gt.sum() > 0
        if has_head_gt:
            header += f" | {'Head correct':>12s} | {'Head_rec':>8s}"
        print(header)
        print("-" * len(header))

        for c in range(num_classes):
            if c == void_id:
                continue
            supp_c = int(hist_la_gt[c].sum())
            if supp_c == 0:
                continue
            la_corr_c = int(hist_la_gt[c, c])
            la_rec = la_corr_c / supp_c
            line = f"{c:3d} | {supp_c:9d} | {la_corr_c:10d} | {la_rec:7.2%}"

            if has_head_gt:
                head_corr_c = int(hist_head_gt[c, c])
                head_rec = head_corr_c / supp_c if supp_c > 0 else 0.0
                line += f" | {head_corr_c:12d} | {head_rec:8.2%}"
            print(line)

        # --- GLOBAL occupancy ratio (GT / LocalAgg / Head) ---
        if total_voxels_all > 0:
            print("\n[GLOBAL occupancy ratio (over all voxels with GT)]")

            gt_labeled_global = int(total_gt_by_class.sum() - total_gt_by_class[void_id])
            gt_void_global = int(total_gt_by_class[void_id])
            print(f"  GT   labeled (!= {void_id:2d}): {gt_labeled_global:9d} ({gt_labeled_global/total_voxels_all:6.2%})")
            print(f"  GT   void    (== {void_id:2d}): {gt_void_global:9d} ({gt_void_global/total_voxels_all:6.2%})")

            la_labeled_global = int(total_pred_la_by_class.sum() - total_pred_la_by_class[void_id])
            la_void_global = int(total_pred_la_by_class[void_id])
            print(f"  LA   labeled (!= {void_id:2d}): {la_labeled_global:9d} ({la_labeled_global/total_voxels_all:6.2%})")
            print(f"  LA   void    (== {void_id:2d}): {la_void_global:9d} ({la_void_global/total_voxels_all:6.2%})")

            if total_pred_head_by_class.sum() > 0:
                head_labeled_global = int(total_pred_head_by_class.sum() - total_pred_head_by_class[void_id])
                head_void_global = int(total_pred_head_by_class[void_id])
                print(f"  Head labeled (!= {void_id:2d}): {head_labeled_global:9d} ({head_labeled_global/total_voxels_all:6.2%})")
                print(f"  Head void    (== {void_id:2d}): {head_void_global:9d} ({head_void_global/total_voxels_all:6.2%})")
        else:
            print("\n[GLOBAL occupancy ratio] total_voxels_all == 0 (no GT)")

        # --- GLOBAL binary occupancy metrics (occupied = class != void_id) ---
        def _print_occ_metrics_global(tag, tp, fp, fn, tn):
            total = tp + fp + fn + tn
            if total == 0:
                print(f"  {tag}: no voxels")
                return
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            iou  = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
            acc  = (tp + tn) / total
            print(
                f"  {tag}: TP={tp} FP={fp} FN={fn} TN={tn} | "
                f"prec={prec:.4f} rec={rec:.4f} IoU={iou:.4f} acc={acc:.4f}"
            )

        if total_voxels_all > 0:
            print("\n[GLOBAL binary occupancy metrics (occupied = class != void_id)]")
            _print_occ_metrics_global("LocalAgg vs GT", tp_occ_la, fp_occ_la, fn_occ_la, tn_occ_la)
            if (tp_occ_head + fp_occ_head + fn_occ_head + tn_occ_head) > 0:
                _print_occ_metrics_global("Head    vs GT", tp_occ_head, fp_occ_head, fn_occ_head, tn_occ_head)

        # --- GLOBAL per-class volume ratio (% of all voxels with GT) ---
        if total_voxels_all > 0:
            print("\n[GLOBAL per-class volume ratio (GT vs LocalAgg vs Head, % of all voxels)]")
            has_head_vol = total_pred_head_by_class.sum() > 0
            header = f"{'cls':>3s} | {'GT_cnt':>9s} | {'GT_%':>6s} | {'LA_cnt':>9s} | {'LA_%':>6s}"
            if has_head_vol:
                header += f" | {'Head_cnt':>9s} | {'Head_%':>7s}"
            print(header)
            print("-" * len(header))

            for c in range(num_classes):
                cnt_gt_c = int(total_gt_by_class[c])
                cnt_la_c = int(total_pred_la_by_class[c])
                cnt_head_c = int(total_pred_head_by_class[c]) if has_head_vol else 0
                if (cnt_gt_c + cnt_la_c + cnt_head_c) == 0:
                    continue
                ratio_gt = cnt_gt_c / total_voxels_all
                ratio_la = cnt_la_c / total_voxels_all
                line = f"{c:3d} | {cnt_gt_c:9d} | {ratio_gt:6.2%} | {cnt_la_c:9d} | {ratio_la:6.2%}"
                if has_head_vol:
                    ratio_head = cnt_head_c / total_voxels_all
                    line += f" | {cnt_head_c:9d} | {ratio_head:7.2%}"
                print(line)

        # --- GLOBAL per-class IoU / Precision / Recall (LocalAgg) ---
        print("\n[Per-class IoU / Precision / Recall (LocalAgg, excluding void)]")
        header = f"{'cls':>3s} | {'TP':>9s} | {'FP':>9s} | {'FN':>9s} | {'IoU':>7s} | {'Prec':>7s} | {'Rec':>7s}"
        print(header)
        print("-" * len(header))
        for c in range(num_classes):
            if c == void_id:
                continue
            tp_c = int(hist_la_gt[c, c])
            gt_c = int(hist_la_gt[c].sum())         # row sum (GT = c)
            pred_c = int(hist_la_gt[:, c].sum())    # col sum (Pred = c)
            if (gt_c + pred_c) == 0:
                continue
            fn_c = gt_c - tp_c
            fp_c = pred_c - tp_c
            denom_iou = tp_c + fp_c + fn_c
            iou_c = tp_c / denom_iou if denom_iou > 0 else 0.0
            prec_c = tp_c / pred_c if pred_c > 0 else 0.0
            rec_c = tp_c / gt_c if gt_c > 0 else 0.0
            print(
                f"{c:3d} | {tp_c:9d} | {fp_c:9d} | {fn_c:9d} | "
                f"{iou_c:7.2%} | {prec_c:7.2%} | {rec_c:7.2%}"
            )

        print("\n[Per-class triple stats (GT as reference, excluding void)]")
        header = (
            f"{'cls':>3s} | {'GT vox':>9s} | "
            f"{'both_ok':>9s} | {'la_only':>9s} | "
            f"{'head_only':>9s} | {'both_wrong':>11s}"
        )
        print(header)
        print("-" * len(header))

        for c in range(num_classes):
            if c == void_id:
                continue
            gt_c = int(hist_la_gt[c].sum())  # all voxels where GT = c
            if gt_c == 0:
                continue

            bc = int(both_correct_cls[c])
            lo = int(la_only_cls[c])
            ho = int(head_only_cls[c])
            bw = int(both_wrong_cls[c])

            print(
                f"{c:3d} | {gt_c:9d} | "
                f"{bc:9d} | {lo:9d} | {ho:9d} | {bw:11d}"
            )

        if hist_head_gt.sum() > 0:
            print("\n[Per-class IoU / Precision / Recall (Head, excluding void)]")
            header = f"{'cls':>3s} | {'TP':>9s} | {'FP':>9s} | {'FN':>9s} | {'IoU':>7s} | {'Prec':>7s} | {'Rec':>7s}"
            print(header)
            print("-" * len(header))
            for c in range(num_classes):
                if c == void_id:
                    continue
                tp_c = int(hist_head_gt[c, c])
                gt_c = int(hist_head_gt[c].sum())        # row sum (GT = c)
                pred_c = int(hist_head_gt[:, c].sum())   # col sum (Pred = c)
                if (gt_c + pred_c) == 0:
                    continue
                fn_c = gt_c - tp_c
                fp_c = pred_c - tp_c
                denom_iou = tp_c + fp_c + fn_c
                iou_c  = tp_c / denom_iou if denom_iou > 0 else 0.0
                prec_c = tp_c / pred_c  if pred_c  > 0 else 0.0
                rec_c  = tp_c / gt_c    if gt_c    > 0 else 0.0
                print(
                    f"{c:3d} | {tp_c:9d} | {fp_c:9d} | {fn_c:9d} | "
                    f"{iou_c:7.2%} | {prec_c:7.2%} | {rec_c:7.2%}"
                )
        else:
            print("\n[Per-class IoU / Precision / Recall (Head)] hist_head_gt is empty (no Head predictions).")
    else:
        print("\n[Aggregated vs GT] total_gt_valid is 0 (no valid GT voxels).")

    # ======================
    # Global Gaussian α/σ distribution / clamp/floor / coord / gate
    # ======================
    def _alpha_bucket_stats_global(title, arr, weak_thr=5e-4, mid_thr=5e-3):
        print(f"\n[{title}]")
        _stat_1d(title, arr)
        total = arr.size
        n_weak = int((arr < weak_thr).sum())
        n_mid = int(((arr >= weak_thr) & (arr < mid_thr)).sum())
        n_strong = int((arr >= mid_thr).sum())
        print(
            f"  weak(<{weak_thr:g}):   {n_weak:9d} ({n_weak/total:6.2%})\n"
            f"  mid([{weak_thr:g},{mid_thr:g})): {n_mid:9d} ({n_mid/total:6.2%})\n"
            f"  strong(≥{mid_thr:g}):  {n_strong:9d} ({n_strong/total:6.2%})"
        )

    if alphas_pre_all:
        alphas_pre = np.concatenate(alphas_pre_all)
        _alpha_bucket_stats_global("GLOBAL alpha_pre", alphas_pre)

    if alphas_post_all:
        alphas_post = np.concatenate(alphas_post_all)
        _alpha_bucket_stats_global("GLOBAL alpha_post", alphas_post)

    if alphas_eff_all:
        alphas_eff = np.concatenate(alphas_eff_all)
        _alpha_bucket_stats_global("GLOBAL alpha_eff", alphas_eff)

    if smax_pre_all or smax_post_all:
        print("\n[GLOBAL Gaussian s_max stats]")
        s_list = []
        if smax_pre_all:
            s_list.append(np.concatenate(smax_pre_all))
        if smax_post_all:
            s_list.append(np.concatenate(smax_post_all))
        smax = np.concatenate(s_list)
        _stat_1d("smax_all", smax)

    # GLOBAL clamp/floor/coord stats
    if smax_pre_all and smax_post_all:
        s_pre = np.concatenate(smax_pre_all).astype(np.float64)
        s_post = np.concatenate(smax_post_all).astype(np.float64)
        diff_s = s_pre - s_post
        hit_mask = diff_s > 1e-6
        hit_ratio = hit_mask.mean() if hit_mask.size > 0 else 0.0
        print("\n[GLOBAL scale cap stats]")
        print(f"  hit_ratio = {hit_ratio:.2%} ( {hit_mask.sum()} / {hit_mask.size} )")
        if hit_mask.any():
            _alpha_bucket_stats_global("GLOBAL smax_overshoot", diff_s[hit_mask])

    if alphas_pre_all and alphas_post_all:
        a_pre = np.concatenate(alphas_pre_all).astype(np.float64)
        a_post = np.concatenate(alphas_post_all).astype(np.float64)
        diff_a = a_post - a_pre
        floor_mask = diff_a > 1e-8
        floor_ratio = floor_mask.mean() if floor_mask.size > 0 else 0.0
        print("\n[GLOBAL alpha floor stats]")
        print(f"  floor_ratio = {floor_ratio:.2%} ( {floor_mask.sum()} / {floor_mask.size} )")
        if floor_mask.any():
            _alpha_bucket_stats_global("GLOBAL alpha_raise", diff_a[floor_mask])

    if orig_finite_all or oor_before_all:
        print("\n[GLOBAL coord sanitization stats]")
    if orig_finite_all:
        finite = np.concatenate(orig_finite_all).astype(bool).ravel()
        finite_ratio = finite.mean() if finite.size > 0 else 0.0
        print(
            f"  finite_ratio (before nan_to_num) = {finite_ratio:.2%} "
            f"( {finite.sum()} / {finite.size} )"
        )
    if oor_before_all:
        oor = np.concatenate(oor_before_all).astype(bool).ravel()
        oor_ratio = oor.mean() if oor.size > 0 else 0.0
        print(
            f"  out_of_range_before clamp = {oor_ratio:.2%} "
            f"( {oor.sum()} / {oor.size} )"
        )
    if scale_raw_all:
        sr_all = np.concatenate(scale_raw_all)
        print("\n[GLOBAL scale_raw stats]")
        _stat_1d("scale_raw", sr_all)

    if scale_u_all:
        su_all = np.concatenate(scale_u_all)
        print("\n[GLOBAL scale_u stats]")
        _stat_1d("scale_u", su_all)

    if scales_all:
        sc_all = np.concatenate(scales_all)
        print("\n[GLOBAL scales stats]")
        _stat_1d("scales", sc_all)

    # --- Global LocalAgg confidence / density vs GT correctness ---
    if maxprob_correct_all and maxprob_wrong_all:
        mp_c = np.concatenate(maxprob_correct_all)
        mp_w = np.concatenate(maxprob_wrong_all)
        print("\n[GLOBAL LocalAgg max prob vs GT]")
        _stat_1d("la_p_gt_correct_all", mp_c)
        _stat_1d("la_p_gt_wrong_all",  mp_w)

    if dens_correct_all and dens_wrong_all:
        d_c = np.concatenate(dens_correct_all)
        d_w = np.concatenate(dens_wrong_all)
        print("\n[GLOBAL density_localagg vs GT]")
        _stat_1d("dens_gt_correct_all", d_c)
        _stat_1d("dens_gt_wrong_all",   d_w)

    # Gate stats per top-class
    print("\n[Gaussian gate stats per top-class (from semantics argmax)]")
    for c in interesting_classes:
        g = gate_stats[c]
        if g["total"] == 0:
            continue
        vh_ratio = g["valid_hard"] / g["total"]
        ds_base = max(g["valid_hard"], 1)
        ds_ratio = g["drop_soft"] / ds_base
        print(f"Class {c}: total={g['total']}")
        print(f"  valid_hard: {g['valid_hard']} ({vh_ratio:.2%} of class-{c} Gaussians)")
        print(f"  drop_soft:  {g['drop_soft']} ({ds_ratio:.2%} of class-{c} valid_hard)")

    # Per-class σ stats (pre vs post)
    print("\n[Per-top-class Gaussian σ stats (smax_pre → smax_post)]")
    for c in interesting_classes:
        if not percls_s_pre[c]:
            continue
        s_pre_c = np.concatenate(percls_s_pre[c]).astype(np.float64)
        s_post_c = np.concatenate(percls_s_post[c]).astype(np.float64)
        diff_c = s_pre_c - s_post_c
        hit_c = diff_c > 1e-6

        print(f"\nClass {c}: n_gauss={s_pre_c.size}")
        _stat_1d(f"s_pre[c={c}]", s_pre_c)
        _stat_1d(f"s_post[c={c}]", s_post_c)
        if hit_c.any():
            _stat_1d(f"s_over[c={c}]", diff_c[hit_c])
            hit_ratio_c = hit_c.mean()
            print(f"  cap_hit_ratio[c={c}] = {hit_ratio_c:.2%}")
        else:
            print(f"  cap_hit_ratio[c={c}] = 0.00% (no overshoot)")

    # === Planar vs Object σ bucket stats (pre-cap) ===
    def _gather_group_s(percls_dict, cls_list):
        arrs = []
        for cls in cls_list:
            if percls_dict.get(cls):
                arrs.extend(percls_dict[cls])
        if not arrs:
            return None
        return np.concatenate(arrs).astype(np.float64)

    def _sigma_bucket_stats(name, arr, small_thr=0.08, mid_thr=0.26):
        print(f"\n[{name}]")
        _stat_1d(name, arr)
        total = arr.size
        n_small = int((arr < small_thr).sum())
        n_mid   = int(((arr >= small_thr) & (arr < mid_thr)).sum())
        n_large = int((arr >= mid_thr).sum())
        print(
            f"  small(<{small_thr:g}):   {n_small:9d} ({n_small/total:6.2%})\n"
            f"  mid([{small_thr:g},{mid_thr:g})): {n_mid:9d} ({n_mid/total:6.2%})\n"
            f"  large(≥{mid_thr:g}):     {n_large:9d} ({n_large/total:6.2%})"
        )

    planar_classes = [11, 14, 15, 16]
    object_classes = [3, 4, 10]

    print("\n[Planar vs Object σ bucket stats (pre-cap)]")
    s_pre_planar = _gather_group_s(percls_s_pre, planar_classes)
    s_pre_object = _gather_group_s(percls_s_pre, object_classes)
    if s_pre_planar is not None:
        _sigma_bucket_stats("σ_pre_planar", s_pre_planar)
    else:
        print("  σ_pre_planar: (no planar Gaussians collected)")
    if s_pre_object is not None:
        _sigma_bucket_stats("σ_pre_object", s_pre_object)
    else:
        print("  σ_pre_object: (no object Gaussians collected)")

    return hist_la_gt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cmp-dir",
        default="outputs_compare",
        help="Directory path containing the pkl files.",
    )
    parser.add_argument(
        "--idx",
        type=int,
        default=0,
        help="sample_idx used in the file name (e.g., 0, 600, 1200...).",
    )
    parser.add_argument(
        "--save-fig",
        action="store_true",
        help="Whether to save an intermediate z-slice as a PNG.",
    )
    parser.add_argument(
        "--z",
        type=int,
        default=None,
        help="z index to visualize (default: middle depth).",
    )
    parser.add_argument(
        "--void-id",
        type=int,
        default=17,
        help="Class id to be treated as free/void.",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=18,
        help="Total number of classes (for confusion aggregation).",
    )
    parser.add_argument(
        "--aggregate",
        action="store_true",
        help="Print only GT-based aggregate stats (confusion/recall + gate/α/σ) over all pkls in cmp-dir.",
    )
    parser.add_argument(
        "--no-detail",
        action="store_true",
        help="Skip detailed stats (min/quantiles, etc.) in single-sample summary.",
    )
    args = parser.parse_args()

    cmp_dir = Path(args.cmp_dir)
    pkls = list(cmp_dir.glob("*.pkl"))
    if not pkls:
        raise SystemExit(f"No .pkl in {cmp_dir}")

    # Aggregate mode: use all pkls
    if args.aggregate:
        aggregate_stats(
            pkls,
            void_id=args.void_id,
            num_classes=args.num_classes,
        )
        return

    # Single-sample mode: use --idx as sample_idx (e.g., 0, 600, 1200...)
    target = cmp_dir / f"{args.idx}.pkl"
    if not target.exists():
        raise SystemExit(f"{target} not found in {cmp_dir}")

    p = target
    print(f"total {len(pkls)} files, inspecting sample_idx={args.idx}: {p}")

    data = summarize_sample(
        p,
        void_id=args.void_id,
        num_classes=args.num_classes,
        detailed=not args.no_detail,
    )

    if args.save_fig:
        out_png = cmp_dir / "vis" / f"{p.stem}_z{args.z if args.z is not None else 'mid'}.png"
        save_slice_figure(
            data,
            out_png,
            z=args.z,
            void_id=args.void_id,
        )


if __name__ == "__main__":
    main()
