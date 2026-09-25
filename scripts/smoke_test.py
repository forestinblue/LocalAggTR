"""
Smoke test for the released head module on synthetic Gaussians.
Needs only public dependencies: a CUDA GPU, upstream GaussTR on PYTHONPATH, and the
GaussianFormer `local_aggregate` op (third_party/README.md). No dataset, no checkpoint.

What it checks, for both backends (LocalAgg, GaussianVoxelizerCompat):
  1. an opaque Gaussian of class `car` placed at the centre of voxel (100, 100, 8)
     makes that voxel occupied with class `car`;
  2. a Gaussian outside pc_range is flagged by the head's geometry gate;
  3. output shape is [1, 200, 200, 16].

It does NOT check accuracy on real data. results/NUMBERS.md numbers need the lab model.

Usage:
    PYTHONPATH=<GaussTR clone>:src python scripts/smoke_test.py
"""

import math
import time
from pathlib import Path

import torch
import torch.nn as nn
from mmengine.config import Config

import localagg_tr  # noqa: F401  (registers LocalAggWrapper / GaussianVoxelizerCompat)
from localagg_tr import DualBackendVoxelHeadMixin

CONFIGS = Path(__file__).resolve().parent.parent / 'configs'
CAR, FREE = 4, 17
TARGET = (100, 100, 8)


class TinyHead(DualBackendVoxelHeadMixin, nn.Module):

    def __init__(self, voxelizer):
        super().__init__()
        self.prompt_denoising = True
        self.init_dual_backend(voxelizer, tau_quantile=0.90)


def synthetic_gaussians(device):
    # voxel centre = pc_min + (index + 0.5) * voxel_size
    pc_min, vx = torch.tensor([-40., -40., -1.]), 0.4
    target = pc_min + (torch.tensor(TARGET, dtype=torch.float32) + 0.5) * vx
    means = torch.stack([
        target,                          # 0: the car Gaussian
        torch.tensor([100., 0., 2.]),    # 1: outside pc_range -> must be gated
        torch.tensor([-20., 15., 0.]),   # 2: background, class vegetation
    ])[None, None].to(device)            # [B=1, N=1, G=3, 3]

    # raw scales such that 5e-4 + softplus(raw) ~= 0.25 m (below the 0.26 cap)
    raw = math.log(math.exp(0.2495) - 1)
    scales_raw = torch.full((1, 1, 3, 3), raw, device=device)
    opacities_raw = torch.tensor([0.9, 0.9, 0.9], device=device).view(1, 1, 3, 1)

    semantics = torch.zeros(1, 1, 3, 18, device=device)
    semantics[..., 0, CAR] = 10.
    semantics[..., 1, CAR] = 10.
    semantics[..., 2, 16] = 10.
    return means, opacities_raw, scales_raw, semantics


def run(backend_cfg, device):
    head = TinyHead(Config.fromfile(CONFIGS / backend_cfg).voxelizer).to(device).eval()
    means, opacities_raw, scales_raw, semantics = synthetic_gaussians(device)

    with torch.no_grad():
        means, opacities, scales, debug = head.sanitize_gaussians(means, opacities_raw, scales_raw)
        covariances = torch.diag_embed(scales ** 2)          # axis-aligned Σ
        info = head.make_gaussian_info(means, opacities, semantics, covariances, scales,
                                       opacities_raw, debug)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        preds, info = head.predict_occupancy(means, opacities, semantics, scales, covariances,
                                             None, info)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0

    occupied = int((preds != FREE).sum())
    checks = {
        'shape [1,200,200,16]': tuple(preds.shape) == (1, 200, 200, 16),
        'target voxel = car': int(preds[(0, *TARGET)]) == CAR,
        'out-of-range Gaussian gated': bool(info['drop_geo_head'][0, 0, 1]),
        'in-range Gaussians kept': not bool(info['drop_geo_head'][0, 0, [0, 2]].any()),
    }
    print(f'[{head._voxel_backend}] occupied voxels={occupied}  predict time={dt * 1e3:.1f} ms')
    for name, ok in checks.items():
        print(f'   {"PASS" if ok else "FAIL"}  {name}')
    return all(checks.values())


def main():
    assert torch.cuda.is_available(), 'the Local Aggregation op needs a CUDA GPU'
    device = torch.device('cuda')
    ok = [run(cfg, device) for cfg in ('voxelizer_localagg.py', 'voxelizer_gaussvox.py')]
    print('ALL PASS' if all(ok) else 'SOME CHECKS FAILED')
    raise SystemExit(0 if all(ok) else 1)


if __name__ == '__main__':
    main()
