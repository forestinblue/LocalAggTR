# Third-party: Local Aggregation CUDA op

`localagg_tr.LocalAggWrapper` calls the Local Aggregation CUDA op from
[GaussianFormer](https://github.com/huang-yh/GaussianFormer). **This op is not part of this
repository and not my work.** It is not redistributed here; install it from the original repo.

## Install

The version used in this project is byte-identical to GaussianFormer commit `b7e22bf`,
directory `model/head/localagg/`.

```bash
git clone https://github.com/huang-yh/GaussianFormer.git
cd GaussianFormer
git checkout b7e22bf
cd model/head/localagg
pip install -e .          # builds the `local_aggregate` package (needs CUDA + nvcc)
```

Check:

```bash
python -c "from local_aggregate import LocalAggregator; print('ok')"
```

## License

- The op's source files carry this header:
  > Copyright (C) 2023, Inria, GRAPHDECO research group. All rights reserved.
  > This software is free for non-commercial, research and evaluation use under the terms of the LICENSE.md file.

  The code is derived from the [Inria 3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting)
  rasterizer; see its [LICENSE.md](https://github.com/graphdeco-inria/gaussian-splatting/blob/main/LICENSE.md).
- The `LICENSE` file in the GaussianFormer repository is empty (as of `b7e22bf`).
- Commercial use of the op is not permitted under the Inria license. The license of this
  repository (see `../LICENSE`) covers only the code under `src/`, `configs/` and `scripts/`.
