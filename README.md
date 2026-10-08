# LangGraph-MAR

[![Paper](https://img.shields.io/badge/Paper-Phys.%20Med.%20Biol.-b31b1b)](https://iopscience.iop.org/article/10.1088/1361-6560/ae7ec4)
[![DOI](https://img.shields.io/badge/DOI-10.1088%2F1361--6560%2Fae7ec4-1f6feb)](https://doi.org/10.1088/1361-6560/ae7ec4)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

**Adaptive Thresholding for CT Metal Artifact Reduction via LangGraph**

Yeonghyeon Kim, Kyungsang Kim, Dongheon Lee

This repository contains the official code for *LangGraph-MAR*, a graph-based,
QA-guided adaptive framework for metal artifact reduction (MAR) in CT. The
framework combines deep-learning sinogram inpainting, L1-based iterative metal
reconstruction, and a quality-assurance (QA) classifier inside a cyclic
[LangGraph](https://github.com/langchain-ai/langgraph) workflow that
automatically searches for the per-case optimal metal threshold.

> Raw data and trained checkpoints are **not** distributed in this repository.
> See [Data preparation](#data-preparation) for how to obtain the dataset.

---

## Repository structure

```
LangGraphMAR/
├── code/                       # Main reproduction code (run scripts from here)
│   ├── utils/                  # ct.py (geometry SoT), metric.py (eval SoT),
│   │                           # dataset.py, projection.py, graph.py, fastproj.py, ir_modes.py, ...
│   ├── training/               # train_inpainting / train_enhancement / train_gc / train_*_mar
│   ├── scripts/                # run_threshold_sweep.py + metric scripts
│   ├── pl_modules/             # PyTorch Lightning modules
│   ├── loss/ configs/          # losses, Hydra configs (model defs in utils/models.py)
│   ├── paper_results/          # metric / significance scripts
│   ├── index/                  # train/val/test split indices (*.pkl)
│   ├── CT_recon_fanbeam_python_openmp/   # C extension (forward/back projection)
│   │   └── recon_fast/         # opt-in order-preserving fast projector (see below)
│   ├── AAPM_datachallenge/     # AAPM geometry / simulation references
│   ├── param.yaml              # per-anatomy normalization statistics
│   └── requirements.txt
├── baselines/                  # InDuDoNet+, NMAR, MDT (code only)
├── data/                       # dataset goes here (see data/README.md); not committed
└── LICENSE
```

## Installation

Tested with **Python 3.10**, **PyTorch 2.3.0**, **LangGraph 1.0.2** on Linux
(Ubuntu) with an NVIDIA GPU.

```bash
# 1. Create an environment (conda or venv)
conda create -n langgraphmar python=3.10 -y
conda activate langgraphmar

# 2. Install Python dependencies
pip install -r code/requirements.txt

# 3. Build the CT reconstruction C extension (OpenMP)
cd code/CT_recon_fanbeam_python_openmp
python setup_linux.py build_ext --inplace      # macOS: python setup_mac.py build_ext --inplace
# Make the built module importable as `recon` from code/.
# The pipeline does `import recon`, so place the compiled extension on the path:
cp recon*.so ../          # copy the built recon*.so into code/
cd ../..
```

### Experiment logging (optional)

The training scripts log to Weights & Biases through PyTorch Lightning's
`WandbLogger`. W&B is **optional** and not required to reproduce results:

- **To enable it**, authenticate first with `wandb login`, or export your key as
  `WANDB_API_KEY` before running a training script.
- **To run without W&B** (no account, no uploads), set the standard `WANDB_MODE`
  environment variable: `WANDB_MODE=disabled` turns logging off entirely, while
  `WANDB_MODE=offline` records runs locally without uploading. For example:

```bash
WANDB_MODE=disabled python training/train_inpainting.py --anatomy body
```

## Data preparation

This project uses the **AAPM CT-MAR Grand Challenge** dataset. The raw data is
not redistributed here. After obtaining it (see [data/README.md](data/README.md)),
arrange it as:

```
data/01_raw/{body,head}/{Baseline,Target,LI,Mask}/*.raw
```

and create a `dataset` symlink at the repository root (several scripts expect
`dataset/` to point at the raw data):

```bash
ln -s data/01_raw dataset
```

Train/val/test split indices are provided under
[`code/index/`](code/index/) (body: 7424/2475/2475, head: 975/325/326) and are
shared across **all** models for a fair comparison.

## Usage

All commands below are run from the `code/` directory.

```bash
cd code
```

### Training

```bash
python training/train_inpainting.py  --anatomy body      # sinogram inpainting (U-Net / EfficientNet-B4)
python training/train_enhancement.py --anatomy body      # image enhancement (UFormer)
python training/train_gc.py                              # QA "ground-checking" classifier (ResNet-18)
python training/train_image_domain_mar.py --anatomy body # image-domain MAR baselines (UNet/UFormer/...)
```

Checkpoints are expected under `<repo>/checkpoints/`
(`inpainting_{body,head}.ckpt`, `enhancement_{body,head}.ckpt`, `gc.ckpt`).
Train them with the scripts above, or place your own checkpoints there.

### Full LangGraph-MAR pipeline (inference + threshold sweep)

```bash
python scripts/run_threshold_sweep.py \
    --anatomy body --min_threshold 0.01 --max_threshold 0.40 --step 0.01 \
    --output_dir results_threshold_sweep
```

The adaptive search starts at threshold **0.40**, decreases by **0.01** per
trial, keeps the best reconstruction by QA score, and stops after **5**
consecutive non-improving trials (`total_trials = 5`).

Metric (re)computation helpers:

```bash
python scripts/recalculate_metrics.py   --anatomy body
python scripts/merge_threshold_metrics.py --anatomy body
```

### Baselines

Image-domain baselines (UNet, UFormer, Restormer, NAFNet) are trained via the
`code/training/train_mar_*.py` scripts. External baselines live in `baselines/`:

```bash
# NMAR / MDT (sinogram-domain, run from code/)
python ../baselines/NMAR/eval_nmar.py --anatomy body
python ../baselines/MDT/eval_mdt.py  --anatomy body

# InDuDoNet+ (use its own environment, run from its directory)
source baselines/InDuDoNet/activate.sh
python eval.py --anatomy body --model_path <your_indudonet_checkpoint>
```

## Evaluation protocol (source of truth)

| Aspect | Definition | File |
|--------|------------|------|
| Metrics | PSNR, SSIM, FSIM, (N)RMSE | `code/utils/metric.py` (`ImageQualityEvaluator`) |
| Metal region | excluded via the ground-truth metal mask | `code/utils/metric.py` |
| Body eval mask | 470 mm diameter circular mask (wider than the 400 mm FOV) | `code/utils/metric.py` |
| Head eval mask | full image (no mask) | `code/utils/metric.py` |
| Scanner geometry | SID/SDD/detector/FOV (see below) | `code/utils/ct.py` |
| Splits | shared train/val/test indices | `code/index/` |

### Scanner geometry

| Parameter | Value |
|-----------|-------|
| Source-to-isocenter (SID) | 550.0 mm |
| Source-to-detector (SDD) | 950.0 mm |
| Detector channels | 900 (1.0 mm spacing, offset -1.25) |
| Views / rotation | 1000 (360 deg) |
| Sinogram size | 1000 x 900 (views x detectors) |
| Image size | 512 x 512 |
| Reconstruction FOV | head 220.16 mm, body 400 mm |

## Fast projection (optional)

The published pipeline is unchanged. Three opt-in additions make the CPU-bound iterative reconstruction (IR) faster without
changing the method or, where stated, the results.

**`recon_fast` (drop-in projector).** Same operator and float32 arithmetic as `recon.c`; only the order of independent work
changes. Build it next to `recon` and switch it on before building the graph:

```bash
cd code/CT_recon_fanbeam_python_openmp/recon_fast
python setup_fast.py build_ext --inplace      # needs an AVX2 CPU; flags: -O3 -mavx2 -mno-fma -ffp-contract=off
cp recon_fast*.so ../../                      # next to recon*.so in code/
cd ../..
python scripts/verify_fastproj.py             # exact equivalence checks against `recon` (OMP_NUM_THREADS=4 by default)
```

```python
from utils.fastproj import install_fastproj
install_fastproj()          # rebinds fp/bp in utils.projection / algorithm / graph; P=128 by default
```

- **FP** is bit-identical to the original for any thread count.
- **BP**: the original is an OpenMP static-schedule reduction whose per-thread copies are added in arrival order, so its
  output varies from run to run (relative L2 about 5e-8 at 4 threads, about 2e-7 at 128). `recon_fast` reproduces the same
  `P` chunks bit for bit and adds them in a fixed order, so its output is one member of the original's run-to-run result set.
  `P` must equal the original's team size to share its partition; libgomp uses every visible core when `OMP_NUM_THREADS` is
  unset (128 on the machine used for the experiments), hence the default `P=128`. For that case all 128 chunk partial sums
  were compared with the original's (single-chunk inputs, 128 runs) and matched exactly on the checked slice;
  `verify_fastproj.py` repeats the check for a small geometry at `T=4`.
- **Threads:** with `recon_fast` you may set `OMP_NUM_THREADS` freely, the result does not change. With the original
  `recon` do **not** set it: that moves the chunk boundaries and changes the result (about 4e-7 relative to the 128-thread run).
- **Speed** (shared, CPU-saturated 20-core container; read the ratios, not the seconds): one IR call (11 FP + 12 BP) took a
  median 81.7 s with the original at 4 threads and 43-47 s with `recon_fast` (about 1.8x); at 12 threads 37.1-37.8 s vs
  20.6-20.8 s. Single calls: FP 1.7-3.2x, BP 1.2-2.1x. A pixel-gather BP (about 5x slower) and a row-stripe BP (no speed-up) were tried and dropped; neither
  reproduces the original's chunk partition.

**`utils/gpuproj.py` (GPU projector).** The same FP and BP as `recon.c` on a CUDA GPU (Triton), switched on the same way:

```python
from utils.gpuproj import install_gpuproj
install_gpuproj()           # rebinds fp/bp in utils.projection / algorithm / graph
```

Needs a CUDA GPU, PyTorch with Triton, and a C compiler Triton can find (set `CC` if the machine has no system gcc); nothing to build.
The geometry comes from the same `param` dict as the original wrappers, so any pixel size works. Check it with
`python scripts/verify_gpuproj.py` (synthetic phantoms, two pixel sizes, with negative controls).

- **FP** is bit-identical to `recon.FP` (every ray is independent, so the original is deterministic).
- **BP** is the same ray-driven scatter, but it is **not** bit-identical and **not** a member of the original's run-to-run result set:
  the original sums float32 per-thread partial images, the kernel scatters with atomic adds in an arbitrary order. Accumulating in
  float32 first gave a difference about 8x the original's own run-to-run noise, so the accumulation is float64 (the products stay
  float32). With that the difference to `recon.BP` is as large as two original runs differ from each other (about 3e-7 of the
  image maximum; 0.012 HU on a body slice). Use `recon_fast` if you need exact membership in the original's result set.
- **Where it was checked:** the synthetic phantoms in `verify_gpuproj.py`, and 16 CT slices with simulated metal artifacts (pixel
  sizes 0.56-0.87 mm): FP bit-identical on all 16, BP difference at the original's noise level on all 16. Through the whole
  workflow (`inference` mode, one trial) the image before enhancement differed from the original by at most 0.005 HU (the original's
  two runs differ from each other by 0.004 HU) and after enhancement by at most 3.1 HU (the original's two runs differ by 4.4 HU).
  We did not test other artifact types or the head.
- **Speed** (shared, busy machine with one H200; read the ratios, not the seconds): one slice through the whole workflow including
  IR took about 13.9 s with the original `recon`, 8.2 s with `recon_fast` (12 threads) and 0.23 s with `gpuproj`, about 60x. What is
  left is the numpy filtering, the three networks and the per-call host-device copies. Model loading (about 25 s) is paid once per process.

**`utils/ir_modes.py` (best-trial metal image).** The threshold search never reads the IR output `img_m`, and the workflow keeps
it only for the last trial. To re-insert the metal into the final image you need the *best* trial's `img_m`:

- `RecordIR` (default) leaves the graph exactly as published and records every trial's `img_m`.
- `DeferredIR` skips IR on every trial and runs it once, for the best trial, from a captured `sino_m`. Nodes, their order,
  the search and all decisions are unchanged; only when IR runs. On two slices (8 trials each) the best trial, threshold,
  `gc` trace, `img_b` and `img_m` matched `RecordIR` within the original's own run-to-run noise, and a slice took 283-320 s
  instead of 831-964 s (CPU run, 4 threads, same contended machine). One slice failed a strict single-pair noise rule by up
  to 25% on five arrays (the same rule also flagged `img_b`, which `DeferredIR` cannot touch), so treat the equivalence as
  verified on few slices and re-check on your data.

The repository's `IterativeReconstruction` uses a fixed field of view (400 mm body, 220.16 mm head, the AAPM constants). If your
pixel size differs, set `utils.ir_modes.SLICE_FOV["mm"] = pixel_size * 512` per case and call `install_slice_fov(UG)`; otherwise
`img_m` comes out at the wrong scale (we saw median body-minus-metal RMSE of the re-inserted image rise from about 170 HU to about 970 HU
as the pixel size moved 0.1-0.2 mm away from 0.78125 mm).

## Notes and caveats

- **Working directory:** run scripts from `code/` (some scripts `chdir` there).
- **C extension required:** without a built, importable `recon` module the
  projection / reconstruction steps will fail.
- **Reproducibility:** seed is fixed to 42 across training scripts.
- **Inference time:** the L1 iterative reconstruction runs on CPU and dominates
  runtime (~12.7 s/slice for body, ~8.5 s/slice for head); MAR is intended as an
  offline post-processing step.
- **Training vs. test metrics:** validation-time SSIM/loss differ from the final
  test evaluation, which uses `ImageQualityEvaluator` on the test split.

## Data and code availability

- Paper: [Phys. Med. Biol., DOI 10.1088/1361-6560/ae7ec4](https://iopscience.iop.org/article/10.1088/1361-6560/ae7ec4)
- Dataset: AAPM CT-MAR Grand Challenge (generated with a hybrid simulation
  framework using the XCIST toolkit and publicly available clinical images).
- Code: https://github.com/KimYeongHyeon/LangGraphMAR

## Citation

If you find this work useful, please cite:

```bibtex
@article{Kim2026LangGraphMAR,
  title   = {Adaptive Thresholding for CT Metal Artifact Reduction via LangGraph},
  author  = {Kim, Yeonghyeon and Kim, Kyungsang and Lee, Dongheon},
  journal = {Physics in Medicine \& Biology},
  year    = {2026},
  doi     = {10.1088/1361-6560/ae7ec4},
  url     = {https://iopscience.iop.org/article/10.1088/1361-6560/ae7ec4}
}
```

## License

Released under the MIT License. See [LICENSE](LICENSE).

## Acknowledgements

This work was supported by IITP (NO.RS-2021-II211343, AI Graduate School
Program, Seoul National University), the Korea Health Technology R&D Project
(KHIDI, RS-2025-02307233), and the "Advanced GPU Utilization Support Program"
(MSIT, Republic of Korea).

## History

In reverse chronological order.

- 2026-10-08: added the opt-in GPU projector (`utils/gpuproj.py`, `scripts/verify_gpuproj.py`): FP bit-identical to `recon`, BP at the original's run-to-run noise level (branch `feat/fast-projection`), by Yeonghyeon Kim
- 2026-10-08: added the opt-in fast projector `recon_fast` (`utils/fastproj.py`, `scripts/verify_fastproj.py`) and the
  `RecordIR` / `DeferredIR` modes with per-case FOV (`utils/ir_modes.py`); the published code paths are untouched (branch `feat/fast-projection`), by Yeonghyeon Kim
- 2026-07-12: fixed the ROI metrics to match the paper's Table 2 definitions (`091e0b6`), by Yeonghyeon Kim
- 2026-07-11: added the reconstructed annular ROI evaluator (`163b2d2`), by Yeonghyeon Kim
- 2026-06-23: initial public release of LangGraph-MAR (`79fe6e0`), by Yeonghyeon Kim
