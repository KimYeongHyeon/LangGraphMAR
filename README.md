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
│   │                           # dataset.py, projection.py, graph.py, ...
│   ├── training/               # train_inpainting / train_enhancement / train_gc / train_*_mar
│   ├── scripts/                # run_threshold_sweep.py + metric scripts
│   ├── pl_modules/             # PyTorch Lightning modules
│   ├── loss/ configs/          # losses, Hydra configs (model defs in utils/models.py)
│   ├── paper_results/          # metric / significance scripts
│   ├── index/                  # train/val/test split indices (*.pkl)
│   ├── CT_recon_fanbeam_python_openmp/   # C extension (forward/back projection)
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
