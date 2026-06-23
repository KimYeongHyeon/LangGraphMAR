# `code/` directory

Core reproduction code for **LangGraph-MAR**. See the repository-root
[`README.md`](../README.md) for installation, data preparation, and the full
usage guide. Run all scripts from this `code/` directory.

## Layout

| Path | Description |
|------|-------------|
| `utils/` | Core utilities. `ct.py` (scanner geometry, source of truth), `metric.py` (`ImageQualityEvaluator`, evaluation source of truth), `dataset.py` (`IndexManager`, data loading), `projection.py` (`import recon` FP/BP), `graph.py` (LangGraph workflow), `utils.py` (`load_models`). |
| `training/` | Training entry points: `train_inpainting.py`, `train_enhancement.py`, `train_gc.py`, `train_image_domain_mar.py`, `train_mar_*.py`. |
| `scripts/` | `run_threshold_sweep.py` (full pipeline + threshold sweep) and metric (re)computation scripts. See `scripts/README.md`. |
| `pl_modules/` | PyTorch Lightning modules. |
| `loss/` | Loss functions. (Model definitions live in `utils/models.py`.) |
| `configs/` | Hydra configuration files. |
| `paper_results/` | Metric aggregation and statistical-significance scripts. |
| `index/` | Train/val/test split indices (`*.pkl`), shared across all models. |
| `CT_recon_fanbeam_python_openmp/` | C extension (`recon.c` + `setup_linux.py`/`setup_mac.py`) for forward/back projection. Build with `python setup_linux.py build_ext --inplace` and make it importable as `recon`. |
| `AAPM_datachallenge/` | AAPM challenge geometry / simulation references. |
| `param.yaml` | Per-anatomy intensity normalization statistics. |

## Quick reference

```bash
# from code/
python training/train_inpainting.py --anatomy body
python scripts/run_threshold_sweep.py --anatomy body \
    --min_threshold 0.01 --max_threshold 0.40 --step 0.01
```

## Notes

- Training scripts add the `code/` directory to `sys.path` automatically, so
  `utils` (and any optional external model packages you clone under
  `code/models/external/`) import correctly from subfolders.
- A built, importable `recon` C extension is required for projection and
  reconstruction.
