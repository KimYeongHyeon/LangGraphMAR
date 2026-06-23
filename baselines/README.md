# MAR Baseline Methods

CT Metal Artifact Reduction (MAR) baseline methods used for comparison in the
**LangGraph-MAR** paper. This release ships **code only** — no weights, no data.
Run the evaluation scripts from the `code/` directory so that the `dataset` and
`index` symlinks resolve (see the repository-root [`README.md`](../README.md)).

## Included baselines

| Method | Type | Domain | Entry point | Environment |
|--------|------|--------|-------------|-------------|
| **LMAR** | Traditional | Sinogram | `proj_interp()` in `NMAR/nmar.py` | system Python / NMAR venv |
| **NMAR** (Meyer et al., Med. Phys. 2010) | Traditional | Sinogram | `NMAR/eval_nmar.py` | system Python / NMAR venv |
| **MDT** (Boas & Fleischmann, Radiology 2011) | Traditional (iterative) | Sinogram | `MDT/eval_mdt.py` | system Python / NMAR venv |
| **InDuDoNet+** (Wang et al., Med. Image Anal. 2023) | Deep learning | Dual (image + sinogram) | `InDuDoNet/eval.py` | dedicated venv |

## Usage

```bash
cd code

# NMAR (and LMAR via proj_interp)
python ../baselines/NMAR/eval_nmar.py --anatomy body

# MDT (2 iterations by default; --iterations 1 == plain LI)
python ../baselines/MDT/eval_mdt.py --anatomy body --iterations 2

# InDuDoNet+ (uses its own virtual environment)
source ../baselines/InDuDoNet/activate.sh
python ../baselines/InDuDoNet/eval.py --anatomy body --model_path <your_checkpoint>
```

All evaluation scripts use the shared test-split indices in
[`code/index/`](../code/index/) and the evaluation logic in
[`code/utils/metric.py`](../code/utils/metric.py) for a fair comparison.

## Folder layout

```
baselines/
├── NMAR/         # LMAR + NMAR (traditional, sinogram-domain)
├── MDT/          # Metal Deletion Technique (iterative, sinogram-domain)
├── InDuDoNet/    # InDuDoNet+ (PyTorch, dual-domain; needs ODL + ASTRA, own venv)
└── README.md     # this file
```

## Notes

- **InDuDoNet/** keeps the upstream project's original `README.md` and network
  code for attribution; this paper adds `eval.py`, `dump_predictions*.py`, and
  `benchmark_inference_time.py` for evaluation against the LangGraph-MAR splits.
  It requires an older stack (ODL + ASTRA); use its dedicated virtual
  environment via `activate.sh`.
- **MDT** reuses `proj_interp` / `setup_ct_params` from `NMAR/nmar.py` and the
  FBP utilities from `code/utils/projection.py`. See `MDT/README.md` for details.
- Trained checkpoints and raw data are **not** distributed here.
