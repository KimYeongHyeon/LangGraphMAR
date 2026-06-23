# MDT — Metal Deletion Technique

Sinogram-domain traditional MAR baseline for CTMAR paper.

## Algorithm

Reference: Boas FE & Fleischmann D.
*"Evaluation of Two Iterative Techniques for Reducing Metal Artifacts in CT"*,
Radiology 259(3), 2011.

MDT performs 2+ FBP passes with sinogram-domain metal-trace infilling:

1. Forward-project the binary metal mask to obtain the **metal trace** in sinogram.
2. **Pass 1 (LI)**: linearly interpolate the metal trace, reconstruct via FBP.
3. **Pass 2+ (refinement)**: replace metal region in the image with water,
   re-project to get a model-based sinogram, fill the metal trace from that,
   reconstruct via FBP. Repeat.

Distinction vs NMAR: NMAR normalises the sinogram by a KMeans-segmented prior
image (air/water/bone) before interpolation. MDT uses a uniform water prior and
works directly on the raw sinogram values.

## Files

| File | Purpose |
|------|---------|
| `mdt.py` | Core MDT algorithm. Reuses `proj_interp`/`setup_ct_params` from `../NMAR/nmar.py` and FBP utilities from `code/utils/projection.py`. |
| `eval_mdt.py` | Test-set evaluation with paper SSIM/FSIM/RMSE metrics (copied from `eval_nmar.py` for identical eval). |

## Usage

Run from `code/` so that `dataset` and `index` symlinks resolve:

```bash
cd code

# Body test set (2475 samples)
python ../baselines/MDT/eval_mdt.py --anatomy body --iterations 2

# Head test set (326 samples)
python ../baselines/MDT/eval_mdt.py --anatomy head --iterations 2

# Quick spot-check (first 10 samples)
python ../baselines/MDT/eval_mdt.py --anatomy body --limit 10
```

Output: `logs/mdt_metrics_{anatomy}.csv` with per-sample SSIM/FSIM/RMSE/time_ms.

`--iterations 1` reduces to plain LI (no refinement).
`--iterations 3+` adds more refinement passes.

## Environment

Uses the system Python (`python`) or NMAR's venv.
Requires `odl`, `numpy`, `cv2`, `phasepack`, `skimage`, `pandas`.
