# Data

The raw CT data used by LangGraph-MAR is **not** included in this repository.
This directory only documents the expected layout so you can plug in your own
copy of the dataset.

## Dataset

We use the **AAPM CT-MAR Grand Challenge** dataset, a clinically representative
benchmark for CT metal artifact reduction. It contains paired sinograms and
images, with and without metal artifacts, produced by a hybrid simulation
framework (clinical images from NIH DeepLesion and UCLH Stroke EIT collections,
with metal artifacts simulated via the XCIST toolkit).

- 1,626 head + 12,374 body paired cases.
- Obtain the dataset from the AAPM CT-MAR Grand Challenge (see the paper's data
  availability statement). It is not redistributed here.

## Expected layout

Place the data so that the repository looks like:

```
data/
└── 01_raw/
    ├── body/
    │   ├── Baseline/   # metal-artifact images + sinograms (network input)
    │   ├── Target/     # metal-free ground-truth images + sinograms
    │   ├── LI/         # linear-interpolation prior (used by some baselines)
    │   └── Mask/       # metal-only masks + metal info
    └── head/
        ├── Baseline/
        ├── Target/
        ├── LI/
        └── Mask/
```

Then create a `dataset` symlink at the repository root (several scripts expect
`dataset/` to resolve to the raw data):

```bash
ln -s data/01_raw dataset
```

## File naming and format

Files are raw little-endian `float32` arrays named
`training_{anatomy}_{prefix}{index}.raw`, for example
`training_body_metalart_sino123.raw` or `training_head_nometal_img45.raw`.

| Folder | Prefixes | Content |
|--------|----------|---------|
| `Baseline` | `metalart_img`, `metalart_sino` | image / sinogram with metal artifacts |
| `Target`   | `nometal_img`, `nometal_sino`   | metal-free ground truth |
| `Mask`     | `metalonlymask_img`, `metalonlymask_sino`, `metalinfo` (`.json`) | metal-only masks and metadata |
| `LI`       | linear-interpolation prior | used by sinogram-domain baselines |

Array shapes:

- Image: `512 x 512` (float32)
- Sinogram: `1000 x 900` = views x detector channels (float32)

See `code/utils/dataset.py` (`load_data_list`, `IndexManager`) for the exact
loading logic, and `code/utils/ct.py` for the scanner geometry.

## Splits

Train/val/test split indices are provided under `code/index/` and are shared
across all models:

- Body: 7424 / 2475 / 2475
- Head: 975 / 325 / 326
