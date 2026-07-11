#!/usr/bin/env python3
"""Reconstructed Section 4.4/Table 2 per-metal annular ROI evaluator.

The original IQR mask code was deleted.  ``reconstructed_iqr_mask`` therefore
uses the documented reconstruction: full-image GT-HU Tukey 1.5-IQR rejection.
"""

from __future__ import annotations

import argparse
import csv
import math
import multiprocessing as mp
from collections import defaultdict
from pathlib import Path
from typing import Final

import cv2
import numpy as np
from phasepack import phasecong
from scipy import ndimage, stats
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

MASK_THRESHOLD: Final = 0.5
ANNULUS_RADIUS_PX: Final = 30.0
METHOD_FILES: Final = {
    "FBP": "FBP_image.npy",
    "U-Net": "unet_image.npy",
    "UFormer": "uformer_image.npy",
    "Restormer": "restormer_image.npy",
    "NAFNet": "nafnet_image.npy",
    "Ours": "proposed_image.npy",
}
METRICS: Final = ("ssim", "psnr", "fsim", "rmse")


def annular_roi_mask(metal_mask: np.ndarray) -> np.ndarray:
    """Return the union of all non-metal pixels within 30 px of metal."""
    metal = metal_mask > MASK_THRESHOLD
    distance = ndimage.distance_transform_edt(~metal)
    return (distance > 0.0) & (distance <= ANNULUS_RADIUS_PX)


def reconstructed_iqr_mask(gt_hu: np.ndarray) -> np.ndarray:
    """Reconstruction of the lost global full-image IQR outlier mask."""
    q1, q3 = np.quantile(gt_hu, (0.25, 0.75))
    iqr = q3 - q1
    return (gt_hu >= q1 - 1.5 * iqr) & (gt_hu <= q3 + 1.5 * iqr)


def body_fov_mask(shape: tuple[int, int]) -> np.ndarray:
    """Mirror ImageQualityEvaluator.get_ring_mask's 470-mm body FOV."""
    rows, cols = np.indices(shape)
    radius = 470.0 / (400.0 / 512.0) / 2.0
    return (rows - (shape[0] - 1) / 2.0) ** 2 + (
        cols - (shape[1] - 1) / 2.0
    ) ** 2 < radius**2


def _normalize01(image: np.ndarray) -> np.ndarray:
    return np.clip((image + 2000.0) / 8000.0, 0.0, 1.0)


def _normalize_uint8(image: np.ndarray) -> np.ndarray:
    return ((np.clip(image, -150.0, 400.0) + 150.0) / 550.0 * 255.0).astype(
        np.uint8
    )


def _fsim_map(image1: np.ndarray, image2: np.ndarray) -> np.ndarray:
    pc1 = np.asarray(phasecong(image1)[0])
    pc2 = np.asarray(phasecong(image2)[0])
    gm1 = cv2.Sobel(image1, cv2.CV_64F, 1, 1, ksize=3)
    gm2 = cv2.Sobel(image2, cv2.CV_64F, 1, 1, ksize=3)
    gm1 = np.sqrt(gm1**2 + gm1**2)
    gm2 = np.sqrt(gm2**2 + gm2**2)
    pc_sim = 2.0 * pc1 * pc2 / (pc1**2 + pc2**2 + 0.85)
    gm_sim = 2.0 * gm1 * gm2 / (gm1**2 + gm2**2 + 160.0)
    return pc_sim * gm_sim


def annular_roi_metrics(
    gt: np.ndarray, prediction: np.ndarray, metal_mask: np.ndarray, anatomy: str
) -> dict[str, float]:
    """Compute the verified Table 2 metric definitions for one sample."""
    metal = metal_mask > MASK_THRESHOLD
    gt_hu = gt.astype(np.float64, copy=True) + 1000.0
    pred_hu = prediction.astype(np.float64, copy=True) + 1000.0
    iqr_mask = reconstructed_iqr_mask(gt_hu)
    gt_hu[metal] = 0.0
    pred_hu[metal] = 0.0
    roi = annular_roi_mask(metal_mask) & iqr_mask & ~metal
    if anatomy.lower() == "body":
        roi &= body_fov_mask(gt_hu.shape)
    ssim_roi = roi[5:-5, 5:-5]
    if not roi.any() or not ssim_roi.any():
        return {metric: math.nan for metric in METRICS}
    gt01, pred01 = _normalize01(gt_hu), _normalize01(pred_hu)
    _, ssim_map = structural_similarity(
        gt01,
        pred01,
        gaussian_weights=True,
        win_size=11,
        data_range=1.0,
        full=True,
    )
    fsim_map = _fsim_map(_normalize_uint8(gt_hu), _normalize_uint8(pred_hu))
    return {
        "ssim": float(ssim_map[5:-5, 5:-5][ssim_roi].mean()),
        "psnr": float(
            peak_signal_noise_ratio(gt01[roi], pred01[roi], data_range=1.0)
        ),
        "fsim": float(fsim_map[roi].mean() * 100.0),
        "rmse": float(np.sqrt(np.mean((gt_hu[roi] - pred_hu[roi]) ** 2))),
    }


def _rows_for_folder(folder: Path) -> list[dict[str, str | float]]:
    rows: list[dict[str, str | float]] = []
    try:
        anatomy, _ = folder.name.rsplit("_", 1)
    except ValueError:
        print(f"skip malformed sample folder: {folder}")
        return rows
    mask_path = folder / "gt_metal_mask.npy"
    if not mask_path.exists():
        print(f"skip missing metal mask: {folder}")
        return rows
    gt, metal_mask = np.squeeze(np.load(folder / "gt_image.npy")), np.squeeze(
        np.load(mask_path)
    )
    for method, filename in METHOD_FILES.items():
        prediction_path = folder / filename
        if not prediction_path.exists():
            continue
        scores = annular_roi_metrics(
            gt, np.squeeze(np.load(prediction_path)), metal_mask, anatomy
        )
        rows.append(
            {"sample": folder.name, "anatomy": anatomy, "method": method}
            | scores
        )
    return rows


def _iter_rows(data_root: Path) -> list[dict[str, str | float]]:
    folders = sorted(path.parent for path in data_root.rglob("gt_image.npy"))
    # ponytail: one child/sample bounds PhasePack FFT-buffer growth; reuse when fixed.
    with mp.Pool(processes=1, maxtasksperchild=1) as pool:
        return [
            row
            for sample_rows in pool.imap(_rows_for_folder, folders)
            for row in sample_rows
        ]


def _write_rows(path: Path, rows: list[dict[str, str | float]]) -> None:
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=("sample", "anatomy", "method", *METRICS)
        )
        writer.writeheader()
        writer.writerows(rows)


def _print_table(rows: list[dict[str, str | float]]) -> None:
    grouped: defaultdict[tuple[str, str], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        key = (str(row["anatomy"]), str(row["method"]))
        for metric in METRICS:
            grouped[key][metric].append(float(row[metric]))
    print("anatomy method       SSIM          PSNR          FSIM          RMSE")
    for (anatomy, method), values in sorted(grouped.items()):
        cells = [
            f"{np.mean(values[metric]):.2f}±{np.std(values[metric], ddof=1):.2f}"
            for metric in METRICS
        ]
        print(f"{anatomy:<7} {method:<12} " + " ".join(f"{cell:<13}" for cell in cells))


def _paired_significance(rows: list[dict[str, str | float]], output: Path) -> None:
    by_key: defaultdict[tuple[str, str, str], dict[str, float]] = defaultdict(dict)
    for row in rows:
        for metric in METRICS:
            by_key[(str(row["anatomy"]), str(row["sample"]), metric)][
                str(row["method"])
            ] = float(row[metric])
    results: list[dict[str, str | float | int]] = []
    for anatomy in sorted({key[0] for key in by_key}):
        for metric in METRICS:
            samples = [
                values
                for (part, _, name), values in by_key.items()
                if part == anatomy and name == metric
            ]
            for method in sorted({name for values in samples for name in values} - {"Ours"}):
                pairs = [
                    (values["Ours"], values[method])
                    for values in samples
                    if "Ours" in values and method in values
                ]
                if len(pairs) < 2:
                    continue
                ours, other = np.asarray(pairs, dtype=float).T
                t_pvalue = float(stats.ttest_rel(ours, other).pvalue)
                try:
                    wilcoxon_pvalue = float(stats.wilcoxon(ours, other).pvalue)
                except ValueError:
                    wilcoxon_pvalue = math.nan
                results.append(
                    {
                        "anatomy": anatomy,
                        "metric": metric,
                        "comparison": f"Ours vs {method}",
                        "n": len(pairs),
                        "paired_t_pvalue": t_pvalue,
                        "wilcoxon_pvalue": wilcoxon_pvalue,
                    }
                )
    with output.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=(
                "anatomy",
                "metric",
                "comparison",
                "n",
                "paired_t_pvalue",
                "wilcoxon_pvalue",
            ),
        )
        writer.writeheader()
        writer.writerows(results)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root", type=Path, default=Path(__file__).parent / "data"
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path(__file__).parent / "roi_annular_per_sample.csv",
    )
    parser.add_argument(
        "--significance-csv",
        type=Path,
        default=Path(__file__).parent / "roi_annular_significance.csv",
    )
    args = parser.parse_args()
    rows = _iter_rows(args.data_root)
    if not rows:
        raise SystemExit(f"no usable samples under {args.data_root}")
    _write_rows(args.output_csv, rows)
    _paired_significance(rows, args.significance_csv)
    _print_table(rows)
    print(f"wrote {args.output_csv} and {args.significance_csv}")


if __name__ == "__main__":
    main()
