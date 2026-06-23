#!/usr/bin/env python
"""
NMAR test set evaluation.

Uses the SAME metric computation as run_threshold_sweep.py and
recalculate_metrics.py for fair comparison with all other methods.

Metrics: SSIM (raw HU, metal masked), RMSE (raw HU, metal masked)

Usage:
    cd <repo_root>/code
    python ../baselines/NMAR/eval_nmar.py --anatomy body
"""
import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

CODE_DIR = Path(__file__).parent.parent.parent / 'code'
sys.path.insert(0, str(CODE_DIR))
os.chdir(CODE_DIR)

from phasepack import phasecong
from utils.dataset import IndexManager, load_data_list
from utils.projection import filtering, bp
from utils.ct import get_FOV, transform_image_unit_mm_to_HU

sys.path.insert(0, str(Path(__file__).parent))
from nmar import nmar, setup_ct_params


# ---------- Same metric functions as run_threshold_sweep.py ----------

def calculate_ssim(img1, img2):
    """Calculate SSIM (same as sweep script)."""
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]

    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean()


def calculate_rmse(image1, image2):
    """Calculate RMSE (same as paper script)."""
    mse = np.mean((image1 - image2) ** 2)
    return np.sqrt(mse)


def normalize_image(image, vmin, vmax):
    """Normalize and scale an image to uint8 range (same as paper script)."""
    tmp_image = np.clip(image, vmin, vmax)
    tmp_image = (tmp_image - vmin) / (vmax - vmin) * 255
    return tmp_image.astype(np.uint8)


def calculate_fsim(gray1, gray2):
    """Calculate FSIM (same as paper script)."""
    pc1_data = phasecong(gray1)
    pc2_data = phasecong(gray2)
    pc1 = np.array(pc1_data[0])
    pc2 = np.array(pc2_data[0])

    gm1 = cv2.Sobel(gray1, cv2.CV_64F, 1, 1, ksize=3)
    gm2 = cv2.Sobel(gray2, cv2.CV_64F, 1, 1, ksize=3)
    gm1 = np.sqrt(gm1**2 + gm1**2)
    gm2 = np.sqrt(gm2**2 + gm2**2)

    T1, T2 = 0.85, 160
    pc_sim = 2 * pc1 * pc2 / (pc1**2 + pc2**2 + T1)
    gm_sim = 2 * gm1 * gm2 / (gm1**2 + gm2**2 + T2)
    pc_sim = np.where((pc1**2 + pc2**2 + T1) > 0, pc_sim, 0)
    gm_sim = np.where((gm1**2 + gm2**2 + T2) > 0, gm_sim, 0)

    return np.mean(pc_sim * gm_sim)


def compute_metrics(gt_image, pred_image, gt_metal_mask, anatomy):
    """Compute SSIM, FSIM, RMSE (same as paper script calculate_metrics_all.py)."""
    vmin, vmax = -150, 400

    pos_mask = np.where(gt_metal_mask > 0.5)
    gt_masked = gt_image.copy()
    pred_masked = pred_image.copy()
    gt_masked[pos_mask] = 0
    pred_masked[pos_mask] = 0

    ssim_score = calculate_ssim(gt_masked, pred_masked)
    rmse_score = calculate_rmse(gt_masked, pred_masked)

    gt_norm = normalize_image(gt_masked, vmin, vmax)
    pred_norm = normalize_image(pred_masked, vmin, vmax)
    fsim_score = calculate_fsim(gt_norm, pred_norm) * 100  # same as paper script

    return {'ssim': ssim_score, 'fsim': fsim_score, 'rmse': rmse_score}


# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--anatomy', type=str, required=True, choices=['body', 'head'])
    parser.add_argument('--output_dir', type=str, default=None)
    args = parser.parse_args()

    path_dict = load_data_list(Path('dataset'))
    _, _, test_indices = IndexManager('index', args.anatomy).load()

    ma_sino_paths = path_dict[args.anatomy]['metalart_sino_path_list'][test_indices]
    gt_sino_paths = path_dict[args.anatomy]['nometal_sino_path_list'][test_indices]
    mask_img_paths = path_dict[args.anatomy]['metalonlymask_img_path_list'][test_indices]

    ct_param = setup_ct_params(args.anatomy)

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    print(f"NMAR evaluation: {args.anatomy}, {len(ma_sino_paths)} test samples")

    all_ssim, all_fsim, all_rmse = [], [], []
    times = []

    for i in tqdm(range(len(ma_sino_paths)), desc="NMAR"):
        sino = np.fromfile(str(ma_sino_paths[i]), dtype=np.float32).reshape(1000, 900)
        mask_img = np.fromfile(str(mask_img_paths[i]), dtype=np.float32).reshape(512, 512)
        mask_bin = (mask_img > 0.5).astype(np.float32)

        # GT: reconstruct from GT sinogram (same as sweep)
        gt_sino = np.fromfile(str(gt_sino_paths[i]), dtype=np.float32).reshape(1000, 900)
        ct_fbp = ct_param.copy()
        ct_fbp['filter'] = 'ram-lak'
        gt_mm = np.maximum(bp(filtering(gt_sino, ct_fbp), ct_fbp).astype(np.float32), 0)
        gt_HU = transform_image_unit_mm_to_HU(gt_mm)

        # Run NMAR
        t0 = time.perf_counter()
        nmar_mm = nmar(sino, mask_bin, args.anatomy, ct_param)
        elapsed = (time.perf_counter() - t0) * 1000
        times.append(elapsed)

        nmar_HU = transform_image_unit_mm_to_HU(nmar_mm)

        # Metrics (same as paper script)
        metrics = compute_metrics(gt_HU, nmar_HU, mask_bin, args.anatomy)
        all_ssim.append(metrics['ssim'])
        all_fsim.append(metrics['fsim'])
        all_rmse.append(metrics['rmse'])

        if args.output_dir:
            nmar_HU.astype(np.float32).tofile(
                os.path.join(args.output_dir, f"nmar_{i:05d}.raw"))

    all_ssim = np.array(all_ssim)
    all_fsim = np.array(all_fsim)
    all_rmse = np.array(all_rmse)
    times = np.array(times)

    print(f"\n{'='*60}")
    print(f"NMAR Results ({args.anatomy}, n={len(ma_sino_paths)})")
    print(f"{'='*60}")
    print(f"  SSIM:  {all_ssim.mean():.4f} +/- {all_ssim.std():.4f}")
    print(f"  FSIM:  {all_fsim.mean():.4f} +/- {all_fsim.std():.4f}")
    print(f"  RMSE:  {all_rmse.mean():.4f} +/- {all_rmse.std():.4f}")
    print(f"  Time:  {times.mean():.1f} +/- {times.std():.1f} ms/sample")
    print(f"{'='*60}")

    df = pd.DataFrame({'ssim': all_ssim, 'fsim': all_fsim, 'rmse': all_rmse, 'time_ms': times})
    out_dir = CODE_DIR.parent / 'logs'
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = str(out_dir / f'nmar_metrics_{args.anatomy}.csv')
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")


if __name__ == '__main__':
    main()
