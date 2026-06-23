#!/usr/bin/env python
"""
Metric Recalculation Script for Enhanced Threshold Sweep Results

기존 threshold sweep 결과 (_image_b.raw)에서 메트릭을 재계산합니다.
"""

import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phasepack import phasecong
from utils.dataset import IndexManager, load_data_list


def normalize_image(image, vmin, vmax):
    """Normalize and scale an image to uint8 range."""
    tmp_image = np.clip(image, vmin, vmax)
    tmp_image = (tmp_image - vmin) / (vmax - vmin) * 255
    return tmp_image.astype(np.uint8)


def calculate_fsim(gray1, gray2):
    """Calculate Feature Similarity Index (FSIM)."""
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
    
    sim = pc_sim * gm_sim
    return np.mean(sim)


def calculate_ssim(img1, img2):
    """Calculate Structural Similarity Index (SSIM)."""
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
    """Calculate Root Mean Square Error (RMSE)."""
    mse = np.mean((image1 - image2) ** 2)
    return np.sqrt(mse)





def compute_metrics(gt_image, pred_image, gt_metal_mask, anatomy):
    """Compute all metrics for a single image pair."""
    pos_mask = np.where(gt_metal_mask == 1)
    gt_masked = gt_image.copy()
    pred_masked = pred_image.copy()
    gt_masked[pos_mask] = 0
    pred_masked[pos_mask] = 0
    
    ssim_score = calculate_ssim(gt_masked, pred_masked)
    rmse_score = calculate_rmse(gt_masked, pred_masked)
    
    return {
        'ssim': ssim_score,
        'rmse': rmse_score
    }


def process_threshold(thr_dir, anatomy, path_dict, test_indices):
    """Process a single threshold directory."""
    from gecatsim.pyfiles.CommonTools import rawread
    
    threshold = os.path.basename(thr_dir)
    try:
        thr_val = float(threshold)
    except:
        return []
    
    results = []
    img_files = glob.glob(os.path.join(thr_dir, '*_image_b.raw'))
    
    metalart_sino_paths = path_dict['metalart_sino_path_list'][test_indices]
    
    for img_path in tqdm(img_files, desc=f"Thr={threshold}"):
        sample_id = os.path.basename(img_path).replace('_image_b.raw', '')
        num = sample_id.replace('sino', '')
        
        # Find corresponding GT
        gt_path = None
        for p in metalart_sino_paths:
            if f'sino{num}_' in str(p):
                gt_path = (str(p)
                    .replace('Baseline', 'Target')
                    .replace('metalart', 'nometal')
                    .replace('sino', 'img')
                    .replace('900x1000', '512x512x1')
                )
                break
        
        if gt_path is None or not os.path.exists(gt_path):
            continue
        
        # Load data
        pred_image = np.fromfile(img_path, dtype=np.float32).reshape(512, 512)
        
        mask_path = img_path.replace('_image_b.raw', '_image_m.raw')
        if os.path.exists(mask_path):
            metal_mask = np.fromfile(mask_path, dtype=np.float32).reshape(512, 512)
            metal_mask = (metal_mask > 0).astype(np.float32)
        else:
            continue
        
        # Load GT image directly (no sinogram reconstruction needed)
        gt_image_HU = rawread(gt_path, [512, 512, 1], 'float').squeeze()
        
        metrics = compute_metrics(gt_image_HU, pred_image, metal_mask, anatomy)
        results.append({
            'threshold': thr_val,
            'sample_id': sample_id,
            'anatomy': anatomy,
            **metrics
        })
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--anatomy', type=str, required=True, choices=['head', 'body'])
    parser.add_argument('--input_dir', type=str, default='results_threshold_sweep')
    parser.add_argument('--threshold', type=str, default=None, help='Single threshold to process (e.g., 0.10)')
    args = parser.parse_args()
    
    print(f"Calculating metrics for {args.anatomy}" + (f" threshold={args.threshold}" if args.threshold else ""))
    
    # Setup (repo-root `dataset` symlink -> data/01_raw)
    dataset_path = Path(__file__).resolve().parents[2] / 'dataset'
    path_dict = load_data_list(dataset_path, import_new_gt=True)[args.anatomy]
    _, _, test_indices = IndexManager('index', args.anatomy).load()
    
    # Get threshold directories
    if args.threshold:
        threshold_dirs = [os.path.join(args.input_dir, args.anatomy, args.threshold)]
    else:
        threshold_dirs = sorted(glob.glob(os.path.join(args.input_dir, args.anatomy, '*')))
        threshold_dirs = [d for d in threshold_dirs if os.path.isdir(d)]
    
    all_results = []
    
    for thr_dir in tqdm(threshold_dirs, desc="Thresholds"):
        if not os.path.isdir(thr_dir):
            continue
        results = process_threshold(thr_dir, args.anatomy, path_dict, test_indices)
        all_results.extend(results)
    
    # Save results
    df = pd.DataFrame(all_results)
    if args.threshold:
        csv_path = os.path.join(args.input_dir, args.anatomy, f'metrics_{args.threshold}.csv')
    else:
        csv_path = os.path.join(args.input_dir, args.anatomy, 'metrics_by_threshold.csv')
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")
    
    if not args.threshold and len(df) > 0:
        # Stats
        stats = df.groupby('threshold')[['ssim', 'rmse']].agg(['mean', 'std'])
        stats.columns = ['_'.join(col).strip() for col in stats.columns.values]
        stats = stats.reset_index()
        stats_path = os.path.join(args.input_dir, args.anatomy, 'threshold_stats.csv')
        stats.to_csv(stats_path, index=False)
        print(f"Saved: {stats_path}")
        
        # Best thresholds
        best_ssim_thr = stats.loc[stats['ssim_mean'].idxmax(), 'threshold']
        best_rmse_thr = stats.loc[stats['rmse_mean'].idxmin(), 'threshold']
        print(f"\nOptimal thresholds for {args.anatomy}:")
        print(f"  Best SSIM: {best_ssim_thr:.2f}")
        print(f"  Best RMSE: {best_rmse_thr:.2f}")


def merge_csv_files(input_dir, anatomy):
    """Merge individual threshold CSV files into metrics_by_threshold.csv and threshold_stats.csv"""
    csv_files = sorted(glob.glob(os.path.join(input_dir, anatomy, 'metrics_0.*.csv')))
    
    if not csv_files:
        print(f"No metrics_0.*.csv files found in {input_dir}/{anatomy}/")
        return
    
    print(f"Found {len(csv_files)} CSV files for {anatomy}")
    
    dfs = [pd.read_csv(f) for f in csv_files]
    df = pd.concat(dfs, ignore_index=True)
    print(f"Total records: {len(df)}")
    
    # Save merged metrics
    metrics_path = os.path.join(input_dir, anatomy, 'metrics_by_threshold.csv')
    df.to_csv(metrics_path, index=False)
    print(f"Saved: {metrics_path}")
    
    # Calculate and save stats
    stats = df.groupby('threshold')[['ssim', 'rmse']].agg(['mean', 'std'])
    stats.columns = ['_'.join(col).strip() for col in stats.columns.values]
    stats = stats.reset_index()
    stats_path = os.path.join(input_dir, anatomy, 'threshold_stats.csv')
    stats.to_csv(stats_path, index=False)
    print(f"Saved: {stats_path}")
    
    # Best thresholds
    best_ssim_thr = stats.loc[stats['ssim_mean'].idxmax(), 'threshold']
    best_rmse_thr = stats.loc[stats['rmse_mean'].idxmin(), 'threshold']
    print(f"\nOptimal thresholds for {anatomy}:")
    print(f"  Best SSIM: {best_ssim_thr:.2f} (SSIM={stats['ssim_mean'].max():.4f})")
    print(f"  Best RMSE: {best_rmse_thr:.2f} (RMSE={stats['rmse_mean'].min():.2f})")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--anatomy', type=str, choices=['head', 'body'])
    parser.add_argument('--input_dir', type=str, default='results_threshold_sweep')
    parser.add_argument('--threshold', type=str, default=None, help='Single threshold to process')
    parser.add_argument('--merge', action='store_true', help='Merge individual CSV files only')
    args = parser.parse_args()
    
    if args.merge:
        if args.anatomy:
            merge_csv_files(args.input_dir, args.anatomy)
        else:
            for anatomy in ['head', 'body']:
                print(f"\n{'='*50}")
                print(f"Processing {anatomy.upper()}")
                print('='*50)
                merge_csv_files(args.input_dir, anatomy)
    else:
        if not args.anatomy:
            print("Error: --anatomy is required when not using --merge")
            sys.exit(1)
        main()

