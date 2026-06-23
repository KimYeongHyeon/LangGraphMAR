#!/usr/bin/env python
# coding: utf-8
"""
Threshold Sweep Script for Proposed MAR Method

모든 threshold에 대해 proposed method를 실행하고 metric을 계산합니다.

Usage:
    python scripts/run_threshold_sweep.py \
        --anatomy body \
        --min_threshold 0.03 \
        --max_threshold 0.40 \
        --step 0.01 \
        --output_dir results_threshold_sweep
"""

import argparse
import importlib
import os
import random
import sys
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import pytz
import torch
import cv2

from tqdm import tqdm

# Add parent directory to path for imports
CODE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(CODE_DIR))
os.chdir(CODE_DIR)  # Change to code directory for relative paths

import recon
importlib.reload(recon)

from gecatsim.pyfiles.CommonTools import rawread, rawwrite
from phasepack import phasecong

from utils.dataset import IndexManager, load_data_list
from utils.projection import filtering, bp
from utils.ct import get_FOV, transform_image_unit_mm_to_HU
from utils.utils import load_models
from utils.algorithm import inference_enhancement
from utils.graph import get_workflow_graph


# =============================================================================
# Metric Calculation Functions
# =============================================================================
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
    # Set vmin/vmax based on anatomy
    if anatomy == 'body':
        vmin, vmax = -150, 400
    else:  # head
        vmin, vmax = -150, 400
    
    # Mask metal regions
    pos_mask = np.where(gt_metal_mask == 1)
    gt_masked = gt_image.copy()
    pred_masked = pred_image.copy()
    gt_masked[pos_mask] = 0
    pred_masked[pos_mask] = 0
    
    # SSIM (on raw HU values)
    ssim_score = calculate_ssim(gt_masked, pred_masked)
    
    # RMSE (on raw HU values)
    rmse_score = calculate_rmse(gt_masked, pred_masked)
    
    # FSIM (on normalized images)
    gt_norm = normalize_image(gt_masked, vmin, vmax)
    pred_norm = normalize_image(pred_masked, vmin, vmax)
    fsim_score = calculate_fsim(gt_norm, pred_norm)
    
    return {
        'ssim': ssim_score,
        'fsim': fsim_score * 100,  # Scale to percentage
        'rmse': rmse_score
    }


# =============================================================================
# Utility Functions
# =============================================================================
def seed_everything(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    pl.seed_everything(seed)


def setup_ct_params(anatomy):
    """Setup CT reconstruction parameters."""
    param = OrderedDict({
        'nx': 512,
        'ny': 512,
        'DSD': 950.0,
        'DSO': 550.0,
        'nu': 900,
        'du': 1.0,
        'nview': 1000,
        'filter': 'ram-lak'
    })
    
    param['deg'] = np.linspace(0, 360, param['nview'], endpoint=False)
    param['fan_angle'] = param['du'] / param['DSD'] * 180 / np.pi * param['nu']
    param['da'] = param['fan_angle'] / param['nu'] / 180 * np.pi
    param['off_a'] = 1.25
    
    FOV = get_FOV(anatomy=anatomy)
    param['dx'] = FOV / param['nx']
    param['dy'] = FOV / param['ny']
    
    return param


# =============================================================================
# Main Processing Functions
# =============================================================================
def process_single_sample(
    data_path, 
    threshold, 
    output_dir, 
    param, 
    anatomy,
    inpainting_model,
    classifier_model,
    enhancement_model,
    app,
    compute_metrics_flag=True
):
    """Process a single sample with the given threshold."""
    
    # Extract paths
    sample_id = data_path.split('/')[-1].split('_')[-2]
    original_gt_path = (data_path
        .replace('Baseline', 'Target')
        .replace('metalart', 'nometal')
        .replace('sino', 'img')
        .replace('900x1000', '512x512x1')
    )
    
    result_dir = os.path.join(output_dir, anatomy, f'{threshold:.2f}')
    os.makedirs(result_dir, exist_ok=True)
    
    image_m_path = os.path.join(result_dir, f'{sample_id}_image_m.raw')
    image_b_path = os.path.join(result_dir, f'{sample_id}_image_b.raw')
    
    # Check if already processed
    if os.path.exists(image_b_path) and os.path.exists(image_m_path):
        # Load existing results for metric computation
        if compute_metrics_flag:
            pred_image = rawread(image_b_path, [512, 512], 'float')
            gt_image = rawread(
                original_gt_path.replace('Target', 'Target'),
                [512, 512, 1], 'float'
            ).squeeze()
            gt_metal_mask = rawread(
                original_gt_path.replace('Target', 'Mask').replace('nometal', 'metalonlymask'),
                [512, 512, 1], 'float'
            ).squeeze()
            
            # Compute metrics
            gt_sino = rawread(
                original_gt_path.replace('Target', 'Target').replace('img', 'sino')
                              .replace('512x512x1', '900x1000'),
                [1000, 900], 'float'
            )
            sino_filt = filtering(gt_sino, param)
            gt_recon = bp(sino_filt, param).astype('float32')
            gt_image_HU = transform_image_unit_mm_to_HU(np.maximum(gt_recon, 0))
            
            metrics = compute_metrics(gt_image_HU, pred_image, gt_metal_mask, anatomy)
            return {
                'threshold': threshold,
                'sample_id': sample_id,
                'anatomy': anatomy,
                **metrics,
                'skipped': True
            }
        return None
    
    # Load data
    ori_metalart_sinogram = rawread(data_path, [1000, 900], 'float')
    gt_sino = rawread(
        original_gt_path.replace('Target', 'Target').replace('img', 'sino')
                      .replace('512x512x1', '900x1000'),
        [1000, 900], 'float'
    )
    gt_metal_mask = rawread(
        original_gt_path.replace('Target', 'Mask').replace('nometal', 'metalonlymask'),
        [512, 512, 1], 'float'
    ).squeeze()
    
    # Reconstruct GT image
    sino_filt = filtering(gt_sino, param)
    gt_image = bp(sino_filt, param).astype('float32')
    gt_image_mm = np.maximum(gt_image, 0)
    gt_image_HU = transform_image_unit_mm_to_HU(gt_image_mm)
    
    # Set reconstruction parameters
    reconstruction_params = OrderedDict({
        'image_mask_threshold': threshold,
        'kernel_size': (13, 1),
        'sigmaX': 0.8,
        'ir_num_iterations': 10,
        'soft_thresholding': 0.003,
        'metal_striking_threshold': 1.0,
        'current_IR_iteration': 0,
        'num_whole_iterations': 1,
    })
    
    # Prepare inputs
    inputs = {
        "original_sinogram": ori_metalart_sinogram,
        "reconstruction_params": reconstruction_params,
        "total_iterations": 10,
        "current_iteration": 0,
        "ct_param": param,
        "anatomy": anatomy,
        "inpainting_model": inpainting_model,
        "classifier_model": classifier_model,
    }
    
    # Run reconstruction
    image_b_HU = None
    image_m_HU = None
    
    for output in app.stream(inputs):
        if 'reconstruction_pipeline' in output:
            image_b_mm = output['reconstruction_pipeline']['img_b']
            image_m_mm = output['reconstruction_pipeline']['img_m']
            
            image_b_HU = transform_image_unit_mm_to_HU(image_b_mm).astype('float32')
            image_m_HU = transform_image_unit_mm_to_HU(image_m_mm).astype('float32')
            
            # Apply enhancement
            image_b_HU = inference_enhancement(image_b_HU, enhancement_model).squeeze().astype('float32')
            
            # Save results
            rawwrite(image_b_path, image_b_HU)
            rawwrite(image_m_path, image_m_HU)
    
    if image_b_HU is None:
        print(f"Warning: No output for sample {sample_id} at threshold {threshold}")
        return None
    
    # Compute metrics
    if compute_metrics_flag:
        metrics = compute_metrics(gt_image_HU, image_b_HU, gt_metal_mask, anatomy)
        return {
            'threshold': threshold,
            'sample_id': sample_id,
            'anatomy': anatomy,
            **metrics,
            'skipped': False
        }
    
    return None


def run_threshold_sweep(args):
    """Main function to run threshold sweep."""
    
    print("=" * 70)
    print(f"Threshold Sweep: {args.anatomy}")
    print(f"Threshold range: {args.min_threshold} ~ {args.max_threshold} (step: {args.step})")
    print(f"Output directory: {args.output_dir}")
    print("=" * 70)
    
    # Setup
    seed_everything(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Load models
    print("Loading models...")
    model_dict = load_models(load_enhancement=True)
    inpainting_models = model_dict['inpainting']
    classifier_model = model_dict['gc']
    enhancement_models = model_dict['enhancement']
    
    # Get workflow graph
    app = get_workflow_graph()
    
    # Setup CT parameters
    param = setup_ct_params(args.anatomy)
    
    # Load data paths
    dataset_path = Path('../dataset')
    path_dict = load_data_list(dataset_path, import_new_gt=True)[args.anatomy]
    train_indices, val_indices, test_indices = IndexManager('index', args.anatomy).load()
    
    # Get test data paths
    test_path_list = path_dict['metalart_sino_path_list'][test_indices]
    
    if args.max_samples:
        test_path_list = test_path_list[:args.max_samples]
    
    print(f"Total test samples: {len(test_path_list)}")
    
    # Generate threshold list
    threshold_list = np.arange(args.min_threshold, args.max_threshold + args.step/2, args.step)
    if args.reverse:
        threshold_list = threshold_list[::-1]
        print(f"Running in REVERSE order")
    print(f"Thresholds to process: {len(threshold_list)}")
    
    # Output directory
    os.makedirs(os.path.join(args.output_dir, args.anatomy), exist_ok=True)
    
    # CSV for metrics
    csv_path = os.path.join(args.output_dir, args.anatomy, 'metrics_by_threshold.csv')
    
    # Check existing progress
    if os.path.exists(csv_path):
        existing_df = pd.read_csv(csv_path)
        processed = set(zip(existing_df['threshold'], existing_df['sample_id']))
        print(f"Resuming from {len(processed)} processed (threshold, sample) pairs")
    else:
        processed = set()
    
    # Process each threshold
    all_results = []
    
    for threshold in tqdm(threshold_list, desc="Thresholds"):
        threshold = round(threshold, 2)  # Avoid floating point issues
        
        progress_bar = tqdm(test_path_list, desc=f"Thr={threshold:.2f}", leave=False)
        
        for data_path in progress_bar:
            sample_id = data_path.split('/')[-1].split('_')[-2]
            progress_bar.set_postfix(sample=sample_id)
            
            # Skip if already processed
            if (threshold, sample_id) in processed:
                continue
            
            try:
                result = process_single_sample(
                    data_path=data_path,
                    threshold=threshold,
                    output_dir=args.output_dir,
                    param=param,
                    anatomy=args.anatomy,
                    inpainting_model=inpainting_models[args.anatomy],
                    classifier_model=classifier_model,
                    enhancement_model=enhancement_models[args.anatomy],
                    app=app,
                    compute_metrics_flag=True
                )
                
                if result:
                    all_results.append(result)
                    
                    # Save incrementally
                    if len(all_results) % 10 == 0:
                        df_new = pd.DataFrame(all_results)
                        header = not os.path.exists(csv_path)
                        df_new.to_csv(csv_path, mode='a', index=False, header=header)
                        all_results = []
                        
            except Exception as e:
                print(f"Error processing {sample_id} at threshold {threshold}: {e}")
                continue
    
    # Save remaining results
    if all_results:
        df_new = pd.DataFrame(all_results)
        header = not os.path.exists(csv_path)
        df_new.to_csv(csv_path, mode='a', index=False, header=header)
    
    # Generate summary statistics
    print("\n" + "=" * 70)
    print("Generating summary statistics...")
    print("=" * 70)
    
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        
        # Stats by threshold
        stats = df.groupby('threshold')[['ssim', 'fsim', 'rmse']].agg(['mean', 'std'])
        stats.columns = ['_'.join(col).strip() for col in stats.columns.values]
        stats = stats.reset_index()
        
        stats_path = os.path.join(args.output_dir, args.anatomy, 'threshold_stats.csv')
        stats.to_csv(stats_path, index=False)
        print(f"\nSaved threshold statistics to: {stats_path}")
        
        # Find optimal threshold
        best_ssim_thr = stats.loc[stats['ssim_mean'].idxmax(), 'threshold']
        best_rmse_thr = stats.loc[stats['rmse_mean'].idxmin(), 'threshold']
        
        print(f"\nOptimal thresholds for {args.anatomy}:")
        print(f"  Best SSIM: threshold = {best_ssim_thr:.2f}")
        print(f"  Best RMSE: threshold = {best_rmse_thr:.2f}")
    
    print("\nDone!")


def main():
    parser = argparse.ArgumentParser(description='Run threshold sweep for proposed MAR method')
    parser.add_argument('--anatomy', type=str, choices=['head', 'body'], required=True,
                       help='Anatomy type (head or body)')
    parser.add_argument('--min_threshold', type=float, default=0.03,
                       help='Minimum threshold value')
    parser.add_argument('--max_threshold', type=float, default=0.40,
                       help='Maximum threshold value')
    parser.add_argument('--step', type=float, default=0.01,
                       help='Threshold step size')
    parser.add_argument('--output_dir', type=str, default='results_threshold_sweep',
                       help='Output directory for results')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of samples to process (for testing)')
    parser.add_argument('--reverse', action='store_true',
                       help='Run sweep in reverse order')
    
    args = parser.parse_args()
    run_threshold_sweep(args)


if __name__ == '__main__':
    main()
