#!/usr/bin/env python
"""
Enhancement 후처리 스크립트

기존 threshold sweep 결과(_image_b.raw)에 Enhancement 모델을 적용하여
enhanced 결과를 생성하고 메트릭을 재계산합니다.
"""

import os
import sys
import glob
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import cv2

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.models import ImageEnhancement
from utils.algorithm import inference_enhancement


def calculate_ssim(img1, img2):
    """Calculate SSIM"""
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


def calculate_rmse(img1, img2):
    """Calculate RMSE"""
    return np.sqrt(np.mean((img1 - img2) ** 2))


def load_enhancement_model(anatomy):
    """Load enhancement model for given anatomy using utils.utils.load_models"""
    from utils.utils import load_models
    
    model_dict = load_models(load_enhancement=True)
    model = model_dict['enhancement'][anatomy]
    model.eval()
    return model


def get_gt_and_mask(anatomy, sample_id, thr_dir):
    """
    Load GT image from original dataset and mask from sweep results
    GT: 원본 데이터셋에서 로드
    Mask: sweep 결과 폴더의 _image_m.raw 사용
    """
    from pathlib import Path
    from utils.dataset import load_data_list
    from utils.ct import rawread
    
    # Load mask from sweep results
    mask_path = os.path.join(thr_dir, f'{sample_id}_image_m.raw')
    if not os.path.exists(mask_path):
        return None, None
    
    mask = np.fromfile(mask_path, dtype=np.float32).reshape(512, 512)
    
    # Load GT from original dataset (repo-root `dataset` symlink -> data/01_raw)
    dataset_path = Path(__file__).resolve().parents[2] / 'dataset'
    path_dict = load_data_list(dataset_path, import_new_gt=True)[anatomy]
    
    # Find matching sample
    num = sample_id.replace('sino', '')
    gt_path = None
    for p in path_dict:
        if f'sino{num}_' in str(p) and 'Target' in str(p):
            gt_path = str(p)
            break
    
    if gt_path is None:
        return None, mask
    
    gt = rawread(gt_path, [512, 512, 1], 'float').squeeze()
    return gt, mask


def process_anatomy(anatomy, input_dir, output_dir):
    """Process all thresholds for given anatomy"""
    
    print(f"\n{'='*60}")
    print(f"Processing {anatomy.upper()}")
    print(f"{'='*60}")
    
    # Load model
    print("Loading enhancement model...")
    model = load_enhancement_model(anatomy)
    
    # Get all threshold folders
    threshold_dirs = sorted(glob.glob(os.path.join(input_dir, anatomy, '*')))
    threshold_dirs = [d for d in threshold_dirs if os.path.isdir(d)]
    
    all_results = []
    
    for thr_dir in tqdm(threshold_dirs, desc="Thresholds"):
        threshold = os.path.basename(thr_dir)
        
        # Skip non-numeric folders
        try:
            thr_val = float(threshold)
        except:
            continue
        
        # Create output directory
        out_thr_dir = os.path.join(output_dir, anatomy, threshold)
        os.makedirs(out_thr_dir, exist_ok=True)
        
        # Get all image files
        img_files = glob.glob(os.path.join(thr_dir, '*_image_b.raw'))
        
        for img_path in tqdm(img_files, desc=f"Thr={threshold}", leave=False):
            sample_id = os.path.basename(img_path).replace('_image_b.raw', '')
            
            # Load image
            img_b = np.fromfile(img_path, dtype=np.float32).reshape(512, 512)
            
            # Apply enhancement
            enhanced = inference_enhancement(img_b, model)
            enhanced = enhanced.squeeze()
            
            # Denormalize: model outputs normalized, convert back to HU
            enhanced = enhanced * 2000 - 1000
            
            # Save enhanced image
            out_path = os.path.join(out_thr_dir, f'{sample_id}_enhanced.raw')
            enhanced.astype(np.float32).tofile(out_path)
            
            # Load GT and mask for metric calculation
            gt, mask = get_gt_and_mask(anatomy, sample_id, thr_dir)
            
            if gt is not None:
                # Mask metal regions
                pos_mask = np.where(mask > 0)
                gt_masked = gt.copy()
                enhanced_masked = enhanced.copy()
                gt_masked[pos_mask] = 0
                enhanced_masked[pos_mask] = 0
                
                # Calculate metrics
                ssim_score = calculate_ssim(gt_masked, enhanced_masked)
                rmse_score = calculate_rmse(gt_masked, enhanced_masked)
                
                all_results.append({
                    'threshold': thr_val,
                    'sample_id': sample_id,
                    'anatomy': anatomy,
                    'ssim': ssim_score,
                    'rmse': rmse_score,
                    'skipped': False
                })
            else:
                all_results.append({
                    'threshold': thr_val,
                    'sample_id': sample_id,
                    'anatomy': anatomy,
                    'ssim': 0,
                    'rmse': 0,
                    'skipped': True
                })
    
    # Save results
    df = pd.DataFrame(all_results)
    csv_path = os.path.join(output_dir, anatomy, 'metrics_enhanced.csv')
    df.to_csv(csv_path, index=False)
    print(f"Saved metrics to {csv_path}")
    
    # Print summary
    valid_df = df[df['skipped'] == False]
    if len(valid_df) > 0:
        stats = valid_df.groupby('threshold')[['ssim', 'rmse']].mean()
        print("\nThreshold Stats (Enhanced):")
        print(stats.to_string())
    
    return df


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--anatomy', type=str, required=True, choices=['head', 'body'])
    parser.add_argument('--input_dir', type=str, default='results_threshold_sweep')
    parser.add_argument('--output_dir', type=str, default='results_threshold_sweep_enhanced')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    process_anatomy(args.anatomy, args.input_dir, args.output_dir)
    
    print("\nDone!")


if __name__ == '__main__':
    main()
