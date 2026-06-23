#!/usr/bin/env python
"""
InDuDoNet+ test set evaluation.

Uses the SAME metric functions as paper table (calculate_metrics_all.py):
  - SSIM: raw HU, metal masked
  - FSIM: normalized [-150,400] to [0,255], metal masked
  - RMSE: raw HU, metal masked

Usage:
    source activate.sh
    python eval.py --data_path ../../data/01_raw/ --index_path ../../code/index/ \
        --anatomy body --model_path ./models/body_v3/InDuDoNet_best.pt
"""
from __future__ import print_function
import argparse
import os
import sys
import numpy as np
import cv2
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd

from deeplesion.Dataset import CTMARDataset, image_normalize_minmax
from network.indudonet import InDuDoNet

from phasepack import phasecong


# ---------- Same metric functions as paper table (calculate_metrics_all.py) ----------

def normalize_image(image, vmin, vmax):
    tmp_image = np.clip(image, vmin, vmax)
    tmp_image = (tmp_image - vmin) / (vmax - vmin) * 255
    return tmp_image.astype(np.uint8)


def calculate_ssim(img1, img2):
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


def calculate_fsim(gray1, gray2):
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


def calculate_rmse(image1, image2):
    mse = np.mean((image1 - image2) ** 2)
    return np.sqrt(mse)


def compute_metrics(gt_image, pred_image, gt_metal_mask, anatomy):
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
    # @MX:NOTE [AUTO]: match paper scale (eval_nmar/eval_mdt/calculate_metrics_all multiply by 100)
    fsim_score = calculate_fsim(gt_norm, pred_norm) * 100

    return {'ssim': ssim_score, 'fsim': fsim_score, 'rmse': rmse_score}


# ---------- Helpers ----------

def denormalize_to_hu(img_255, minmax=None):
    """Convert [0,255] normalized image back to HU."""
    if minmax is None:
        minmax = image_normalize_minmax()
    data_min, data_max = minmax
    img_01 = img_255 / 255.0
    return img_01 * (data_max - data_min) + data_min


# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--index_path", type=str, required=True)
    parser.add_argument("--anatomy", type=str, required=True, choices=["body", "head"])
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--num_channel", type=int, default=32)
    parser.add_argument("--T", type=int, default=4)
    parser.add_argument("--S", type=int, default=10)
    parser.add_argument("--eta1", type=float, default=1)
    parser.add_argument("--eta2", type=float, default=5)
    parser.add_argument("--alpha", type=float, default=0.5)
    args = parser.parse_args()

    cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load model
    net = InDuDoNet(args).cuda()
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        net.load_state_dict(checkpoint['model_state_dict'])
    elif 'model' in checkpoint:
        net.load_state_dict(checkpoint['model'])
    else:
        net.load_state_dict(checkpoint)
    print(f"Loaded checkpoint: {args.model_path} (epoch {checkpoint.get('epoch', '?')})")
    net.eval()

    # Load test dataset
    test_dataset = CTMARDataset(
        data_root=args.data_path,
        anatomy=args.anatomy,
        index_path=args.index_path,
        split='test'
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)
    print(f"Test set: {len(test_dataset)} samples ({args.anatomy})")

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    all_ssim, all_fsim, all_rmse = [], [], []

    with torch.no_grad():
        for idx, data in enumerate(tqdm(test_loader, desc="Evaluating")):
            Xma, XLI, Xgt, mask, Sma, SLI, Sgt, Tr = [x.cuda() for x in data]

            ListX, ListS, ListYS = net(Xma, XLI, mask, Sma, SLI, Tr)
            pred_255 = ListX[-1].squeeze().cpu().numpy()
            gt_255 = Xgt.squeeze().cpu().numpy()
            metal_mask = mask.squeeze().cpu().numpy()

            # Convert back to HU
            pred_hu = denormalize_to_hu(pred_255)
            gt_hu = denormalize_to_hu(gt_255)

            # Compute metrics (same as paper table)
            metrics = compute_metrics(gt_hu, pred_hu, metal_mask, args.anatomy)
            all_ssim.append(metrics['ssim'])
            all_fsim.append(metrics['fsim'])
            all_rmse.append(metrics['rmse'])

            if args.output_dir:
                pred_hu.astype(np.float32).tofile(
                    os.path.join(args.output_dir, f"pred_{idx:05d}.raw"))

    all_ssim = np.array(all_ssim)
    all_fsim = np.array(all_fsim)
    all_rmse = np.array(all_rmse)

    print(f"\n{'='*60}")
    print(f"InDuDoNet+ Test Results ({args.anatomy}, n={len(test_dataset)})")
    print(f"{'='*60}")
    print(f"  SSIM:  {all_ssim.mean():.4f} +/- {all_ssim.std():.4f}")
    print(f"  FSIM:  {all_fsim.mean():.4f} +/- {all_fsim.std():.4f}")
    print(f"  RMSE:  {all_rmse.mean():.4f} +/- {all_rmse.std():.4f}")
    print(f"{'='*60}")

    df = pd.DataFrame({'ssim': all_ssim, 'fsim': all_fsim, 'rmse': all_rmse})
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'logs')
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f'indudonet_metrics_{args.anatomy}.csv')
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")


if __name__ == '__main__':
    main()
