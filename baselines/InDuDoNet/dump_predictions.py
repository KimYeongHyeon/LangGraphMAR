#!/usr/bin/env python
"""
Dump InDuDoNet+ per-sample predictions to `code/paper_results/data/{anat}_{id}/indudonet_image.npy`.

Output format and shape match the 6 existing methods in that directory (HU float32, (512,512,1)).

Usage (inside baselines/InDuDoNet/ with .venv activated):
    python dump_predictions.py --anatomy body \
        --model_path ./models/body_v3/InDuDoNet_best.pt
    python dump_predictions.py --anatomy head \
        --model_path ./models/head_v3/InDuDoNet_best.pt
"""
from __future__ import print_function

import argparse
import os
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm

from deeplesion.Dataset import CTMARDataset, image_normalize_minmax
from network.indudonet import InDuDoNet


def denormalize_to_hu(img_255, minmax=None):
    if minmax is None:
        minmax = image_normalize_minmax()
    data_min, data_max = minmax
    img_01 = img_255 / 255.0
    return img_01 * (data_max - data_min) + data_min


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anatomy", required=True, choices=["body", "head"])
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--data_path", default="../../data/01_raw/")
    ap.add_argument("--index_path", default="../../code/index/")
    ap.add_argument("--target_root",
                    default="../../code/paper_results/data/")
    ap.add_argument("--num_channel", type=int, default=32)
    ap.add_argument("--T", type=int, default=4)
    ap.add_argument("--S", type=int, default=10)
    ap.add_argument("--eta1", type=float, default=1)
    ap.add_argument("--eta2", type=float, default=5)
    ap.add_argument("--alpha", type=float, default=0.5)
    args = ap.parse_args()

    cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    net = InDuDoNet(args).cuda()
    ckpt = torch.load(args.model_path, map_location=device, weights_only=False)
    net.load_state_dict(ckpt.get('model_state_dict', ckpt.get('model', ckpt)))
    print(f"Loaded: {args.model_path} (epoch {ckpt.get('epoch', '?')})")
    net.eval()

    ds = CTMARDataset(
        data_root=args.data_path, anatomy=args.anatomy,
        index_path=args.index_path, split='test',
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=4)
    print(f"Test set: {len(ds)} samples ({args.anatomy})")

    # idx in loader -> file_idx in raw data -> sample folder name
    file_indices = ds.indices   # list[int]

    target_root = Path(args.target_root)
    n_saved, n_missing = 0, 0

    with torch.no_grad():
        for idx, data in enumerate(tqdm(loader, desc=f"Inferring {args.anatomy}")):
            Xma, XLI, Xgt, mask, Sma, SLI, Sgt, Tr = [x.cuda() for x in data]
            ListX, ListS, ListYS = net(Xma, XLI, mask, Sma, SLI, Tr)
            pred_255 = ListX[-1].squeeze().cpu().numpy()
            pred_hu = denormalize_to_hu(pred_255).astype(np.float32)

            # reshape to (512,512,1) to match other methods' layout
            if pred_hu.ndim == 2:
                pred_hu = pred_hu[..., None]

            file_idx = file_indices[idx]
            sample_dir = target_root / f"{args.anatomy}_{file_idx}"
            if not sample_dir.exists():
                n_missing += 1
                continue
            np.save(sample_dir / "indudonet_image.npy", pred_hu)
            n_saved += 1

    print(f"\nDone. Saved: {n_saved}/{len(ds)} "
          f"(skipped missing folders: {n_missing})")


if __name__ == '__main__':
    main()
