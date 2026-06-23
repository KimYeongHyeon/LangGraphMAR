#!/usr/bin/env python
"""
Dump InDuDoNet+ predictions matching the `code/paper_results/data/{anat}_{N}/` folder set.

Key fix vs. first attempt: data folder N corresponds to raw file number N (1-indexed),
which maps to sorted_idx = N-1 in CTMARDataset's file_lists. The first attempt treated
pkl values as folder names, producing an off-by-one mismatch between stored predictions
and the GT stored in each folder.

Saves float32 arrays of shape (512,512,1) in HU space, matching the layout of the
six existing prediction files (FBP_image.npy, proposed_image.npy, etc.).
"""
from __future__ import print_function

import argparse
import os
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
    return (img_255 / 255.0) * (data_max - data_min) + data_min


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

    net = InDuDoNet(args).cuda()
    ckpt = torch.load(args.model_path, map_location=device, weights_only=False)
    net.load_state_dict(ckpt.get('model_state_dict', ckpt.get('model', ckpt)))
    print(f"Loaded: {args.model_path} (epoch {ckpt.get('epoch', '?')})")
    net.eval()

    # ----- Build indices from data/ folder list (NOT from pkl) -----
    target_root = Path(args.target_root)
    folder_nums = sorted(
        int(p.name.split("_")[1])
        for p in target_root.iterdir()
        if p.is_dir() and p.name.startswith(f"{args.anatomy}_")
    )

    # Construct dataset with split='test' (arbitrary — we override indices next)
    ds = CTMARDataset(
        data_root=args.data_path, anatomy=args.anatomy,
        index_path=args.index_path, split='test',
    )
    # head raw files are non-consecutive, so build file_num -> sorted_idx map
    import re
    file_num_to_idx = {
        int(re.search(r'img(\d+)', os.path.basename(p)).group(1)): i
        for i, p in enumerate(ds.file_lists['ma_img'])
    }
    kept_folder_nums = [N for N in folder_nums if N in file_num_to_idx]
    skipped = [N for N in folder_nums if N not in file_num_to_idx]
    if skipped:
        print(f"WARN: {len(skipped)} folders skipped (no matching raw file): {skipped[:10]}...")
    sorted_indices = [file_num_to_idx[N] for N in kept_folder_nums]
    folder_nums = kept_folder_nums
    print(f"{args.anatomy}: {len(folder_nums)} folders mapped, "
          f"sorted_idx range {min(sorted_indices)}..{max(sorted_indices)}")
    ds.indices = sorted_indices   # override: iterate in folder order
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=4)

    n_saved, n_missing = 0, 0
    with torch.no_grad():
        pbar = tqdm(loader, desc=f"Inferring {args.anatomy}")
        for idx, data in enumerate(pbar):
            Xma, XLI, Xgt, mask, Sma, SLI, Sgt, Tr = [x.cuda() for x in data]
            ListX, ListS, ListYS = net(Xma, XLI, mask, Sma, SLI, Tr)
            pred_255 = ListX[-1].squeeze().cpu().numpy()
            pred_hu = denormalize_to_hu(pred_255).astype(np.float32)
            if pred_hu.ndim == 2:
                pred_hu = pred_hu[..., None]

            folder_N = folder_nums[idx]
            sample_dir = target_root / f"{args.anatomy}_{folder_N}"
            if not sample_dir.exists():
                n_missing += 1
                continue
            np.save(sample_dir / "indudonet_image.npy", pred_hu)
            n_saved += 1

    print(f"\nSaved: {n_saved}/{len(ds)}  (missing folders: {n_missing})")


if __name__ == '__main__':
    main()
