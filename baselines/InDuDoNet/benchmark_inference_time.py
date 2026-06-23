#!/usr/bin/env python
"""Benchmark InDuDoNet+ forward inference time on CTMAR samples.

The timing excludes raw-file loading and output saving. It includes the full
network forward pass, including the ODL forward/backprojection operators inside
InDuDoNet+.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from deeplesion.Dataset import CTMARDataset  # noqa: E402
from network.indudonet import InDuDoNet  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--anatomy", required=True, choices=["body", "head"])
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_path", default="../../data/01_raw/")
    parser.add_argument("--index_path", default="../../code/index/")
    parser.add_argument("--out_dir", default="../../logs")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sample_count", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--num_channel", type=int, default=32)
    parser.add_argument("--T", type=int, default=4)
    parser.add_argument("--S", type=int, default=10)
    parser.add_argument("--eta1", type=float, default=1)
    parser.add_argument("--eta2", type=float, default=5)
    parser.add_argument("--alpha", type=float, default=0.5)
    return parser.parse_args()


def load_model(args, device):
    net = InDuDoNet(args).to(device)
    ckpt = torch.load(args.model_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt.get("model", ckpt))
    net.load_state_dict(state_dict)
    net.eval()
    return net, ckpt.get("epoch", "?")


def load_samples(args, device):
    ds = CTMARDataset(
        data_root=args.data_path,
        anatomy=args.anatomy,
        index_path=args.index_path,
        split="test",
    )
    n = min(args.sample_count, len(ds))
    samples = []
    for idx in range(n):
        data = ds[idx]
        Xma, XLI, _Xgt, mask, Sma, SLI, _Sgt, Tr = [
            x.unsqueeze(0).to(device, non_blocking=True) for x in data
        ]
        samples.append((idx, (Xma, XLI, mask, Sma, SLI, Tr)))
    return samples


@torch.no_grad()
def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")

    cudnn.benchmark = True
    device = torch.device(args.device)
    torch.cuda.set_device(device)

    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"Loading model: {args.model_path}")
    net, epoch = load_model(args, device)
    print(f"Loaded checkpoint epoch: {epoch}")

    samples = load_samples(args, device)
    print(
        f"Benchmarking {args.anatomy}: {len(samples)} sample(s), "
        f"warmup={args.warmup}, runs={args.runs}"
    )

    timings = []
    for sample_idx, sample in samples:
        for _ in range(args.warmup):
            net(*sample)
        torch.cuda.synchronize(device)

        for run_idx in range(args.runs):
            start = time.perf_counter()
            net(*sample)
            torch.cuda.synchronize(device)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            timings.append(
                {
                    "anatomy": args.anatomy,
                    "sample_index": sample_idx,
                    "run_index": run_idx,
                    "time_ms": elapsed_ms,
                }
            )

    arr = np.array([row["time_ms"] for row in timings], dtype=np.float64)
    summary = {
        "anatomy": args.anatomy,
        "checkpoint": args.model_path,
        "checkpoint_epoch": epoch,
        "device": torch.cuda.get_device_name(device),
        "sample_count": len(samples),
        "warmup": args.warmup,
        "runs_per_sample": args.runs,
        "n_timings": len(arr),
        "mean_ms": float(arr.mean()),
        "std_ms": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "median_ms": float(np.median(arr)),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    detail_path = out_dir / f"indudonet_inference_time_{args.anatomy}.csv"
    summary_path = out_dir / f"indudonet_inference_time_{args.anatomy}_summary.csv"

    with detail_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["anatomy", "sample_index", "run_index", "time_ms"])
        writer.writeheader()
        writer.writerows(timings)

    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)

    print(f"Saved detail: {detail_path}")
    print(f"Saved summary: {summary_path}")
    print(
        "RESULT "
        f"{args.anatomy}: mean={summary['mean_ms']:.2f} ms, "
        f"std={summary['std_ms']:.2f}, median={summary['median_ms']:.2f}, "
        f"min={summary['min_ms']:.2f}, max={summary['max_ms']:.2f}"
    )


if __name__ == "__main__":
    main()
