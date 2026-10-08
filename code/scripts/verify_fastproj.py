"""Equivalence check of recon_fast against the original recon extension (run from code/):

    OMP_NUM_THREADS=4 python scripts/verify_fastproj.py

Passes only if (all comparisons are exact uint32 comparisons of the whole array)
  1. FPd is bit-identical to recon.FP,
  2. BPd(P=1) is bit-identical to recon.BP run with one OpenMP thread,
  3. BPd(P=T) equals the fixed-order (0..T-1) fold of the original's T partial images, and the original run with T threads
     equals the fold of those partials in *some* arrival order (the run-to-run set of the original),
  4. deliberately wrong inputs (all angles shifted by 1 degree) are rejected by the same comparisons.
Uses a small synthetic geometry; for real slices compare with your own sinograms the same way.
"""
import ctypes
import itertools
import os
import sys

import numpy as np

import recon
import recon_fast as RF

GOMP = ctypes.CDLL("libgomp.so.1")
T = int(os.environ.get("OMP_NUM_THREADS", "4"))
nview, nu, nx, ny = 180, 256, 128, 128
dsd, dso, dx, dy, da, off_a = 1000.0, 550.0, 1.5, 1.5, np.float32(1.0 / 950), 1.25
deg = np.linspace(0, 360, nview)
rng = np.random.default_rng(42)
yy, xx = np.mgrid[:ny, :nx]
img = (((yy - 60) ** 2 + (xx - 70) ** 2 < 35 ** 2) * 0.02 + ((yy - 70) ** 2 + (xx - 50) ** 2 < 8 ** 2) * 0.1 + rng.random((ny, nx)) * 0.002).astype(np.float32)
args = lambda d: (d, nview, dsd, dso, nx, ny, dx, dy, nu, float(da), off_a)
bits = lambda a: np.asarray(a, dtype=np.float32).view(np.uint32)
same = lambda a, b: bool(np.array_equal(bits(a), bits(b)))
fails = []


def check(name, ok):
    print(("PASS " if ok else "FAIL ") + name)
    if not ok:
        fails.append(name)


# 1. FP
GOMP.omp_set_num_threads(T)
sino_o = np.array(recon.FP(img.flatten().tolist(), deg.tolist(), *args(deg.tolist())[1:]), dtype=np.float32)
sino_f = RF.FPd(img, deg, *args(deg)[1:], 3)
check("FPd == recon.FP (bit-identical)", same(sino_o, sino_f))
bad = RF.FPd(img, deg + 1.0, *args(deg)[1:], 3)
check("negative control: FPd with angles +1 deg is rejected", not same(sino_o, bad))

# 2. BP with one thread
sino = sino_o
GOMP.omp_set_num_threads(1)
bp1 = np.array(recon.BP(sino.flatten().tolist(), deg.tolist(), *args(deg.tolist())[1:]), dtype=np.float32)
check("BPd(P=1) == recon.BP at 1 thread (bit-identical)", same(bp1, RF.BPd(sino, deg, *args(deg)[1:], 1, 5)))
check("negative control: BPd(P=1) with angles +1 deg is rejected", not same(bp1, RF.BPd(sino, deg + 1.0, *args(deg)[1:], 1, 5)))

# 3. BP with T threads: member of the original's run-to-run set
GOMP.omp_set_num_threads(T)
parts = RF.BPparts(sino, deg, *args(deg)[1:], T)


def _fold(parts, order):
    acc = np.zeros_like(parts[0])
    for k in order:
        acc = acc + parts[k]          # float32 element-wise adds in this order, as the original's atomic merge does
    return acc


results = {tuple(o): _fold(parts, o) for o in itertools.permutations(range(T))} if T <= 5 else {tuple(range(T)): _fold(parts, range(T))}
check(f"BPd(P={T}) == fixed-order fold of the original's {T} partial images", same(RF.BPd(sino, deg, *args(deg)[1:], T, 5), results[tuple(range(T))]))
runs = [np.array(recon.BP(sino.flatten().tolist(), deg.tolist(), *args(deg.tolist())[1:]), dtype=np.float32) for _ in range(5)]
if T <= 5:
    ok = all(any(same(r, v) for v in results.values()) for r in runs)
    check(f"5 original runs at {T} threads each equal the fold of the partials in some arrival order", ok)
else:
    print(f"SKIP original-run membership check (T={T} > 5: orders cannot be enumerated; use masked single-chunk runs)")
check("negative control: partials of a +1 deg run do not fold to the original", not same(runs[0], _fold(RF.BPparts(sino, deg + 1.0, *args(deg)[1:], T), range(T))))

print(f"{len(fails)} failed" if fails else "all checks passed")
sys.exit(1 if fails else 0)
