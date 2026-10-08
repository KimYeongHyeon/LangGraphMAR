"""Equivalence check of utils/gpuproj.py against the original recon extension (run from code/, needs a CUDA GPU):

    python scripts/verify_gpuproj.py

Uses the paper's scanner geometry (512x512, 1000 views, 900 detector channels, off_a 1.25) and synthetic phantoms (soft-tissue
ellipses plus bright "metal" discs), at two pixel sizes. Passes only if
  1. the GPU FP is bit-identical to recon.FP (exact uint32 comparison of the whole sinogram),
  2. the GPU BP differs from recon.BP by no more than 3x the original's own run-to-run difference (two runs of recon.BP on the
     same input; the original is an OpenMP reduction, so it is not bit-reproducible) and by at most 1e-5 of the image maximum,
  3. deliberately wrong inputs are rejected by the same comparisons: angles shifted by 1 degree must NOT match the original FP
     bit for bit, and the GPU BP with shifted angles must differ from the original by far more than the noise.
"""
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import recon                                  # noqa: F401  (the original extension, imported by utils.projection too)
import utils.projection as UP
import utils.gpuproj as G


def ct_param(pix):
    p = OrderedDict({"nx": 512, "ny": 512, "DSD": 950.0, "DSO": 550.0, "nu": 900, "du": 1.0, "nview": 1000, "filter": "ram-lak"})
    p["deg"] = np.linspace(0, 360, p["nview"], endpoint=False)
    p["fan_angle"] = p["du"] / p["DSD"] * 180 / np.pi * p["nu"]
    p["da"] = p["fan_angle"] / p["nu"] / 180 * np.pi
    p["off_a"] = 1.25
    p["dx"] = p["dy"] = pix
    return p


def phantom():
    """Attenuation image in 1/mm: water-like body, a lung-like hole, bone and two metal discs (about 0.4-0.8 1/mm)."""
    yy, xx = np.mgrid[:512, :512].astype(np.float32)
    r = lambda cy, cx, ry, rx: (((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2) < 1
    img = np.zeros((512, 512), np.float32)
    img[r(256, 256, 170, 210)] = 0.02
    img[r(250, 190, 90, 60)] = 0.004
    img[r(250, 320, 90, 60)] = 0.004
    img[r(330, 256, 30, 25)] = 0.04
    img[r(240, 160, 9, 9)] = 0.7
    img[r(300, 350, 6, 6)] = 0.45
    return img


bits = lambda a: np.asarray(a, dtype=np.float32).view(np.uint32)
fails = []


def check(name, ok):
    print(("PASS " if ok else "FAIL ") + name)
    if not ok:
        fails.append(name)


img = phantom()
for pix in (400.0 / 512, 0.6):
    p = ct_param(pix)
    tag = f"pixel {pix:.4f} mm"
    s0, s1 = UP.fp(img, p), G.fp(img, p)
    check(f"{tag}: GPU FP == recon.FP (bit-identical)", bool(np.array_equal(bits(s0), bits(s1))))
    q = dict(p, deg=p["deg"] + 1.0)
    check(f"{tag}: negative control: FP with angles +1 deg is rejected", not np.array_equal(bits(s0), bits(G.fp(img, q))))

    f = UP.filtering(s0, p)
    b0, b0b, b1 = UP.bp(f, p), UP.bp(f, p), G.bp(f, p)
    gap, noise, top = float(np.abs(b1 - b0).max()), float(np.abs(b0b - b0).max()), float(np.abs(b0).max())
    print(f"     BP max|GPU-orig| {gap:.3g}, max|orig run 2 - orig run 1| {noise:.3g}, image max {top:.3g}, relative gap {gap / top:.3g}")
    check(f"{tag}: GPU BP within 3x the original's run-to-run noise and 1e-5 of the image maximum",
          gap <= 3 * max(noise, 1e-9) and gap / top <= 1e-5)
    bad = float(np.abs(G.bp(UP.filtering(UP.fp(img, q), q), q) - b0).max())
    check(f"{tag}: negative control: BP of angle-shifted data is far outside the noise", bad > 30 * max(noise, 1e-9))

print(f"{len(fails)} failed" if fails else "all checks passed")
sys.exit(1 if fails else 0)
