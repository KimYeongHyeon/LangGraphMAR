"""Opt-in GPU projector for the LangGraph-MAR pipeline (the original code paths are unchanged unless you call this).

    from utils.gpuproj import install_gpuproj
    install_gpuproj()           # before building the graph: rebinds fp/bp in utils.projection, utils.algorithm, utils.graph

Computes the same FP and BP as `recon.c` on a CUDA GPU (Triton). Needs a CUDA GPU, PyTorch with Triton, and a C compiler that
Triton can find (set the `CC` environment variable if there is no system gcc). Nothing is built; the kernels compile on first use.

Equivalence to the original `recon` extension (checked by scripts/verify_gpuproj.py):
  * FP is bit-identical. Each ray is independent, so the original is deterministic and the kernel follows its float32
    arithmetic line by line (same cast order; float32 divisions are done in float64 and rounded, because Triton's float32 `/`
    is not IEEE-rounded and flips boundary rays).
  * BP is the same ray-driven scatter (the transpose of FP), but it is *not* bit-identical and is not a member of the original's
    run-to-run result set. The original adds float32 per-thread partial images; the kernel scatters with atomic adds, in an
    arbitrary order. Accumulating in float32 made the difference to the original about 8x the original's own run-to-run noise
    (thousands of adds land on one pixel), so the adds are done in float64 (products are still float32, as in the original).
    With that the difference is the same size as the original's run-to-run noise (about 3e-7 relative; 0.012 HU on a body
    slice). If you need exact membership in the original's result set use `recon_fast` instead.

The geometry is read from the same `param` dict as the original wrappers (DSO, da, off_a, dx, dy, nx, ny, nu, nview, deg), so any
pixel size works. Returned arrays have the original wrappers' dtype and shape: float64 arrays holding float32 values.
"""
import math
from functools import lru_cache

import numpy as np
import torch
import triton
import triton.language as tl

BLOCK = 64


@triton.jit
def _ray(ray, sinv, cosv, tana, dist, nu, live):
    v = ray // nu
    u = ray % nu
    s = tl.load(sinv + v, mask=live, other=0.0)
    c = tl.load(cosv + v, mask=live, other=0.0)
    ta = tl.load(tana + u, mask=live, other=0.0)          # float64: tan((double) a)
    di = tl.load(dist + u, mask=live, other=0.0)          # float32: (float)(dy / cos((double) a))
    return s, c, ta, di


@triton.jit
def _fp_kernel(img, out, sinv, cosv, tana, dist, posy, posyd, fc, fd, nview, nu, nx, ny, BLOCK: tl.constexpr):
    ray = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    nray = nview.to(tl.int64) * nu
    live = ray < nray
    s, c, ta, di = _ray(ray, sinv, cosv, tana, dist, nu, live)
    dx = tl.load(fd + 0)
    dy = tl.load(fd + 1)
    cx = tl.load(fc + 2)
    cy = tl.load(fc + 3)
    xm = tl.load(fc + 4)
    ym = tl.load(fc + 5)
    temp = tl.zeros((BLOCK,), dtype=tl.float32)
    for iy in range(0, ny):
        py = tl.load(posy + iy)
        px = (ta * tl.load(posyd + iy).to(tl.float64)).to(tl.float32)
        rx = ((px * c + py * s).to(tl.float64) / dx).to(tl.float32) + cx
        ry = ((-px * s + py * c).to(tl.float64) / dy).to(tl.float32) + cy
        hit = live & (rx > 0.0) & (rx < xm) & (ry > 0.0) & (ry < ym)
        ix = tl.where(hit, rx, 0.0).to(tl.int64)
        jy = tl.where(hit, ry, 0.0).to(tl.int64)
        wx = rx - ix.to(tl.float32)
        wy = ry - jy.to(tl.float32)
        at = jy * nx + ix
        o00 = tl.load(img + at, mask=hit, other=0.0)
        o01 = tl.load(img + at + 1, mask=hit, other=0.0)
        o10 = tl.load(img + at + nx, mask=hit, other=0.0)
        o11 = tl.load(img + at + nx + 1, mask=hit, other=0.0)
        val = ((1.0 - wx) * (1.0 - wy) * o00 + wx * (1.0 - wy) * o01 + (1.0 - wx) * wy * o10 + wx * wy * o11) * di
        temp = tl.where(hit, temp + val, temp)
    tl.store(out + ray, temp, mask=live)


@triton.jit
def _bp_kernel(sino, out, sinv, cosv, tana, dist, posy, posyd, fc, fd, nview, nu, nx, ny, BLOCK: tl.constexpr):
    ray = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    nray = nview.to(tl.int64) * nu
    live = ray < nray
    s, c, ta, di = _ray(ray, sinv, cosv, tana, dist, nu, live)
    dx = tl.load(fd + 0)
    dy = tl.load(fd + 1)
    cx = tl.load(fc + 2)
    cy = tl.load(fc + 3)
    xm = tl.load(fc + 4)
    ym = tl.load(fc + 5)
    sv = tl.load(sino + ray, mask=live, other=0.0)
    for iy in range(0, ny):
        py = tl.load(posy + iy)
        px = (ta * tl.load(posyd + iy).to(tl.float64)).to(tl.float32)
        rx = ((px * c + py * s).to(tl.float64) / dx).to(tl.float32) + cx
        ry = ((-px * s + py * c).to(tl.float64) / dy).to(tl.float32) + cy
        hit = live & (rx > 0.0) & (rx < xm) & (ry > 0.0) & (ry < ym)
        ix = tl.where(hit, rx, 0.0).to(tl.int64)
        jy = tl.where(hit, ry, 0.0).to(tl.int64)
        wx = rx - ix.to(tl.float32)
        wy = ry - jy.to(tl.float32)
        val = sv * di
        at = jy * nx + ix
        # products in float32 as in the original; only the accumulation is float64 (see the module docstring)
        tl.atomic_add(out + at, ((1.0 - wx) * (1.0 - wy) * val).to(tl.float64), mask=hit)
        tl.atomic_add(out + at + 1, (wx * (1.0 - wy) * val).to(tl.float64), mask=hit)
        tl.atomic_add(out + at + nx, ((1.0 - wx) * wy * val).to(tl.float64), mask=hit)
        tl.atomic_add(out + at + nx + 1, (wx * wy * val).to(tl.float64), mask=hit)


def _device():
    assert torch.cuda.is_available(), "install_gpuproj() needs a CUDA GPU"
    return torch.device("cuda", torch.cuda.current_device())


def _key(p):
    return (int(p["nview"]), int(p["nx"]), int(p["ny"]), int(p["nu"]), float(p["dx"]), float(p["dy"]), float(p["DSO"]), float(p["da"]),
            float(p["off_a"]), tuple(np.asarray(p["deg"], np.float64).tolist()), torch.cuda.current_device())


@lru_cache(maxsize=8)
def _geometry(key):
    """View, channel and row constants, built with recon.c's cast order."""
    nview, nx, ny, nu, dx, dy, dso, da, off_a, deg, _ = key
    f32 = np.float32
    pi32 = float(f32(3.141592))
    th = [d / 180.0 * pi32 for d in deg]
    s = np.array([math.sin(t) for t in th], f32)
    c = np.array([math.cos(t) for t in th], f32)
    dx_, dy_, dso_, da_, off_ = f32(dx), f32(dy), f32(dso), f32(da), f32(off_a)
    a = ((np.arange(nu).astype(f32) - f32((nu - 1.0) / 2.0)) - off_) * da_
    tana = np.array([math.tan(float(x)) for x in a], np.float64)
    dist = np.array([float(dy_) / math.cos(float(x)) for x in a], np.float64).astype(f32)
    posy = (np.arange(ny).astype(f32) - f32((ny - 1.0) / 2.0)) * dy_
    posyd = posy + dso_
    fc = np.array([dx_, dy_, (nx - 1.0) / 2.0, (ny - 1.0) / 2.0, nx - 1, ny - 1], f32)
    fd = np.array([dx_, dy_], np.float64)
    dev = _device()
    return tuple(torch.as_tensor(x, device=dev) for x in (s, c, tana, dist, posy, posyd, fc, fd))


def fp(img, p):
    """Same signature and return as `utils.projection.fp`: image (ny, nx) -> sinogram (nview, nu)."""
    key = _key(p)
    g = _geometry(key)
    nview, nx, ny, nu = key[:4]
    t = torch.as_tensor(np.ascontiguousarray(img, np.float32), device=_device()).reshape(-1)
    out = torch.zeros(nview * nu, dtype=torch.float32, device=_device())
    _fp_kernel[(triton.cdiv(nview * nu, BLOCK),)](t, out, *g, nview, nu, nx, ny, BLOCK=BLOCK, enable_fp_fusion=False)
    return out.reshape(nview, nu).double().cpu().numpy()


def bp(proj, p):
    """Same signature and return as `utils.projection.bp`: sinogram (nview, nu) -> image (nx, ny)."""
    key = _key(p)
    g = _geometry(key)
    nview, nx, ny, nu = key[:4]
    t = torch.as_tensor(np.ascontiguousarray(proj, np.float32), device=_device()).reshape(-1)
    out = torch.zeros(nx * ny, dtype=torch.float64, device=_device())
    _bp_kernel[(triton.cdiv(nview * nu, BLOCK),)](t, out, *g, nview, nu, nx, ny, BLOCK=BLOCK, enable_fp_fusion=False)
    return out.float().reshape(nx, ny).double().cpu().numpy()      # values are float32, as the original wrapper returns


def install_gpuproj():
    import utils.projection as UP
    import utils.algorithm as UA
    import utils.graph as UG
    for m in (UP, UA, UG):
        m.fp, m.bp = fp, bp
    return __file__
