"""
Metal Deletion Technique (MDT) for CT Metal Artifact Reduction.

Reference: Boas FE, Fleischmann D.
    "Evaluation of Two Iterative Techniques for Reducing Metal Artifacts
     in Computed Tomography."
    Radiology 259(3), 2011.

Algorithm (2-pass by default, optionally iterated):

    Pass 1 (initial correction):
        1. Detect metal mask (provided).
        2. Forward-project metal mask -> metal trace in sinogram domain.
        3. Linear interpolation of sinogram inside metal trace.
        4. FBP reconstruction -> img_mdt_initial.

    Pass 2+ (iterative refinement):
        5. Use previous image as prior:
               prior_img[metal] <- u_water  (remove metal contamination)
           Forward-project prior -> proj_prior.
        6. Replace metal-trace values in the *original* sinogram with the
           corresponding values from ``proj_prior`` (i.e. model-based fill
           instead of linear fit).
        7. FBP reconstruction -> img_mdt_refined.
        8. Repeat from step 5 up to ``iterations`` total passes.

Distinction vs NMAR (Meyer 2010):
    NMAR normalises the sinogram by a KMeans-segmented 3-class prior image,
    interpolates in the normalised domain, then de-normalises. MDT uses no
    KMeans and no normalisation -- its prior is a simple water-filled image
    and refinement works directly in the projection domain.

This implementation reuses ``proj_interp`` and ``setup_ct_params`` from
``baselines/NMAR/nmar.py``, plus ``filtering/bp/fp`` from ``code/utils/projection.py``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Pull CT helpers from the project
CODE_DIR = Path(__file__).parent.parent.parent / "code"
sys.path.insert(0, str(CODE_DIR))

# Reuse NMAR's tested utilities
NMAR_DIR = Path(__file__).parent.parent / "NMAR"
sys.path.insert(0, str(NMAR_DIR))

from utils.projection import filtering, bp, fp  # noqa: E402
from nmar import setup_ct_params, proj_interp  # noqa: E402

MIU_AIR = 0.0
MIU_WATER = 0.02  # /mm (project convention, see nmar.py)


def fbp_reconstruct(sinogram: np.ndarray, ct_param: dict) -> np.ndarray:
    """Filtered back-projection using project FP/BP utilities."""
    p = ct_param.copy()
    p["filter"] = "ram-lak"
    sino_filt = filtering(sinogram, p)
    return np.maximum(bp(sino_filt, p).astype(np.float32), 0.0)


def mdt(
    sinogram: np.ndarray,
    metal_mask_img: np.ndarray,
    anatomy: str,
    ct_param: dict | None = None,
    iterations: int = 2,
) -> np.ndarray:
    """Run MDT on a single sample.

    Parameters
    ----------
    sinogram : ndarray (1000, 900)
        Metal-corrupted sinogram (post-BHC /mm path integral, same domain as nmar).
    metal_mask_img : ndarray (512, 512)
        Binary metal mask in image domain (1 = metal).
    anatomy : str
        'body' or 'head'.
    ct_param : dict, optional
        CT parameter dict from setup_ct_params(); auto-created if None.
    iterations : int, default=2
        Number of full passes. 1 = plain LI, 2 = Boas & Fleischmann baseline, 3+ = more refinement.

    Returns
    -------
    img_mdt : ndarray (512, 512)
        Reconstructed image in /mm linear attenuation.
    """
    if iterations < 1:
        raise ValueError(f"iterations must be >= 1, got {iterations}")
    if ct_param is None:
        ct_param = setup_ct_params(anatomy)

    # Forward-project metal mask to get sinogram trace once (shared across passes)
    metal_trace_sino = fp(metal_mask_img.astype(np.float32), ct_param).astype(np.float32)
    metal_trace = (metal_trace_sino > 0).astype(np.float32)

    # ---- Pass 1: Linear Interpolation + FBP ----
    proj_li = proj_interp(sinogram, metal_trace)
    img_current = fbp_reconstruct(proj_li, ct_param)

    if iterations == 1:
        return img_current

    # ---- Passes 2..N: iterative refinement ----
    for _ in range(iterations - 1):
        # Replace metal region in current image with water to form prior
        prior_img = img_current.copy()
        prior_img[metal_mask_img > 0.5] = MIU_WATER

        # Forward-project prior to get model-based sinogram
        proj_prior = fp(prior_img.astype(np.float32), ct_param).astype(np.float32)

        # Fill metal trace in measured sinogram with prior projections
        proj_filled = sinogram.copy()
        proj_filled[metal_trace > 0] = proj_prior[metal_trace > 0]

        # FBP
        img_current = fbp_reconstruct(proj_filled, ct_param)

    return img_current


if __name__ == "__main__":
    # quick sanity check
    print("MDT module loaded.")
    print(f"  MIU_WATER = {MIU_WATER}")
    print(f"  reusing proj_interp from {NMAR_DIR / 'nmar.py'}")
    for anat in ("body", "head"):
        p = setup_ct_params(anat)
        print(f"  {anat}: DSD={p['DSD']}, DSO={p['DSO']}, dx={p['dx']:.4f}, nview={p['nview']}")
