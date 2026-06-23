"""
Normalized Metal Artifact Reduction (NMAR)

Reference: "Normalized metal artifact reduction (NMAR) in computed tomography"
           https://pubmed.ncbi.nlm.nih.gov/21089784/

Adapted to use CTMAR project's FP/BP (utils/projection.py) instead of LEAP.
"""

import sys
from pathlib import Path
from collections import OrderedDict

import numpy as np
from sklearn.cluster import KMeans
from skimage.filters import gaussian

# Add code/ to path for project imports
CODE_DIR = Path(__file__).parent.parent.parent / 'code'
sys.path.insert(0, str(CODE_DIR))

from utils.projection import filtering, bp, fp
from utils.ct import get_FOV

MIU_AIR = 0
MIU_WATER = 0.02  # /mm (our scanner uses /mm, not /cm)


def setup_ct_params(anatomy):
    """Setup CT params matching CTMAR project scanner geometry."""
    param = OrderedDict({
        'nx': 512, 'ny': 512,
        'DSD': 950.0, 'DSO': 550.0,
        'nu': 900, 'du': 1.0, 'nview': 1000,
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


def proj_interp(proj, metal_trace):
    """Linear interpolation of sinogram in metal regions."""
    num_of_view, num_of_bin = proj.shape
    p_interp = np.zeros_like(proj)

    for i in range(num_of_view):
        mslice = metal_trace[i]
        pslice = proj[i].copy()

        metal_pos = np.nonzero(mslice)[0]
        non_metal_pos = np.where(mslice == 0)[0]

        if len(non_metal_pos) > 1 and len(metal_pos) > 0:
            pslice[metal_pos] = np.interp(metal_pos, non_metal_pos, pslice[non_metal_pos])

        p_interp[i] = pslice

    return p_interp


def nmar_proj_interp(proj, proj_prior, metal_trace):
    """
    Normalized metal artifact reduction in projection domain.

    Steps:
        1. Normalize projection by prior: proj_norm = proj / proj_prior
        2. Interpolate normalized projection in metal regions
        3. Denormalize: proj_nmar = proj_norm_interp * proj_prior
        4. Keep original values in non-metal regions
    """
    proj_prior = proj_prior.copy()
    proj_prior[proj_prior < 0] = 0
    eps = 1e-6
    proj_prior = proj_prior + eps
    proj_norm = proj / proj_prior
    proj_norm_interp = proj_interp(proj_norm, metal_trace)
    proj_nmar = proj_norm_interp * proj_prior
    proj_nmar[metal_trace == 0] = proj[metal_trace == 0]

    return proj_nmar


def nmar(sinogram, metal_mask_img, anatomy, ct_param=None):
    """
    Run NMAR on a single sample.

    Args:
        sinogram: metalart sinogram (1000, 900), /mm attenuation
        metal_mask_img: binary metal mask in image domain (512, 512)
        anatomy: 'body' or 'head'
        ct_param: CT parameters (optional, auto-created if None)

    Returns:
        img_nmar: NMAR-corrected image in /mm attenuation (512, 512)
    """
    if ct_param is None:
        ct_param = setup_ct_params(anatomy)

    # Step 1: Get metal trace in sinogram domain
    metal_trace_sino = fp(metal_mask_img.astype(np.float32), ct_param).astype(np.float32)
    metal_trace = (metal_trace_sino > 0).astype(np.float32)

    # Step 2: Linear interpolation correction
    proj_li = proj_interp(sinogram, metal_trace)
    ct_param_fbp = ct_param.copy()
    ct_param_fbp['filter'] = 'ram-lak'
    sino_filt = filtering(proj_li, ct_param_fbp)
    img_li = np.maximum(bp(sino_filt, ct_param_fbp).astype(np.float32), 0)

    # Step 3: Generate prior image from LI result using KMeans
    img_li_for_prior = img_li.copy()
    img_li_for_prior[metal_mask_img > 0.5] = MIU_WATER

    model = KMeans(
        n_clusters=3,
        init=[[MIU_AIR], [MIU_WATER], [2 * MIU_WATER]],
        n_init=1,
    ).fit(img_li_for_prior.reshape(-1, 1))
    src_label = model.predict(img_li_for_prior.reshape(-1, 1)).reshape(img_li_for_prior.shape)

    thresh_bone = max(1.2 * MIU_WATER, np.min(img_li_for_prior[src_label == 2]))
    thresh_water = np.min(img_li_for_prior[src_label == 1])

    img_li_smooth = gaussian(img_li_for_prior, sigma=1)
    prior_img = img_li_smooth.copy()
    prior_img[img_li_smooth <= thresh_water] = MIU_AIR
    prior_img[(img_li_smooth > thresh_water) & (img_li_smooth < thresh_bone)] = MIU_WATER

    # Step 4: NMAR projection correction
    proj_prior = fp(prior_img.astype(np.float32), ct_param).astype(np.float32)
    proj_nmar = nmar_proj_interp(sinogram, proj_prior, metal_trace)

    # Step 5: Reconstruct NMAR image
    sino_filt = filtering(proj_nmar, ct_param_fbp)
    img_nmar = np.maximum(bp(sino_filt, ct_param_fbp).astype(np.float32), 0)

    return img_nmar


def nmar_from_paths(ma_sino_path, mask_img_path, anatomy):
    """
    Convenience function: run NMAR from file paths.

    Args:
        ma_sino_path: path to metalart sinogram .raw (1000x900 float32)
        mask_img_path: path to metal mask image .raw (512x512x1 float32)
        anatomy: 'body' or 'head'

    Returns:
        img_nmar: NMAR result (512, 512) in /mm attenuation
    """
    sinogram = np.fromfile(ma_sino_path, dtype=np.float32).reshape(1000, 900)
    mask_img = np.fromfile(mask_img_path, dtype=np.float32).reshape(512, 512)
    mask_img = (mask_img > 0.5).astype(np.float32)

    return nmar(sinogram, mask_img, anatomy)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--sino', type=str, required=True, help='metalart sinogram .raw')
    parser.add_argument('--mask', type=str, required=True, help='metal mask image .raw')
    parser.add_argument('--anatomy', type=str, required=True, choices=['body', 'head'])
    parser.add_argument('--output', type=str, default=None, help='output .raw path')
    args = parser.parse_args()

    result = nmar_from_paths(args.sino, args.mask, args.anatomy)

    if args.output:
        result.astype(np.float32).tofile(args.output)
        print(f"Saved: {args.output}")
    else:
        print(f"Result shape: {result.shape}, range: [{result.min():.6f}, {result.max():.6f}]")
