import json

import numpy as np
import skimage.transform
from scipy.io import loadmat, savemat
from scipy.ndimage import zoom

import gecatsim as xc
from gecatsim.reconstruction.pyfiles import recon


def apply_random_values_to_metal(metal_mask, min_value=1500, max_value=10000):
    """
    Applies a continuous gradient of values ranging from 15000 to 10000 to the non-zero pixels of a metal mask,
    based on the distance from the center of non-zero pixels. The central point is calculated only using non-zero pixels.

    Parameters:
    - metal_mask (numpy.ndarray): A 2D array representing the metal mask, where non-zero pixels indicate metal regions.

    Returns:
    - numpy.ndarray: A 2D array with the same shape as `metal_mask`, where non-zero pixels have been assigned values
      that decrease from 15000 to 10000 from the calculated center to the edges in a continuous gradient.
    """
    
    # Identify non-zero pixels
    nonzero_y, nonzero_x = np.nonzero(metal_mask)
    
    # Calculate the center of non-zero pixels
    center_x = np.mean(nonzero_x)
    center_y = np.mean(nonzero_y)
    
    # Calculate distances from the center for each non-zero pixel
    distances = np.sqrt((nonzero_x - center_x)**2 + (nonzero_y - center_y)**2)
    
    # Normalize distances to a 0-1 scale and calculate gradient values
    max_distance = np.max(distances)
    normalized_distances = distances / max_distance
    gradient_values = max_value - (normalized_distances * (max_value - min_value))
    
    # Create an output array with the same shape as the input mask, initialized with zeros
    output_array = np.zeros_like(metal_mask)
    
    # Apply the calculated gradient values to the non-zero pixels in the output array
    for i, (y, x) in enumerate(zip(nonzero_y, nonzero_x)):
        output_array[y, x] = gradient_values[i]
    
    return output_array

def apply_gradient_with_options(metal_mask, min_value=1500, max_value=10000, inverse_gradient=False, add_randomness=False):
    """
    Applies a gradient (normal or inverse) of values to the non-zero pixels of a metal mask,
    with an option to add randomness to the gradient values. The gradient can spread from the center to the edges
    or vice versa, based on the inverse_gradient parameter.

    Parameters:
    - metal_mask (numpy.ndarray): A 2D array representing the metal mask, where non-zero pixels indicate metal regions.
    - inverse_gradient (bool): Determines the direction of the gradient. False for normal gradient, True for inverse.
    - add_randomness (bool): If True, applies additional randomness to the gradient values.

    Returns:
    - numpy.ndarray: A 2D array with the same shape as `metal_mask`, where non-zero pixels have been assigned values
      that follow the specified gradient, with optional randomness.
    """
    
    # Identify non-zero pixels
    nonzero_y, nonzero_x = np.nonzero(metal_mask)
    
    # Calculate the center of non-zero pixels
    center_x = np.mean(nonzero_x)
    center_y = np.mean(nonzero_y)
    
    # Calculate distances from the center for each non-zero pixel
    distances = np.sqrt((nonzero_x - center_x)**2 + (nonzero_y - center_y)**2)
    
    # Normalize distances to a 0-1 scale
    max_distance = np.max(distances)
    normalized_distances = distances / max_distance
    
    # Determine gradient direction and apply it
    if inverse_gradient:
        # Inverse gradient: larger distances result in higher values
        gradient_values = min_value + (normalized_distances * (max_value - min_value))
    else:
        # Normal gradient: larger distances result in lower values
        gradient_values = max_value - (normalized_distances * (max_value - min_value))
    
    # Apply randomness if required
    if add_randomness:
        # Applying randomness as a percentage variation of the gradient values
        random_variation = 0.1 # 10% variation
        randomness_factor = 1 + (np.random.rand(len(gradient_values)) - 0.5) * random_variation * 2
        gradient_values *= randomness_factor
    
    # Create an output array with the same shape as the input mask, initialized with zeros
    output_array = np.zeros_like(metal_mask)
    
    # Apply the calculated gradient values to the non-zero pixels in the output array
    for i, (y, x) in enumerate(zip(nonzero_y, nonzero_x)):
        output_array[y, x] = gradient_values[i]
    
    return output_array


def generate_mask_with_metal_masks(x_offset, y_offset):
    """
    generate mask with metal masks file given

    Args:
        x_offset(int), y_offset(int): offset 
    
    Returns:
        metal mask
    """
    
    metal_file = loadmat('metal_masks.mat')
    metal_file = metal_file['tumor_imgs']/255
    metal_file[metal_file>=0.1] = 1
    metal_file[metal_file<0.1] = 0

    # set random values
    # select mask
    random_metal_idx = np.random.randint(0, metal_file.shape[0])
    tgt_mtdiam = np.random.randint(5, 15) # target metal diameter, in mm

    # following parameters are random/customizable in real simulations
    # end of random parameters
    metal = metal_file[random_metal_idx]
    metaldiam = 2.*np.sqrt(metal.sum()/np.pi)
    # pix size for phantoms (excluding metal)
    ph_pixsize = 400./512 #in mm/pixel


    # metal pixel size, in mm/pixel
    mt_pixsize = tgt_mtdiam/metaldiam
    print(f"Metal Pixel Size: {mt_pixsize:.2f} mm/pixel")

    current_size = metal.shape[0]  # 가정: 금속 마스크는 정사각형이라고 가정
    target_size = mt_pixsize * current_size  # 목표 사이즈 (mm)
    resize_factor = target_size / current_size  # 리사이징 비율
    resized_metal = zoom(metal, resize_factor, order=1)  # order=1은 선형 보간 사용
    resized_metal_shape = resized_metal.shape
    applied_metal = apply_random_values_to_metal(resized_metal, 4000, 10000)
    # make always applied_metal to be resized_metal_shape size with zero padding
    applied_metal = np.pad(applied_metal, ((0, resized_metal_shape[0] - applied_metal.shape[0]), (0, resized_metal_shape[1] - applied_metal.shape[1])), mode='constant', constant_values=0)

    resized_metal = np.zeros((512, 512))
    resized_metal[y_offset:y_offset+applied_metal.shape[0], x_offset:x_offset+applied_metal.shape[1]] = applied_metal
    return resized_metal