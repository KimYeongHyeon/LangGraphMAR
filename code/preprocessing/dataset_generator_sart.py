# code for generating new dataset using SART

import numpy as np
import tqdm
import os 
import glob
import sys
from pathlib import Path

# code 디렉토리를 Python path에 추가
code_dir = Path(__file__).parent.parent
sys.path.insert(0, str(code_dir))

from utils.projection import filtering, bp, fp
from gecatsim.pyfiles.CommonTools import *
from gecatsim.pyfiles.CommonTools import rawread, rawwrite
from utils.ct import projection
from utils.ct import reconstruction
from utils.ct import transform_image_unit_mm_to_HU

data_dir = './dataset/{anatomy}*/Target_original/*sino*.raw'
param = {}
param['nx'] = 512
param['ny'] = 512
param['DSD'] = 950.0  # Distance from Source to Detector
param['DSO'] = 550.0  # Distance from Source to Object
param['nu'] = 900     # Number of detector elements
param['du'] = 1.0     # Size of each detector element
# param['du'] = 400/512.     # Size of each detector element

param['nview'] = 1000 # Number of views
param['deg'] = np.linspace(0, 360, param['nview'], endpoint=False) # Projection angles
param['fan_angle'] = param['du']/param['DSD']*180/np.pi*param['nu']  # Fan angle
param['da'] = param['fan_angle']/param['nu']/180*np.pi  # Angle increment
param['off_a'] = 1.25  # Detector offset angle

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--anatomy', type=str, default='body', help='anatomy')
parser.add_argument('--start', type=int, default=0, help='start')
parser.add_argument('--end', type=int, help='end')
parser.add_argument('--reverse', action='store_true', default=False, help='reverse')
args = parser.parse_args()



anatomy_list = [args.anatomy]
fov_dict = {
    'body': 400,
    'head': 221.6,
}
param['filter'] = 'hann'

def SART(img, sino, Norimg, param):
    import tqdm  # Import tqdm inside the functionx``
    for iter in tqdm.tqdm(range(200), desc="SART Iterations", file=sys.stderr):
        sino_diff = fp(img,param) - sino
        img_diff = bp(sino_diff,param)
        img = img - img_diff/Norimg
        img = np.maximum(img,0)
    return img

for anatomy in anatomy_list:
    file_path_list = sorted(glob.glob(data_dir.format(anatomy=anatomy)), key=lambda x: int(x.split('_')[-2].replace('sino', '')))
    if args.end is None:
        args.end = len(file_path_list)
        
    file_path_list = file_path_list[args.start:args.end]
    if args.reverse:
        file_path_list = file_path_list[::-1]
    fov = fov_dict[anatomy]
    param['dx'] = fov/param['nx']
    param['dy'] = fov/param['ny']
    Norimg = bp(fp(np.ones((param['nview'],param['nu']),dtype=np.float32), param),param)
    
    # progress_bar = tqdm.tqdm(file_path_list[:len(file_path_list)//2], desc=f"Processing {anatomy}")
    progress_bar = tqdm.tqdm(file_path_list, desc=f"Processing {anatomy}", file=sys.stderr) # file=sys.stderr 추가

    for file_path in progress_bar:
        progress_bar.set_description(f"Processing {file_path}")
        if os.path.exists(file_path.replace('Target_original', 'Target')) and file_path.replace('Target_original', 'Target').replace('sino', 'img').replace('900x1000', '512x512x1'):
            continue
        sino = rawread(file_path, [1000, 900], 'float')
        sino_filt = filtering(sino, param)
        img = bp(sino_filt, param).astype('float32')
        img = np.maximum(img,0)
        img = SART(img, sino, Norimg, param)
        img = img.astype('float32')
        
        sino = fp(img, param).astype('float32')
        
        parent_dir = os.path.dirname(file_path).replace('Target_original', 'Target')
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir, exist_ok=True)
        rawwrite(file_path.replace('Target_original', 'Target'), sino)
        rawwrite(file_path.replace('Target_original', 'Target').replace('sino', 'img').replace('900x1000', '512x512x1'), transform_image_unit_mm_to_HU(img))
    
# python preprocessing/dataset_generator_sart.py --anatomy body --start 7000 --end 8000 > log15.log 2>&1

