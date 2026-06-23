'''
Date: 2023-12-11
Make metal sinogram from metal image
'''

import importlib
import recon
importlib.reload(recon)

import os, sys
from pathlib import Path
# code 디렉토리를 Python path에 추가
code_dir = Path(__file__).parent.parent
sys.path.insert(0, str(code_dir))

import glob
import numpy as np

from utils.ct import projection
from gecatsim.pyfiles.CommonTools import *

import asyncio
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from utils.ct import get_FOV

import argparse

args = argparse.ArgumentParser()
args.add_argument('--anatomy', type=str, default='head', choices=['head', 'body'])
args = args.parse_args()

def bp(proj, param):
    img = recon.BP(proj.flatten().tolist(), param['deg'].tolist(), param['nview'], param['DSD'], param['DSO'], param['nx'], param['ny'], param['dx'], param['dy'], param['nu'], param['da'], param['off_a'])
    img =  np.reshape(img, (param['nx'], param['ny'])).astype('float32')
    return img

def fp(img, param):
    proj = recon.FP(img.flatten().tolist(), param['deg'].tolist(), param['nview'], param['DSD'], param['DSO'], param['nx'], param['ny'], param['dx'], param['dy'], param['nu'], param['da'], param['off_a'])
    proj =  np.reshape(proj, (param['nview'],param['nu'])).astype('float32')
    return proj
param = {}
param['nx'] = 512
param['ny'] = 512
param['DSD'] = 950.0
param['DSO'] = 550.0
param['nu'] = 900
param['du'] = 1.0
param['nview'] = 1000
param['deg'] = np.linspace(0, 360, param['nview'], endpoint=False)
param['fan_angle'] = param['du']/param['DSD']*180/np.pi*param['nu']
param['da'] = param['fan_angle']/param['nu']/180*np.pi
param['off_a'] = 1.25

def read(file_path):
    shape = (512, 512, 1)
    data = rawread(file_path, shape, 'float')
    return data.reshape(shape)

async def read_and_process_file(file_path, executor, anatomy, param):
    try:
        data = await asyncio.to_thread(read, file_path)
    except Exception as e:
        print(e, file_path)
    processed_data = await asyncio.get_event_loop().run_in_executor(executor, fp, 
                                                                    data, param)
    file_path = str(file_path).replace('img', 'sino').replace('512x512x1', '900x1000')
    await asyncio.to_thread(rawwrite, file_path, processed_data)

async def main():
    with ProcessPoolExecutor() as executor:
        tasks = []
        for type in [args.anatomy]:
            FOV = get_FOV(type)
            param['dx'] = FOV / param['nx']
            param['dy'] = FOV / param['ny']
            type_dir = Path('dataset') / type
            for folder in type_dir.glob('Mask'):
                for file in tqdm(folder.glob('*_metalonlymask_img*.raw')):
                    file_path = str(file).replace('img', 'sino').replace('512x512x1', '900x1000')
                    # if os.path.exists(file_path):
                    #     continue
                    anatomy = type[0]
                    task = asyncio.create_task(read_and_process_file(file, executor, anatomy=anatomy, param=param))
                    tasks.append(task)
        # for folder in Path('training_data/bbbb').glob('Mask'):
        #     for file in folder.glob('*_metalonlymask_img*.raw'):
        #         task = asyncio.create_task(read_and_process_file(file, executor, anatomy='body'))
        #         tasks.append(task)
        await asyncio.gather(*tasks)

if __name__ == '__main__':
    asyncio.run(main())

