import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from tqdm import tqdm

from utils.ct import (
    DD2FanProj, 
    rawwrite, 
    rawread, 
    reconstruction, 
    transform_image_unit_HU_to_mm
)
from utils.utils import MinMaxScaling, grayscale_to_rgb
from utils.projection import filtering, bp, fp
from utils.logging import log_debug, should_show_progress

import albumentations as A
from albumentations.pytorch import ToTensorV2
from IPython.display import clear_output
from IPython.display import display, update_display



class IterativeReconstruction():
    def __init__(self, anatomy, FOV=None):
        """
        Initializes an instance of the class.

        Args:
            anatomy (str): The anatomy of the object being scanned.

        Returns:
            None
        """
        
        self.anatomy = anatomy[0].lower()
        assert self.anatomy in ['h', 'b'], "Anatomy must be either 'head' or 'body'."
        if self.anatomy.lower() == 'h':
            self.FOV = 220.16
        else:
            self.FOV = 400
        if FOV is not None:
            self.FOV = FOV

        self.param = {}
        self.param['nx'] = 512
        self.param['ny'] = 512
        self.param['DSD'] = 950.0  # Distance from Source to Detector
        self.param['DSO'] = 550.0  # Distance from Source to Object
        self.param['nu'] = 900     # Number of detector elements
        self.param['du'] = 1.0     # Size of each detector element
        # self.param['du'] = 400/512.     # Size of each detector element

        self.param['nview'] = 1000 # Number of views
        self.param['deg'] = np.linspace(0, 360, self.param['nview'], endpoint=False) # Projection angles
        self.param['fan_angle'] = self.param['du']/self.param['DSD']*180/np.pi*self.param['nu']  # Fan angle
        self.param['da'] = self.param['fan_angle']/self.param['nu']/180*np.pi  # Angle increment
        self.param['off_a'] = 1.25  # Detector offset angle
        self.param['dx'] = self.FOV/self.param['nx']
        self.param['dy'] = self.FOV/self.param['ny']
        self.folder = 'iterative_reconstruction'
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)
        self.prepare_normalized_image()
    def set_metal_sinogram(self, sino_m):
        if len(sino_m.shape) == 2:
            sino_m = np.expand_dims(sino_m, axis=-1)
    def prepare_normalized_image(self):
        ## Step 1: making a normalize image
        self.NorSin = fp(np.ones((self.param['nview'],self.param['nu']),dtype=np.float32), self.param)
        self.NorImg = bp(self.NorSin,self.param).astype('float32') 

    def perform_iterative_reconstruction(self, sino_m, num_iterations=10, soft_thresholding=0.0035):
        self.param['filter'] = 'ram-lak' # Originally filtered by "bone"  
        sino_filt = filtering(sino_m, self.param)
        img = bp(sino_filt, self.param).astype('float32') 
        img = np.maximum(img, 0)
        
        log_debug(f"img intensity- max:{np.max(img):.4f}, min:{np.min(img):.4f}")
        
        # verbose 레벨에 따라 progress bar 표시 여부 결정
        iterator = range(num_iterations)
        if should_show_progress():
            iterator = tqdm(iterator, desc="반복 재구성", leave=False)
        
        for ii in iterator:
            tmp_sino = fp(img, self.param).astype('float32')
            sub_sino = sino_m - tmp_sino
            sub_img1 = bp(sub_sino, self.param).astype('float32')
            sub_img1 = np.maximum(sub_img1, 0)
            img = np.maximum(img + sub_img1/self.NorImg - soft_thresholding, 0)
        return img

    def set_metal_sinogram_shape(self, sino_m):
        if len(sino_m.shape) == 2:
            sino_m = np.expand_dims(sino_m, axis=-1)

# old version
# GE + Proposed 방법으로 구성되어있음
# class IterativeReconstruction():
#     def __init__(self, anatomy, FOV=None):
#         """
#         Initializes an instance of the class.

#         Args:
#             anatomy (str): The anatomy of the object being scanned.

#         Returns:
#             None
#         """
        
#         self.anatomy = anatomy[0].lower()
#         assert self.anatomy in ['h', 'b'], "Anatomy must be either 'head' or 'body'."
#         if self.anatomy.lower() == 'h':
#             self.FOV = 220.16
#         else:
#             self.FOV = 400
#         if FOV is not None:
#             self.FOV = FOV
#         sid = 550.
#         sdd = 950.

#         self.nrdetcols = 900
#         self.nrcols = 512
#         self.nrrows = 512
#         self.pixsize = self.FOV/512
#         self.nrviews = 1000

#         self.x0 = 0.0/self.pixsize
#         self.y0 = 550.0/self.pixsize
#         self.xCor = 0.0/self.pixsize
#         self.yCor = 0.0/self.pixsize

#         dalpha = 2.*np.arctan2(1.0/2, sdd)
#         alphas = (np.arange(self.nrdetcols)-(self.nrdetcols-1)/2-1.25)*dalpha
#         self.xds = np.single(sdd*np.sin(alphas)/self.pixsize)
#         self.yds = np.single((sid - sdd*np.cos(alphas))/self.pixsize)

#         # self.viewangles = np.single(1*(0+np.arange(self.nrviews)/(self.nrviews-1)*2*np.pi))
#         self.viewangles = np.single(1*(0+np.arange(self.nrviews)/self.nrviews*2*np.pi))

#         self.param = {}
#         self.param['nx'] = 512
#         self.param['ny'] = 512
#         self.param['DSD'] = 950.0  # Distance from Source to Detector
#         self.param['DSO'] = 550.0  # Distance from Source to Object
#         self.param['nu'] = 900     # Number of detector elements
#         self.param['du'] = 1.0     # Size of each detector element
#         # self.param['du'] = 400/512.     # Size of each detector element

#         self.param['nview'] = 1000 # Number of views
#         self.param['deg'] = np.linspace(0, 360, self.param['nview'], endpoint=False) # Projection angles
#         self.param['fan_angle'] = self.param['du']/self.param['DSD']*180/np.pi*self.param['nu']  # Fan angle
#         self.param['da'] = self.param['fan_angle']/self.param['nu']/180*np.pi  # Angle increment
#         self.param['off_a'] = 1.25  # Detector offset angle
#         self.param['dx'] = self.FOV/self.param['nx']
#         self.param['dy'] = self.FOV/self.param['ny']
#         self.folder = 'iterative_reconstruction'
#         if not os.path.exists(self.folder):
#             os.makedirs(self.folder)
#         self.prepare_normalized_image()
#     def set_metal_sinogram(self, sino_m):
#         if len(sino_m.shape) == 2:
#             sino_m = np.expand_dims(sino_m, axis=-1)
#     def prepare_normalized_image(self):
#         ## Step 1: making a normalize image
#         img = np.ones([self.nrcols, self.nrrows,1], dtype = np.single)
#         tmp_sino = np.zeros([self.nrviews, self.nrdetcols, 1], dtype=np.single)     
#         tmp_sino = DD2FanProj(self.nrdetcols, self.x0, self.y0, self.xds, self.yds, self.xCor, self.yCor, self.viewangles, self.nrviews, tmp_sino, self.nrcols, self.nrrows, img)
#         tmp_sino2 = fp(img, self.param)
#         # plt.subplot(1,2,1)
#         # plt.title('GE tmp sino')
#         # plt.imshow(tmp_sino, cmap='gray')
#         # plt.subplot(1,2,2)
#         # plt.title('Proposed tmp sino')
#         # plt.imshow(tmp_sino2, cmap='gray')
#         # plt.show()
#         # print(f"tmp_sino intensity- max:{np.max(tmp_sino)}, min:{np.min(tmp_sino)}")
#         # print(f"tmp_sino2 intensity- max:{np.max(tmp_sino2)}, min:{np.min(tmp_sino2)}")
#         nor_sino = tmp_sino#*self.pixsize
#         self.nor_sino = nor_sino
#         # rawwrite(self.nor_sino_path, nor_sino)
#         nor_img = bp(nor_sino, self.param).astype('float32')
#         nor_img2 = bp(tmp_sino2, self.param).astype('float32')
#         from .ct import transform_image_unit_HU_to_mm
        
#         # plt.subplot(1,2,1)
#         # plt.title('GE nor img (bp)')
#         # plt.imshow(nor_img, cmap='gray')
#         # plt.subplot(1,2,2)
#         # plt.title('Proposed nor img (bp)')
#         # plt.imshow(nor_img2, cmap='gray')
#         # plt.show()
#         # print(f"nor_img intensity (bp) - max:{np.max(nor_img)}, min:{np.min(nor_img)}")
#         # print(f"nor_img2 intensity (bp) - max:{np.max(nor_img2)}, min:{np.min(nor_img2)}")
        
#         nor_img = transform_image_unit_HU_to_mm(nor_img)
#         nor_img2 = transform_image_unit_HU_to_mm(nor_img2)
#         # print(f"nor_img intensity (bp,mm) - max:{np.max(nor_img)}, min:{np.min(nor_img)}")
#         # print(f"nor_img2 intensity (bp,mm) - max:{np.max(nor_img2)}, min:{np.min(nor_img2)}")
        
#         nor_img = reconstruction(nor_sino, self.anatomy, kernelType='none', unit='/mm', FOV=self.FOV) # <- 이게 유효한 방법임
#         # nor_img2 = reconstruction(tmp_sino2.astype('float32'), self.anatomy, kernelType='none', unit='/mm', FOV=self.FOV)
#         # plt.subplot(1,2,1)
#         # plt.title('GE nor img (recon)')
#         # plt.imshow(nor_img, cmap='gray')
#         # plt.subplot(1,2,2)
#         # plt.title('Proposed nor img (recon)')
#         # plt.imshow(nor_img2, cmap='gray')
#         # plt.show()
#         # print(f"nor_img intensity (recon) - max:{np.max(nor_img)}, min:{np.min(nor_img)}")
#         # print(f"nor_img2 intensity (recon) - max:{np.max(nor_img2)}, min:{np.min(nor_img2)}")
#         # print(f"nor_img intensity- max:{np.max(nor_img)}, min:{np.min(nor_img)}")
#         # rawwrite(os.path.join(self.folder, f'nor_img_{self.anatomy}.raw'), nor_img)
#         self.nor_img = np.squeeze(nor_img)
        
#         self.NorSin = fp(np.ones((self.param['nview'],self.param['nu']),dtype=np.float32), self.param)
#         self.Norimg = bp(self.NorSin,self.param).astype('float32') 
#         self.Norimg2 = bp(self.nor_sino,self.param).astype('float32') 

#     def perform_iterative_reconstruction(self, sino_m, num_iterations=10, soft_thresholding=0.0035):
#         # Originally filtered by "bone" 
#         self.param['filter'] = 'ram-lak'
#         # self.param['filter'] = 'hann'
#         sino_filt = filtering(sino_m, self.param)
#         img = bp(sino_filt, self.param).astype('float32') 
#         # img = reconstruction(sino_m, self.anatomy, kernelType='bone', unit='/mm', FOV=self.FOV)
#         img = np.maximum(img, 0)
        
#         print(f"img intensity- max:{np.max(img)}, min:{np.min(img)}")
#         for ii in range(num_iterations):
#             print(f"iteration {ii+1}/{num_iterations}")
#             # tmp_sino = np.zeros([self.nrviews, self.nrdetcols, 1], dtype=np.single)     
#             # tmp_sino = DD2FanProj(self.nrdetcols, self.x0, self.y0, self.xds, self.yds, self.xCor, self.yCor, self.viewangles, self.nrviews, tmp_sino, self.nrcols, self.nrrows, img)
#             # tmp_sino = np.squeeze(tmp_sino)
#             tmp_sino = fp(img, self.param).astype('float32')
            
#             sub_sino = sino_m - tmp_sino#*self.pixsize
#             sub_img1 = bp(sub_sino, self.param).astype('float32')
#             sub_img1 = np.maximum(sub_img1, 0)
#             img = np.maximum(img + sub_img1/self.Norimg2 - soft_thresholding,0)
#             print("Intensity- max:%.2f, min:%.2f"%(np.max(img), np.min(img)))
#         return img

#     def set_metal_sinogram_shape(self, sino_m):
#         if len(sino_m.shape) == 2:
#             sino_m = np.expand_dims(sino_m, axis=-1)

# old version
# class IterativeReconstruction():
#     def __init__(self, anatomy):
#         """
#         Initializes an instance of the class.

#         Args:
#             anatomy (str): The anatomy of the object being scanned.

#         Returns:
#             None
#         """
        
#         self.anatomy = anatomy[0].lower()
#         assert self.anatomy in ['h', 'b'], "Anatomy must be either 'head' or 'body'."
#         if self.anatomy.lower() == 'h':
#             self.FOV = 220.16
#         else:
#             self.FOV = 400
#         sid = 550.
#         sdd = 950.

#         self.nrdetcols = 900
#         self.nrcols = 512
#         self.nrrows = 512
#         self.pixsize = self.FOV/512
#         self.nrviews = 1000

#         self.x0 = 0.0/self.pixsize
#         self.y0 = 550.0/self.pixsize
#         self.xCor = 0.0/self.pixsize
#         self.yCor = 0.0/self.pixsize

#         dalpha = 2.*np.arctan2(1.0/2, sdd)
#         alphas = (np.arange(self.nrdetcols)-(self.nrdetcols-1)/2-1.25)*dalpha
#         self.xds = np.single(sdd*np.sin(alphas)/self.pixsize)
#         self.yds = np.single((sid - sdd*np.cos(alphas))/self.pixsize)

#         self.viewangles = np.single(1*(0+np.arange(self.nrviews)/self.nrviews*2*np.pi))

#         self.folder = 'iterative_reconstruction'
#         if not os.path.exists(self.folder):
#             os.makedirs(self.folder)

#         self.nor_sino_path = os.path.join(self.folder, f'nor_sino_{self.anatomy}.raw')
#         self.nor_img = None
#         self.load_normalized_image()
#     def prepare_normalized_image(self):
#         ## Step 1: making a normalize image
#         img = np.ones([self.nrcols, self.nrrows,1], dtype = np.single)
#         tmp_sino = np.zeros([self.nrviews, self.nrdetcols, 1], dtype=np.single)     
#         tmp_sino = DD2FanProj(self.nrdetcols, self.x0, self.y0, self.xds, self.yds, self.xCor, self.yCor, self.viewangles, self.nrviews, tmp_sino, self.nrcols, self.nrrows, img)
#         nor_sino = tmp_sino*self.pixsize
#         rawwrite(self.nor_sino_path, nor_sino)
#         nor_img = reconstruction(nor_sino, self.anatomy, kernelType='none', unit='/mm')
#         rawwrite(os.path.join(self.folder, f'nor_img_{self.anatomy}.raw'), nor_img)

#         # inp_file = 'nor_sino.raw'
#         # if inp_file.split('.')[-1] == 'raw':
#         #     inp_data = rawread(inp_file, [1000, 1, 900], 'float')
#         #     rawwrite(inp_file.replace("raw", "prep"), inp_data)
#         # cfg = AAPMRecon_init(inp_file, FOV)
#         # cfg.recon.kernelType = 'none'
#         # AAPMRecon_main(cfg)
#         # # remove intermediate files
#         # os.remove(inp_file.replace("raw", "prep"))

#         # nor_img = rawread('nor_sino_512x512x1.raw' , [512, 512, 1], 'float')
#         print("Intensity- max:%.2f, min:%.2f"%(np.max(nor_img), np.min(nor_img)))

#     def set_metal_sinogram(self, sino_m):
#         if len(sino_m.shape) == 2:
#             sino_m = np.expand_dims(sino_m, axis=-1)
    
#     def load_normalized_image(self):
#         """고정파일, anatomy에 따라 nor_img_{anatomy}.raw 파일이 존재하지 않으면 prepare_normalized_image()를 실행한다."""
#         if not os.path.exists(os.path.join(self.folder, f'nor_img_{self.anatomy}.raw')):
#             self.prepare_normalized_image()
#         self.nor_img = rawread(os.path.join(self.folder, f'nor_img_{self.anatomy}.raw'), [512, 512, 1], 'float')
#         print("Intensity- max:%.2f, min:%.2f"%(np.max(self.nor_img), np.min(self.nor_img)))

            
#     def perform_iterative_reconstruction(self, sino_m, num_iterations=10, soft_thresholding=0.0035):
#         ## Iterative reconstruction for metal part: using compressed sensing method, soft-thresholding
#         if self.nor_img is None:
#             self.load_normalized_image()
#         if len(sino_m.shape) == 2:
#             sino_m = np.expand_dims(sino_m, axis=-1)
            
#         # img = np.ones([self.nrcols,self.nrrows,1], dtype = np.single)
#         img = reconstruction(sino_m, self.anatomy, kernelType='bone', unit='/mm')
#         img = np.maximum(img, 0)
#         print("Intensity- max:%.2f, min:%.2f"%(np.max(img), np.min(img)))
#         for ii in range(num_iterations):
#             print(f"iteration {ii+1}/{num_iterations}")
#             tmp_sino = np.zeros([self.nrviews, self.nrdetcols, 1], dtype=np.single)     
#             tmp_sino = DD2FanProj(self.nrdetcols, self.x0, self.y0, self.xds, self.yds, self.xCor, self.yCor, self.viewangles, self.nrviews, tmp_sino, self.nrcols, self.nrrows, img)
#             sub_sino = sino_m - tmp_sino*self.pixsize
#             # sub_sino_path = os.path.join(self.folder,f"sub_sino_{self.anatomy}.raw")
#             # rawwrite(sub_sino_path, sub_sino)
#             sub_img = reconstruction(sub_sino, self.anatomy, kernelType='none', unit='/mm')
#             img = np.maximum(img + sub_img/self.nor_img - soft_thresholding,0)
#             print("Intensity- max:%.2f, min:%.2f"%(np.max(img), np.min(img)))
#             # plt.imshow(img,cmap='gray', vmax = 0.1, vmin=0)
#             # plt.show()
#         return img

#     def set_metal_sinogram_shape(self, sino_m):
#         if len(sino_m.shape) == 2:
#             sino_m = np.expand_dims(sino_m, axis=-1)
            
            
def merge_masked_prediction(input, pred, mask):
    """
    Calculates the complete prediction by adding the input and predicted values.

    Args:
        input (torch.Tensor): The input tensor.
        pred (torch.Tensor): The predicted tensor.

    Returns:
        torch.Tensor: The complete prediction tensor.

    """
    
    input = input.cpu().numpy()[0][0]
    pred = pred.cpu().detach().numpy()[0][0]
    if input.shape != pred.shape:
        raise ValueError("Input and predicted tensors must have the same shape.")
    
    pred = pred * mask.squeeze().cpu().numpy()
    pred = input + pred
    return pred

transform_classifier = A.Compose([
    A.Lambda(image=grayscale_to_rgb),
    MinMaxScaling(p=1),
    # A.Normalize(mean=(0),
    #             std=(1)
    #             ),
    ToTensorV2(),
])
test_transforms = A.Compose([
    ToTensorV2(),
])
transform_enhancement = A.Compose([
    MinMaxScaling(p=1),
    ToTensorV2(),
], additional_targets={'label': 'image'})
def Inpainting(sinogram_without_metal,
            metal_mask, model):
    """
    Inpainting using deep learning model.
    
    Args:
        sinogram_without_metal(numpy.ndarray): sinogram data to inpaint.
        metal_mask(numpy.ndarray): metal mask data.
        model(torch.nn.Module): deep learning model.
        device(str): device to use. Defaults to 'cuda'.
    
    Returns:
        inpainted sinogram data
    """
    
    # check device
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    input = torch.as_tensor(sinogram_without_metal, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    mask = torch.as_tensor(metal_mask, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    ## 모델 입력 전처리 ##
    if device == 'mps':
        input = input.cpu()
        input = F.pad(input, (0, 124, 0, 24, 0, 0, 0, 0))
        input = input.to('mps')
    else:
        input = F.pad(input, (0, 124, 0, 24, 0, 0, 0, 0))

    mask = F.pad(mask, (0, 124, 0, 24))
    input = torch.stack([input, mask], dim=1)
    input = input.squeeze(2)
    ## 모델 입력 전처리 끝 ##
    with torch.no_grad():   
        pred = model(input.to(model.device))
    
    ## 모델 출력 후처리 ##
    sino_b = merge_masked_prediction(input, pred, mask)
    sino_b = sino_b[:1000, :900]
    ## 모델 출력 후처리 끝 ##
    return sino_b

def inference_classifier(image, model):
    image = transform_classifier(image=image)['image']
    image = image.to(model.device)
    image = image.unsqueeze(0)
    with torch.no_grad():   
        output = model(image)   
    # threshold = 0.5
    # predictions = (output >= threshold).float()  # Apply threshold to get 0 or 1
    return output
def inference_enhancement(image, model):
    image = image + 1000
    image = image / 2000

    image = test_transforms(image=image)['image']
    image = image.to(model.device)
    image = image.unsqueeze(0)
    with torch.no_grad():
        try:   
            enhanced, residual = model(image)   
        except:
            enhanced, residual = model(image.repeat(1, 3, 1, 1))
    # threshold = 0.5
    # predictions = (output >= threshold).float()  # Apply threshold to get 0 or 1
    enhanced = enhanced.detach().cpu().numpy()

    if enhanced.shape[1] == 3:
        enhanced = enhanced.mean(axis=1)
    enhanced = np.clip(enhanced, 0, image.max().item())

    return enhanced
