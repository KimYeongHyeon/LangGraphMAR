import os
import numpy as np
import cv2
import json
import glob 
import re
from icecream import ic
from gecatsim.pyfiles.CommonTools import rawread, rawwrite

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from pathlib import Path
from utils.ct import projection
from utils.utils import pad_to_divisible
from utils.ct import transform_image_unit_HU_to_mm
# from hydra.utils import get_original_cwd

import copy
from scipy.io import loadmat
from scipy.ndimage import zoom
import random

# CT image and sinogram dimension constants
CT_IMAGE_SIZE = 512
CT_IMAGE_SHAPE = (CT_IMAGE_SIZE, CT_IMAGE_SIZE, 1)
SINOGRAM_ROWS = 900
SINOGRAM_COLS = 1000
SINOGRAM_SHAPE_STR = f'{SINOGRAM_ROWS}x{SINOGRAM_COLS}'

# HU normalization constants
HU_MIN = -1000
HU_MAX = 6000
HU_OFFSET = 1000
HU_SCALE = 2000


class IndexManager:
    """
    class for managing index of dataset
    """
    def __init__(self, folder_path, target_type):
        assert target_type in ['head', 'body']
        self.folder_path = folder_path
        self.target_type = target_type
        

    def _json_path(self, split: str) -> str:
        return os.path.join(self.folder_path, f'{split}_indices_{self.target_type}.json')

    def dump(self, train_indices, val_indices, test_indices):
        os.makedirs(self.folder_path, exist_ok=True)

        for split, indices in [('train', train_indices), ('val', val_indices), ('test', test_indices)]:
            with open(self._json_path(split), 'w') as f:
                json.dump(indices, f)

    def load(self):
        results = []
        for split in ('train', 'val', 'test'):
            json_path = self._json_path(split)
            pkl_path = json_path.replace('.json', '.pkl')

            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    results.append(json.load(f))
            elif os.path.exists(pkl_path):
                import pickle
                with open(pkl_path, 'rb') as f:
                    results.append(pickle.load(f))
            else:
                raise FileNotFoundError(f"No index file found: {json_path} or {pkl_path}")

        return tuple(results)


# Example usage
# Assuming array is your input numpy array with shape (H, W) or (H, W, C)
# array = np.random.rand(H, W) or np.random.rand(H, W, C)
# padded_array = pad_numpy_array_to_divisible_by_32(array)

# Example usage
# Assuming tensor is your input tensor with shape (C, H, W)
# tensor = torch.randn(C, H, W)
# padded_tensor = pad_tensor_to_divisible_by_32(tensor)

# def read_raw_file(file_path: str, type: str) -> np.ndarray:
#     """
#     Read a raw file with specified dimensions.

#     Args:
#         file_path (str): Path to the .raw file.
#         shape (tuple): Tuple of the shape of the data (height, width, channels).
#         type (str): Type of the data (image, sinogram).

#     Returns:
#         numpy.ndarray: Numpy array of the data.

#         assert np.array(metalart_sino == inp_data.reshape(1000,900)).all()
#     """
#     # Read the file as a binary file
#     with open(file_path, "rb") as file:
#         # Read the file content and interpret it as float32 (32 bit floating, little endianess)
#         data = np.fromfile(file, dtype=np.float32)
#     try:
#         shape = list(map(int, file_path.split("_")[-1].split(".")[0].split("x")))
#     except:
#         ic(file_path)
#         shape = (512, 512, 1)
#     print(data.shape)
#     # Reshape the data to the specified shape
#     if type == "image":
#         return data.reshape(shape)
#     elif type == "sinogram":
#         return data.reshape((1000, 900))
def load_data_list(dataset_path: Path,
                   import_new_gt: bool = False,
                   ) -> dict[str, dict]:
    """Load paths for all data files organized by anatomy type (body/head).

    Args:
        dataset_path: Path to the root dataset directory.
        import_new_gt: Whether to import new gt or not. Defaults to False.

    Returns:
        Dictionary containing organized file paths for each anatomy type.
    """
    def extract_number(path):
        # Extract number from filenames like 'sino123' or 'img456'
        match = re.search(r"(sino|img|metalinfo)(\d+)", path)
        return (match.group(1), int(match.group(2))) if match else ("", 0)

    path_dict = {}
    
    # Process each anatomy type
    for anatomy in ["body", "head"]:
        # Define base paths for different data categories
        base_paths = {
            "Baseline": ["metalart_img", "metalart_sino"],
            "Mask": ["metalonlymask_img", "metalinfo", "metalonlymask_sino"],
            "Target" if not import_new_gt else "Target": ["nometal_img", "nometal_sino"] 
        }
        
        # Initialize paths dictionary for this anatomy
        paths = {}
        
        # Load standard file paths
        for folder, prefixes in base_paths.items():
            for prefix in prefixes:
                key = f"{prefix}_path_list"
                pattern = str(dataset_path / anatomy / folder / f"training_{anatomy}_{prefix}*.raw")
                if "metalinfo" in prefix:
                    pattern = pattern.replace(".raw", ".json")
                    
                paths[key] = np.array(sorted(
                    glob.glob(pattern),
                    key=extract_number
                ))
        
        # Handle enhancement task specific paths
        # if task == 'enhancement':
        # Get prediction directories
        pred_dirs = sorted(glob.glob(str(dataset_path / anatomy / "Pred*")))
        filtered_pred_dirs = [
            d for d in pred_dirs 
            if re.search(r'Pred\d+$', Path(d).stem)
        ]
        
        # Load inpainted image paths for each prediction directory
        inpainted_paths = {}
        for pred_dir in filtered_pred_dirs:
            pattern = str(Path(pred_dir) / f"training_{anatomy}_inpainted_img*.raw")
            inpainted_paths[pred_dir] = sorted(
                glob.glob(pattern),
                key=extract_number
            )
        paths['inpainted_img_dir_list'] = inpainted_paths
    
        path_dict[anatomy] = paths

    return path_dict
def load_raw(file_path: str, type: str) -> np.ndarray:
    """
    Read a raw file with specified dimensions.

    Args:
        file_path (str): Path to the .raw file.
        shape (tuple): Tuple of the shape of the data (height, width, channels).
        type (str): Type of the data (image, sinogram).

    Returns:
        numpy.ndarray: Numpy array of the data.

        assert np.array(metalart_sino == inp_data.reshape(1000,900)).all()
    """
    shape = list(map(int, file_path.split("_")[-1].split(".")[0].split("x")))
    data = rawread(file_path, shape, 'float')

    if type == "image":
        return data.reshape(shape)
    elif type == "sinogram":
        return data.reshape(shape[::-1])
    else:
        raise ValueError(f"Unknown type '{type}': expected 'image' or 'sinogram'")
    
def load_json(file_path: str) -> dict:
    with open(file_path, "r") as f:
        return json.load(f)


# Metal mask search paths (shared across Dataset classes)
METAL_MASK_SEARCH_PATHS = [
    'metal_masks.mat',
    'code/AAPM_datachallenge/simulation_scripts/metal_masks.mat',
    os.path.join(os.path.dirname(__file__), '..', 'AAPM_datachallenge', 'simulation_scripts', 'metal_masks.mat'),
]


def load_metal_masks(search_paths=None):
    """Load and binarize metal masks from .mat file.

    Args:
        search_paths: List of paths to search. Defaults to METAL_MASK_SEARCH_PATHS.

    Returns:
        np.ndarray: Binarized metal mask array.
    """
    if search_paths is None:
        search_paths = METAL_MASK_SEARCH_PATHS

    for path in search_paths:
        if os.path.exists(path):
            metal_file = loadmat(path)
            metal_file = metal_file['tumor_imgs'] / 255
            metal_file[metal_file >= 0.1] = 1
            metal_file[metal_file < 0.1] = 0
            return metal_file

    raise FileNotFoundError(f"metal_masks.mat not found. Searched in: {search_paths}")


def _build_path_dict(path_dict: dict, idx: int) -> dict:
    """Extract file paths for a single sample index from the path dictionary.

    Args:
        path_dict: Dictionary mapping path list keys to arrays of file paths.
        idx: Sample index.

    Returns:
        Dictionary of individual file paths for the given index.
    """
    data = {
        'metalart_sinogram_path': path_dict["metalart_sino_path_list"][idx],
        'nometal_sinogram_path': path_dict["nometal_sino_path_list"][idx],
        'nometal_image_path': path_dict["nometal_img_path_list"][idx],
        'metalart_image_path': path_dict["metalart_img_path_list"][idx],
        'metalonlymask_image_path': path_dict["metalonlymask_img_path_list"][idx],
    }
    if "metalonlymask_sino_path_list" in path_dict:
        data['metalonlymask_sinogram_path'] = path_dict["metalonlymask_sino_path_list"][idx]
    return data


class CTMarSinogramDataset(Dataset):
    def __init__(self, path_dict: dict, type: str, is_train: bool = True, 
                 transform=None, task: str='mask', gen_random_mask=False):
        """        
        Args:
            path_dict (dict): path_dict from load_data_list()
            type (str): type of dataset (head, body)
            is_trian (bool): train or test. Defaults to True.
            transform (callable, optional): Optional transform to be applied on a sample.
        
        Returns:
            None
        """
        super().__init__()
        assert type in ["head", "body"]

        self.type = type
        self.transform = transform
        self.path_dict = path_dict
        
        # 필수 키 존재 여부 검증
        required_keys = [
            "metalart_img_path_list",
            "metalart_sino_path_list",
            "metalonlymask_img_path_list",
            "metalinfo_path_list",
            "nometal_img_path_list",
            "nometal_sino_path_list"
        ]
        missing_keys = [key for key in required_keys if key not in self.path_dict]
        if missing_keys:
            available_keys = list(self.path_dict.keys())
            raise KeyError(
                f"Missing required keys in path_dict: {missing_keys}\n"
                f"Available keys: {available_keys}"
            )
        
        assert len(self.path_dict["metalart_img_path_list"]) == len(self.path_dict["metalart_sino_path_list"]) == len(self.path_dict["metalonlymask_img_path_list"]) == len(self.path_dict["metalinfo_path_list"]) == len(self.path_dict["nometal_img_path_list"]) == len(self.path_dict["nometal_sino_path_list"]),\
        f'{len(self.path_dict["metalonlymask_sino_path_list"])}, {len(self.path_dict["metalart_img_path_list"])}, {len(self.path_dict["metalart_sino_path_list"])}, {len(self.path_dict["metalonlymask_img_path_list"])}, {len(self.path_dict["metalinfo_path_list"])}, {len(self.path_dict["nometal_img_path_list"])}, {len(self.path_dict["nometal_sino_path_list"])}'
        assert task in ['mask', 'inpainting', 'enhancement']
        self.is_train = is_train
        self.transform = transform
        self.task = task
        self.gen_random_mask = gen_random_mask

        self._load_metal_file()
    
    def __len__(self):
        return len(self.path_dict["metalart_img_path_list"])

    def __getitem__(self, idx):
        # as dict
        data = {}
        # fill with none
        data['metalart_image'] = None
        data['metalart_sinogram'] = None
        data['metalonlymask_image'] = None
        data['metalinfo'] = None
        data['nometal_image'] = None
        data['nometal_sinogram'] = None
        data['metalonlymask_sinogram'] = None
        
        if self.task == 'mask': # not used
            data = self.load_mask(idx)
        elif self.task == 'inpainting':
            data = self.load_inpainting(idx)
        elif self.task == 'enhancement':  # not used
            data = self.load_enhancement(idx)
        
        return data
    def _load_metal_file(self):
        self.metal_file = load_metal_masks()

    def load_mask(self, idx):
        """ mask task를 수행하기 위한 데이터 로드
        Input: sinogram with metal
        Label: sinogram mask
        """
        data = self.get_path(idx)
        
        data['metalart_sinogram'] = None
        data['metalonlymask_sinogram'] = None

        data['metalart_sinogram'] = load_raw(self.path_dict["metalart_sino_path_list"][idx], type="sinogram")
        data['metalonlymask_sinogram'] = load_raw(self.path_dict["metalonlymask_sino_path_list"][idx], type="sinogram")
        data['metalonlymask_sinogram'] = np.where(data['metalonlymask_sinogram'] != 0, 1, 0)
        
        data['input'] = data['metalart_sinogram']
        data['label'] = data['metalonlymask_sinogram']

        if self.transform:
            transformed_data = self.transform(image=data['input'], target=data['label'])
            data['input'], data['label'] = transformed_data['image'], transformed_data['target']
        
        return data
    
    def load_inpainting(self, idx):
        """ inpainting task를 수행하기 위한 데이터 로드
        Input: sinogram with metal, sinogram mask
        Label: sinogram without metal
        
        
        Args:
            idx (int): index of data
        
        """
        data = self.get_path(idx)
        data['metalart_sinogram'] = None
        data['metalonlymask_sinogram'] = None
        data['nometal_sinogram'] = None
        

        data['metalart_sinogram'] = load_raw(self.path_dict["metalart_sino_path_list"][idx], type="sinogram")
        data['nometal_sinogram'] = load_raw(self.path_dict["nometal_sino_path_list"][idx], type="sinogram")
        data['nometal_image'] = load_raw(self.path_dict["nometal_img_path_list"][idx], type="image")
        try:
            data['metalonlymask_sinogram'] = load_raw(self.path_dict["metalonlymask_sino_path_list"][idx], type="sinogram")
        except FileNotFoundError:
            traceback.print_exc()
            metalonymask_image = rawread(data['metalonlymask_image_path'], CT_IMAGE_SHAPE, 'float')
            metalonymask_sinogram = projection(metalonymask_image, self.type, True).squeeze()
            rawwrite(data['metalonlymask_image_path'].replace('img', 'sino').replace(f'{CT_IMAGE_SIZE}x{CT_IMAGE_SIZE}x1', SINOGRAM_SHAPE_STR), metalonymask_sinogram)
            # self.path_dict["metalonlymask_sino_path_list"][idx] = data['metalonlymask_image_path'].replace('img', 'sino').replace(f'{CT_IMAGE_SIZE}x{CT_IMAGE_SIZE}x1', SINOGRAM_SHAPE_STR)
            data['metalonlymask_sinogram'] = metalonymask_sinogram            
        
        
        
        if np.random.randint(1,11)>3 and self.gen_random_mask:
            # Parameters
            width, height = data['nometal_image'].shape[:2]
            num_artifacts = np.random.randint(1, 7)  # Number of metal artifacts
            min_distance = width//16   # Minimum distance between artifacts


            # Generate the image
            metalonlymask_image = generate_ct_image_with_metal_artifacts(data['nometal_image'].squeeze(), self.metal_file, num_artifacts, min_distance)
            data['metalonlymask_image'] = np.expand_dims(metalonlymask_image, axis=-1)

            metalonlymask_sinogram = projection(metalonlymask_image, self.type, is_metal_only=True)
            data['metalonlymask_sinogram'] = metalonlymask_sinogram[:, :, 0]
            data['mask'] = np.where(data['metalonlymask_sinogram'] != 0, 1, 0)
            metal_zero = np.expand_dims(data['nometal_sinogram'] * (1 - data['mask']), axis=-1)
            is_random_mask = True
        else:
            data['metalonlymask_image'] = rawread(self.path_dict["metalonlymask_img_path_list"][idx], CT_IMAGE_SHAPE, 'float')
            data['mask'] = np.where(data['metalonlymask_sinogram'] != 0, 1, 0)
            metal_zero = np.expand_dims(data['metalart_sinogram'] * (1 - data['mask']), axis=-1)
            is_random_mask = False
            
        metal_one = np.expand_dims(data['mask'], axis=-1)
        data['input'] = np.concatenate((metal_zero, metal_one), axis=-1)
        data['label'] = data['nometal_sinogram']

        if self.transform:
            transformed_data = self.transform(image=data['input'], target=data['label'])
            data['input'], data['label'] = transformed_data['image'], transformed_data['target']
            

        data['input'] = data['input'].to(torch.float32)
        data['label'] = np.expand_dims(data['label'], axis=0).astype(np.float32)
        data['mask'] = data['mask'].astype(np.float32)
        idx = extract_real_index(data['metalart_image_path'])
        if idx == -1:
            print("NO INDEX -1")
        data['meta'] = {'idx': idx, 'is_random_mask': is_random_mask}

        real_data = {}
        real_data['input'] = data['input']
        real_data['label'] = data['label']
        real_data['mask'] = data['mask']
        real_data['meta'] = data['meta']
        # make all data to torch tensor and float32 by for loop
        # for key in data.keys():
        #     if key in ['input', 'label']:
        #         data[key] = torch.from_numpy(data[key]).to(dtype=torch.float32)
        #         data[key] = pad_tensor_to_divisible_by_32(data[key])
        #     else:
        #         data[key] = torch.from_numpy(data[key]).to(dtype=torch.float32)
        return real_data
    def get_path(self, idx):
        return _build_path_dict(self.path_dict, idx)
    

class CTMarEnhancementDataset(Dataset):
    """ Enhancement 모델 학습을 위한 데이터 로드
    """
    def __init__(self, path_dict: dict, 
                 transform=None):
        super().__init__()

        self.transform = transform
        self.path_dict=  path_dict

    def __len__(self):
        return len(self.path_dict['label'])
        
    def __getitem__(self, idx):
        # select image path
        try:
            image_path = np.random.choice(self.path_dict['img_path_list'][idx])
        except (IndexError, ValueError) as e:
            raise ValueError(
                f"Failed to select image path at index {idx}: "
                f"{self.path_dict['img_path_list'][idx]}"
            ) from e
        # Load images
        image = load_raw(image_path, type="image")
        label = load_raw(self.path_dict['label'][idx], type="image")

        # Convert HU to mm^-1
        # image = transform_image_unit_HU_to_mm(image)
        # label = transform_image_unit_HU_to_mm(label)

        # Collect min/max statistics
        data_stats = {
            'image_min': image.min(),
            'image_max': image.max(),
            'label_min': label.min(),
            'label_max': label.max()
        }
        image = np.clip(image, HU_MIN, HU_MAX)
        label = np.clip(label, HU_MIN, HU_MAX)
        image += HU_OFFSET
        image /= HU_SCALE
        label += HU_OFFSET
        label /= HU_SCALE

        if self.transform:
            transformed = self.transform(image=image, label=label)
            image = transformed['image']
            label = transformed['label']
        if label.shape[-1] == 3 or label.shape[-1] == 1:
            label = label.permute(2, 0, 1)

        residual = label - image
            
        return {
            'image': image,
            'original': label,
            'label': residual,
            **data_stats  # Include min/max stats
        }    

class CTMarDLDataset(Dataset):
    """ DL-based Metal Artifact Reduction Network Dataset
    """
    def __init__(self, path_dict: dict, 
                 anatomy: str, 
                 transform=None):
        super().__init__()
        assert anatomy in ["head", "body"]

        self.anatomy = anatomy
        self.transform = transform
        self.path_dict=  path_dict

    def __len__(self):
        return len(self.path_dict['metalart_img_path_list'])
        
    def __getitem__(self, idx):
        # Load images
        image = load_raw(self.path_dict['metalart_img_path_list'][idx], type="image")
        label = load_raw(self.path_dict['nometal_img_path_list'][idx], type="image")
        mask = load_raw(self.path_dict['metalonlymask_img_path_list'][idx], type="image")

        # Collect min/max statistics
        data_stats = {
            'image_min': image.min(),
            'image_max': image.max(),
            'label_min': label.min(),
            'label_max': label.max()
        }
        # overlay mask on image
        label[mask==1] = 4000
        
        image = np.clip(image, HU_MIN, HU_MAX)
        label = np.clip(label, HU_MIN, HU_MAX)
        image += HU_OFFSET
        image /= HU_SCALE
        label += HU_OFFSET
        label /= HU_SCALE
        
        if self.transform:
            transformed = self.transform(image=image, mask1=mask, label=label)
            image = transformed['image']
            label = transformed['label']
            mask = transformed['mask1']
        
        if isinstance(label, np.ndarray):
            label = torch.from_numpy(label).to(dtype=torch.float32)
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).to(dtype=torch.float32)
        if label.shape[-1] == 1 or label.shape[-1] == 3:
            label = label.permute(2, 0, 1)
        if mask.shape[-1] == 1 or mask.shape[-1] == 3:
            mask = mask.permute(2, 0, 1)
            
        return {
            'image': image,
            'label': label,
            'mask': mask,
            **data_stats  # Include min/max stats
        }    
    
class CTMarClassificationDataset(Dataset):
    def __init__(self, path_dict: dict, anatomy: str, is_train: bool = True, 
                 transform=None):
        super().__init__()
        assert anatomy in ["head", "body"]

        self.anatomy = anatomy
        self.transform = transform

        self.path_dict=  path_dict
        # assert task in ['mask', 'inpainting', 'enhancement']
        self.is_train = is_train
        self.transform = transform

    def __len__(self):
        return len(self.path_dict['img_path_list'])
        
    def __getitem__(self, idx):
        data = {}
        data['image'] = load_raw(self.path_dict['img_path_list'][idx], type="image") 
        data['image'] = transform_image_unit_HU_to_mm(data['image'])
        data['label'] = self.path_dict['label'][idx] 
        data['label'] = np.expand_dims(data['label'], axis=-1).astype(np.float32)
        
        if self.transform:
            transformed_data = self.transform(image=data['image'])
            data['image'] = transformed_data['image']
    
        return data

class CTMarDataset(Dataset):
    def __init__(self, path_dict: dict, anatomy: str, is_train: bool = True, 
                 transform=None, task: str='mask', gen_random_mask=False):
        """        
        Args:
            path_dict (dict): path_dict from load_data_list()
            anatomy (str): type of dataset (head, body)
            is_trian (bool): train or test. Defaults to True.
            transform (callable, optional): Optional transform to be applied on a sample.
        
        Returns:
            None
        """
        super().__init__()
        assert anatomy in ["head", "body"]

        self.anatomy = anatomy
        self.transform = transform

        self.path_dict=  path_dict
            
        assert len(self.path_dict["metalart_img_path_list"]) == len(self.path_dict["metalart_sino_path_list"]) == len(self.path_dict["metalonlymask_img_path_list"]) == len(self.path_dict["metalinfo_path_list"]) == len(self.path_dict["nometal_img_path_list"]) == len(self.path_dict["nometal_sino_path_list"]),\
        f'{len(self.path_dict["metalonlymask_sino_path_list"])}, {len(self.path_dict["metalart_img_path_list"])}, {len(self.path_dict["metalart_sino_path_list"])}, {len(self.path_dict["metalonlymask_img_path_list"])}, {len(self.path_dict["metalinfo_path_list"])}, {len(self.path_dict["nometal_img_path_list"])}, {len(self.path_dict["nometal_sino_path_list"])}'
        # assert task in ['mask', 'inpainting', 'enhancement']
        self.is_train = is_train
        self.transform = transform
        self.task = task
        self.gen_random_mask = gen_random_mask

        self._load_metal_file()
    
    def __len__(self):
        return len(self.path_dict["metalart_img_path_list"])

    def __getitem__(self, idx):
        # as dict
        data = {}
        # fill with none
        data['metalart_image'] = None
        data['metalart_sinogram'] = None
        data['metalonlymask_image'] = None
        data['metalinfo'] = None
        data['nometal_image'] = None
        data['nometal_sinogram'] = None
        data['metalonlymask_sinogram'] = None
        
        if self.task == 'inpainting':
            data = self.load_inpainting(idx)
        elif self.task == 'enhancement_1':
            data = self.load_enhancement_1(idx)
        elif self.task == 'classification':
            data = self.load_classification(idx)
        return data
    
    def _load_metal_file(self):
        self.metal_file = load_metal_masks()

    def load_classification(self, idx):
        """ classification task를 수행하기 위한 데이터 로드
        """
        data = self.get_path(idx)
        data['anatomy'] = self.anatomy
        data['metalart_image'] = None
        data['metalonlymask_image'] = None
        data['nometal_image'] = None
        
        data['metalart_image'] = load_raw(self.path_dict["metalart_img_path_list"][idx], type="image")
        data['metalonlymask_image'] = load_raw(self.path_dict["metalonlymask_img_path_list"][idx], type="image")
        data['nometal_image'] = load_raw(self.path_dict["nometal_img_path_list"][idx], type="image")
        # merge two list
        data['image'] = data['metalart_image'] +data['nometal_image']
        
        data['label'] = [1] * len(data['metalart_image']) + [0] * len(data['nometal_image'])
        print(data['label'])
        
        if self.transform:
            transformed_data = self.transform(image=data['image'])
            data['image'] = transformed_data['image']

        return data
        
    def load_inpainting(self, idx):
        """ inpainting task를 수행하기 위한 데이터 로드
        Input: sinogram with metal, sinogram mask
        Label: sinogram without metal
        
        
        Args:
            idx (int): index of data
        
        """
        data = self.get_path(idx)
        data['anatomy'] = self.anatomy
        data['metalart_sinogram'] = None
        data['metalonlymask_sinogram'] = None
        data['nometal_sinogram'] = None
        

        data['metalart_sinogram'] = load_raw(self.path_dict["metalart_sino_path_list"][idx], type="sinogram")
        data['nometal_sinogram'] = load_raw(self.path_dict["nometal_sino_path_list"][idx], type="sinogram")
        data['nometal_image'] = load_raw(self.path_dict["nometal_img_path_list"][idx], type="image")
        try:
            data['metalonlymask_sinogram'] = load_raw(self.path_dict["metalonlymask_sino_path_list"][idx], type="sinogram")
        except FileNotFoundError:
            traceback.print_exc()
            metalonymask_image = rawread(data['metalonlymask_image_path'], CT_IMAGE_SHAPE, 'float')
            metalonymask_sinogram = projection(metalonymask_image, self.type, True).squeeze()
            rawwrite(data['metalonlymask_image_path'].replace('img', 'sino').replace(f'{CT_IMAGE_SIZE}x{CT_IMAGE_SIZE}x1', SINOGRAM_SHAPE_STR), metalonymask_sinogram)
            # self.path_dict["metalonlymask_sino_path_list"][idx] = data['metalonlymask_image_path'].replace('img', 'sino').replace(f'{CT_IMAGE_SIZE}x{CT_IMAGE_SIZE}x1', SINOGRAM_SHAPE_STR)
            data['metalonlymask_sinogram'] = metalonymask_sinogram            
        
        
        # 임의로 구멍내기
        if np.random.randint(1,11)>3 and self.gen_random_mask:
            # Parameters
            width, height = data['nometal_image'].shape[:2]
            num_artifacts = np.random.randint(1, 7)  # Number of metal artifacts
            min_distance = width//16   # Minimum distance between artifacts


            # Generate the image
            metalonlymask_image = generate_ct_image_with_metal_artifacts(data['nometal_image'].squeeze(), self.metal_file, num_artifacts, min_distance)
            data['metalonlymask_image'] = np.expand_dims(metalonlymask_image, axis=-1)

            metalonlymask_sinogram = projection(metalonlymask_image, self.anatomy, is_metal_only=True)
            data['metalonlymask_sinogram'] = metalonlymask_sinogram[:, :, 0]
            data['mask'] = np.where(data['metalonlymask_sinogram'] != 0, 1, 0)
            metal_zero = np.expand_dims(data['nometal_sinogram'] * (1 - data['mask']), axis=-1)
            is_random_mask = True
        else:
            data['metalonlymask_image'] = rawread(self.path_dict["metalonlymask_img_path_list"][idx], CT_IMAGE_SHAPE, 'float')
            data['mask'] = np.where(data['metalonlymask_sinogram'] != 0, 1, 0)
            metal_zero = np.expand_dims(data['metalart_sinogram'] * (1 - data['mask']), axis=-1)
            is_random_mask = False
            
        metal_one = np.expand_dims(data['mask'], axis=0)
        data['image'] = metal_zero
        data['mask'] = metal_one
        # data['input'] = np.concatenate((metal_zero, metal_one), axis=-1)
        data['label'] = data['nometal_sinogram']

        if self.transform:
            transformed_data = self.transform(image=data['image'], target=data['label'], mask=data['mask'])
            data['image'], data['label'], data['mask'] = transformed_data['image'], transformed_data['target'], transformed_data['mask']
        
        data['image'] = torch.as_tensor(data['image'], dtype=torch.float32).permute(2, 0, 1)
        data['mask'] = torch.as_tensor(data['mask'], dtype=torch.float32)
        data['label'] = torch.as_tensor(data['label'], dtype=torch.float32).unsqueeze(0)

        # data['image'] = data['image'].to(torch.float32)
        # data['mask'] = data['mask'].astype(np.float32)
        # data['label'] = np.expand_dims(data['label'], axis=0)
        
        idx = extract_real_index(data['metalart_image_path'])
        if idx == -1:
            print("NO INDEX -1")
        data['meta'] = {'idx': idx, 'is_random_mask': is_random_mask}
    
        return data
    def load_enhancement_1(self, idx):
        """ enhancement task를 수행하기 위한 데이터 로드
        Input: sinogram with metal, sinogram mask
        Label: sinogram without metal
        
        
        Args:
            idx (int): index of data
        
        """
        data = self.get_path(idx)
        data['anatomy'] = self.anatomy
        data['metalart_image'] = None
        data['inpainted_image'] = None
        data['nometal_image'] = None
        

        data['metalart_image'] = load_raw(self.path_dict["metalart_img_path_list"][idx], type="image")
        data['nometal_image'] = load_raw(self.path_dict["nometal_img_path_list"][idx], type="image")
        data['metalonlymask_image'] = load_raw(self.path_dict["metalonlymask_img_path_list"][idx], type="image")
        # data['nometal_sinogram'] = load_raw(self.path_dict["nometal_sino_path_list"][idx], type="sinogram")
        
        ###################### 랜덤으로 inpainted image를 가져옴 ###############################
        inpainted_image_dir_list = self.path_dict["inpainted_image_dir_list"]    
        random_key = random.choice(list(inpainted_image_dir_list.keys()))
        inpainted_img_path_list = inpainted_image_dir_list[random_key]
        data['inpainted_image_path'] = inpainted_img_path_list[idx]
        data['inpainted_image'] = load_raw(inpainted_img_path_list[idx], type="image")
        ####################################################################################
        
        data['image'] = data['metalart_image']
        data['mask'] = data['inpainted_image'] # general naming rule
        data['label'] = data['nometal_image']

        # if self.transform:
        #     transformed_data = self.transform(image=data['image'], target=data['label'])
        #     data['image'], data['label'] = transformed_data['image'], transformed_data['target']
        
        data['image'] = torch.as_tensor(data['image'], dtype=torch.float32).permute(2, 0, 1)
        data['mask'] = torch.as_tensor(data['mask'], dtype=torch.float32).permute(2, 0, 1)
        data['label'] = torch.as_tensor(data['label'], dtype=torch.float32).permute(2, 0, 1)

        # data['image'] = data['image'].to(torch.float32)
        # data['mask'] = data['mask'].astype(np.float32)
        # data['label'] = np.expand_dims(data['label'], axis=0)
        
        idx = extract_real_index(data['metalart_image_path'])
        if idx == -1:
            print("NO INDEX -1")
        data['meta'] = {'idx': idx}
    
        return data
    def get_path(self, idx):
        return _build_path_dict(self.path_dict, idx)

def extract_real_index(text: str) -> int:
    """Extract numeric index from filenames like 'metalart_img123_512x512x1.raw'.

    Returns:
        The extracted integer index, or -1 if no match is found.
    """
    match = re.search(r'img(\d+)', text)
    return int(match.group(1)) if match else -1
    
#### 랜덤 메탈
def generate_ct_image_with_metal_artifacts(image, metal_list, num_artifacts, min_distance):
    """
    Generates a CT image with metal artifacts.

    Args:
        image (numpy.ndarray): The original CT image.
        metal_list (list): A list of metal masks.
        num_artifacts (int): The number of metal artifacts to generate.
        min_distance (float): The minimum distance between metal artifacts.

    Returns:
        numpy.ndarray: The CT image with metal artifacts.

    """
    # Generate a blank CT image
    width, height = image.shape[:2]
    metalonly_image = np.zeros((height, width), dtype=np.float64)

    # Randomly place metal artifacts
    positions = []
    num_trial = 0
    max_trial = 1000
    for _ in range(num_artifacts):
        while True:
            x, y = np.random.randint(0, width), np.random.randint(0, height)
            too_close = False
            num_trial += 1
            if image[y, x] < 0:
                continue
            for pos in positions:
                if np.linalg.norm(np.array([x, y]) - np.array(pos)) < min_distance:
                    too_close = True
                    break
            if not too_close:
                positions.append((x, y))
                break
            if num_trial > max_trial:
                break
            
            

    # Add metal artifacts to the image
    for pos in positions:
        random_metal_idx = np.random.randint(0, len(metal_list))
        metal_mask = metal_list[random_metal_idx]
        metaldiam = 2. * np.sqrt(metal_mask.sum() / np.pi)
        tgt_mtdiam = np.random.randint(2, 15)  # target metal diameter, in mm
        mt_pixsize = tgt_mtdiam / metaldiam
        current_size = metal_mask.shape[0]  # 가정: 금속 마스크는 정사각형이라고 가정
        target_size = mt_pixsize * current_size  # 목표 사이즈 (mm)
        resize_factor = target_size / current_size  # 리사이징 비율
        resized_metal = zoom(metal_mask, resize_factor, order=1)  # order=1은 선형 보간 사용
        
        x_offset, y_offset = pos[:2]
        metal = np.zeros((CT_IMAGE_SIZE, CT_IMAGE_SIZE))
        # metal 배열 내에서 resized_metal이 위치할 수 있는 최대 x_offset과 y_offset 계산
        max_x_offset = CT_IMAGE_SIZE - resized_metal.shape[1]
        max_y_offset = CT_IMAGE_SIZE - resized_metal.shape[0]

        # 주어진 offset이 최대값을 넘지 않도록 조정
        x_offset = min(x_offset, max_x_offset)
        y_offset = min(y_offset, max_y_offset)

        # 조정된 offset을 사용하여 metal 배열에 resized_metal 할당
        metal[y_offset:y_offset + resized_metal.shape[0], x_offset:x_offset + resized_metal.shape[1]] = resized_metal
        metalonly_image += metal

    return metalonly_image


def extract_subsets_by_indices(data_paths, indices, split_type, task='inpainting'):
    """
    Extracts specific subsets of data paths based on given indices for a specified dataset split.
    
    Args:
        data_paths (dict): A dictionary containing paths categorized by anatomy, each with further categorization by dataset split.
        indices (list of int): List of indices specifying the elements to be extracted from each path list.
        split_type (str): The type of data split ('train', 'valid', 'test') to be processed.

    Returns:
        dict: A dictionary containing the filtered paths, structured similarly to the input dictionary but containing only paths at the specified indices.
        
    Raises:
        AssertionError: If the `split_type` is not one of 'train', 'valid', 'test'.
    """
    split_type = split_type.lower()
    assert split_type in ['train', 'valid', 'test'], "split_type must be 'train', 'valid', or 'test'"
    subset_paths = {}
    if task == 'enhancement':
        subset_paths = {}
        for category, paths in data_paths.items():
            if category == "inpainted_image_dir_list":
                break
            subset_paths[category] = [paths[i] for i in indices]
        subset_paths['inpainted_image_dir_list'] = {}
        for inpainted_image_dir, inpainted_image_paths in data_paths['inpainted_image_dir_list'].items():
            basename = os.path.basename(inpainted_image_dir)
            subset_paths['inpainted_image_dir_list'][inpainted_image_dir] = [path.replace('Baseline', basename).replace('metalart_img', 'inpainted_img') for path in subset_paths['metalart_img_path_list']]
        # for inpainted_image_dir, inpainted_image_paths in data_paths['inpainted_image_dir_list'].items():
        #     template = inpainted_image_paths[0]
        #     subset_paths['inpainted_image_dir_list'][inpainted_image_dir] = [template.replace('img1', f"img{i+1}") for i in indices]
    else:
        subset_paths = {}
        for category, paths in data_paths.items():
            if category == "inpainted_img_dir_list":
                continue
            subset_paths[category] = [paths[i] for i in indices]
        
    return subset_paths
def extract_subsets_by_indices_old(data_paths, indices, split_type, task='enhancement'):
    """
    Extracts specific subsets of data paths based on given indices for a specified dataset split.
    
    Args:
        data_paths (dict): A dictionary containing paths categorized by anatomy, each with further categorization by dataset split.
        indices (list of int): List of indices specifying the elements to be extracted from each path list.
        split_type (str): The type of data split ('train', 'valid', 'test') to be processed.

    Returns:
        dict: A dictionary containing the filtered paths, structured similarly to the input dictionary but containing only paths at the specified indices.
        
    Raises:
        AssertionError: If the `split_type` is not one of 'train', 'valid', 'test'.
    """
    split_type = split_type.lower()
    assert split_type in ['train', 'valid', 'test'], "split_type must be 'train', 'valid', or 'test'"
    subset_paths = {}
    if task == 'enhancement':
        for anatomy in ['body', 'head']:
            subset_paths[anatomy] = {}
            for category, paths in data_paths[anatomy].items():
                if category == "inpainted_image_dir_list":
                    break
                subset_paths[anatomy][category] = [paths[i] for i in indices]
        subset_paths['inpainted_image_dir_list'] = {}
        for inpainted_image_dir, inpainted_image_paths in data_paths[anatomy]['inpainted_image_dir_list'].items():
            template = inpainted_image_paths[0]
            subset_paths['inpainted_image_dir_list'][inpainted_image_dir] = [template.replace('img1', f"img{i+1}") for i in indices]
    else:
        subset_paths = {}
        for anatomy in ['body', 'head']:
            subset_paths[anatomy] = {}
            for category, paths in data_paths[anatomy].items():
                subset_paths[anatomy][category] = [paths[i] for i in indices]
        
    return subset_paths

