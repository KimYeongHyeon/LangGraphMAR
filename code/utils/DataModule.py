import collections
import glob
import os
from operator import index
from pathlib import Path
from typing import Callable, Optional, Sequence, Tuple

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from albumentations.core.transforms_interface import ImageOnlyTransform
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import Subset
from hydra.utils import get_original_cwd

from utils.dataset import (
    CTMarDataset,
    load_data_list,
    load_json,
    load_raw,
    IndexManager,
    extract_subsets_by_indices
)


class MinMaxNormalize(ImageOnlyTransform):
    """
    Min-max normalization
    """

    def apply(self, img, **param):
        # minmax normalize
        # img = (img - img.min()) / (img.max() - img.min())
        img = img / 255.
        return img
    
class CTMarSinogrmaDataModule(pl.LightningDataModule):
    """
    class for loading CTMar sinogram data
    """
    def __init__(self, root_dir: str, anatomy: str, gen_random_mask: bool,
                 batch_size: int, 
                 augment: bool, num_workers: int,
                 
                 ):
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.augment = augment
        self.anatomy = anatomy
        self.gen_random_mask = gen_random_mask
        self.prepare_data_per_node = False
        self.allow_zero_length_dataloader_with_multiple_devices = False
        path_dict = load_data_list(Path(os.path.join(get_original_cwd(), self.root_dir)))

        self._train_dataset = None
        self._val_dataset = None

        train_transforms = self.get_transform(is_train=True)
        test_transforms = self.get_transform(is_train=False)

        train_indices, val_indices, test_indices = IndexManager(f'{get_original_cwd()}/index', self.anatomy).load()
        train_path_dict = extract_subsets_by_indices(path_dict, train_indices, 'train')
        valid_path_dict = extract_subsets_by_indices(path_dict, val_indices, 'valid')
        test_path_dict = extract_subsets_by_indices(path_dict, test_indices, 'test')


        self._train_dataset = CTMarDataset(train_path_dict, anatomy=self.anatomy, transform=train_transforms, task='inpainting', gen_random_mask=True)
        self._val_dataset = CTMarDataset(valid_path_dict, anatomy=self.anatomy, transform=test_transforms, task='inpainting', gen_random_mask=False)
        self._test_dataset = CTMarDataset(test_path_dict, anatomy=self.anatomy, transform=test_transforms, task='inpainting', gen_random_mask=False)
        
    def _log_hyperparams(self):
        pass

    @staticmethod
    def get_transform(is_train: bool):
        if is_train:
            transforms = A.Compose([
                ToTensorV2(),
            ])
        else:
            transforms = A.Compose([
                ToTensorV2(),
            ])
        return transforms
    @property
    def _get_path_dict(self):
        dataset_path = Path(os.path.join(os.getcwd(), self.root_dir))
        path_dict = load_data_list(dataset_path)
        return path_dict
    
    def train_dataloader(self):       
        return DataLoader(self._train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self._val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self._test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)