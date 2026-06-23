#!/usr/bin/env python
# coding: utf-8

# Enhancement 모델 학습



import os
import sys
import random
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass

# code 디렉토리를 Python path에 추가
code_dir = Path(__file__).parent.parent
sys.path.insert(0, str(code_dir))

# Data processing
import numpy as np

# Deep Learning
import torch
import torch.nn as nn
import pytorch_lightning as pl
import segmentation_models_pytorch as smp


# Image processing
import cv2
import PIL
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Utilities
import pytz
from tqdm import tqdm
from omegaconf import OmegaConf
from icecream import ic
from datetime import datetime

# Custom imports
from utils.dataset import CTMarEnhancementDataset
import re
import glob
from utils.dataset import (
    IndexManager,
    load_data_list,
)
from utils.utils import create_checkpoint_dir, setup_logger, setup_early_stopping, create_model_checkpoint
from utils.models import ResidualEnhancementModel, ImageEnhancement, create_model

# Data management
from torch.utils.data import DataLoader
from torchmetrics.image import StructuralSimilarityIndexMeasure
import argparse

korea_timezone = pytz.timezone('Asia/Seoul')
current_date = datetime.now(korea_timezone).strftime('%Y%m%d')
current_time = datetime.now(korea_timezone).strftime('%Y%m%d_%H%M%S')

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    pl.seed_everything(seed)

seed_everything()
print(torch.__version__, device)


import wandb
# W&B logging is optional. Log in only when a standard WANDB_API_KEY env var
# is set; otherwise training proceeds without remote experiment logging.
_wandb_api_key = os.getenv("WANDB_API_KEY")
if _wandb_api_key:
    wandb.login(key=_wandb_api_key)


@dataclass
class TrainingConfig:
    batch_size: int = 2
    num_workers: int = 4
    epoch: int = 2000
    learning_rate: float = 5e-4
    patience: int = 20
    log_every_n_steps: int = 50
    val_check_interval: float = 1.0
conf = TrainingConfig()


parser = argparse.ArgumentParser()
parser.add_argument('--anatomy', type=str, default='body', choices=['head', 'body', 'all'])
parser.add_argument('--model', type=str, default='uformer', 
                    choices=['uformer', 'unet', 'nafnet', 'restormer'],
                    help='Model architecture to use for enhancement')
parser.add_argument('--residual', action='store_true', 
                    help='Use residual learning (output = input + residual). '
                         'If not set, direct prediction (output = model(input))')
parser.add_argument('--width', type=int, default=32, help='NAFNet width parameter')
parser.add_argument('--middle_blk_num', type=int, default=12, help='NAFNet middle block number')
parser.add_argument('--dim', type=int, default=48, help='Restormer dimension parameter')
parser.add_argument('--embed_dim', type=int, default=16, help='Uformer embedding dimension')
args = parser.parse_args()

anatomy=args.anatomy
model_type=args.model
use_residual=args.residual

print(f"\n{'='*60}")
print(f"Training Mode: {'Residual Learning' if use_residual else 'Direct Prediction'}")
print(f"{'='*60}\n")
loss_metric="mae_ssim"
experiment_type = 'enhancement'


train_transforms = A.Compose([
    A.Lambda(
        image=lambda img, **kwargs: np.repeat(img, 3, axis=-1),
        mask=lambda m, **kwargs: np.repeat(m, 3, axis=-1)
    ),
    ToTensorV2(p=1),
], 
additional_targets={'label': 'mask'})

test_transforms = A.Compose([
    A.Lambda(
        image=lambda img, **kwargs: np.repeat(img, 3, axis=-1),
        mask=lambda m, **kwargs: np.repeat(m, 3, axis=-1),
    ),
    ToTensorV2(p=1),
],
additional_targets={'label': 'mask'})



dataset_path = Path('dataset')
path_dict = load_data_list(dataset_path)

# 필수 키 검증은 enhancement에서는 실제로 필요하지 않음 (다른 데이터 구조 사용)

def get_valid_path(path_dict, index_list, anatomy):
    image_path_list = {
    i: [
        path for path in sorted(glob.glob(f'results_old_sart50/{anatomy}/*/training_{anatomy}_image_b_img{index}_512x512x1.raw'))
        if float(path.split('/')[-2]) >= 0.10  # 폴더 이름이 0.10 이상인 경우만 필터링
    ]
    for i, index in enumerate(index_list)
    }
    label_path_list = [(f'dataset/{anatomy}/Target/training_{anatomy}_nometal_img{index}_512x512x1.raw') for index in index_list]
    empty_index = []
    for i, path in enumerate(image_path_list.values()):
        if len(path) == 0:
            empty_index.append(i)
    
    image_path_list = {i:path for i, path in enumerate(image_path_list.values()) if i not in empty_index}
    label_path_list = [path for i, path in enumerate(label_path_list) if i not in empty_index]
    
    image_path_list = {i:path for i, path in enumerate(image_path_list.values())} # 리인덱스
    
    return image_path_list, label_path_list


def merge_dicts_sequentially(dict1, dict2):
    max_index_dict1 = max(dict1.keys()) if dict1 else -1
    dict2_updated = {max_index_dict1 + 1 + i: v for i, (k, v) in enumerate(dict2.items())}
    return {**dict1, **dict2_updated}



head_train_path_dict = {}
head_valid_path_dict = {}
head_test_path_dict = {}
body_train_path_dict = {}
body_valid_path_dict = {}
body_test_path_dict = {}
train_path_dict = {}
valid_path_dict = {}
test_path_dict = {}

if anatomy == 'head':
    train_indices, val_indices, test_indices = IndexManager('code/index', anatomy).load()
    for key, value in path_dict[anatomy].items():
        try:
            train_path_dict[key] = value[train_indices] if len(value) > 0 else np.array([])
            valid_path_dict[key] = value[val_indices] if len(value) > 0 else np.array([])
            test_path_dict[key] = value[test_indices] if len(value) > 0 else np.array([])
        except:
            print(key)
    train_index_list = [int(re.search(r'sino(\d+)_', path).group(1)) for path in train_path_dict['metalart_sino_path_list']]
    valid_index_list = [int(re.search(r'sino(\d+)_', path).group(1)) for path in valid_path_dict['metalart_sino_path_list']]
    test_index_list = [int(re.search(r'sino(\d+)_', path).group(1)) for path in test_path_dict['metalart_sino_path_list']]



    head_train_path_dict['img_path_list'], head_train_path_dict['label'] = get_valid_path(train_path_dict, train_index_list, anatomy)
    head_valid_path_dict['img_path_list'], head_valid_path_dict['label'] = get_valid_path(valid_path_dict, valid_index_list, anatomy)
    head_test_path_dict['img_path_list'], head_test_path_dict['label'] = get_valid_path(test_path_dict, test_index_list, anatomy)


    train_path_dict = head_train_path_dict
    valid_path_dict = head_valid_path_dict
    test_path_dict = head_test_path_dict
    
elif anatomy == 'body':

    train_indices, val_indices, test_indices = IndexManager('code/index', anatomy).load()
    for key, value in path_dict[anatomy].items():
        try:
            train_path_dict[key] = value[train_indices] if len(value) > 0 else np.array([])
            valid_path_dict[key] = value[val_indices] if len(value) > 0 else np.array([])
            test_path_dict[key] = value[test_indices] if len(value) > 0 else np.array([])
        except:
            print(key)
    train_index_list = [int(re.search(r'sino(\d+)_', path).group(1)) for path in train_path_dict['metalart_sino_path_list']]
    valid_index_list = [int(re.search(r'sino(\d+)_', path).group(1)) for path in valid_path_dict['metalart_sino_path_list']]
    test_index_list = [int(re.search(r'sino(\d+)_', path).group(1)) for path in test_path_dict['metalart_sino_path_list']]

    body_train_path_dict['img_path_list'], body_train_path_dict['label'] = get_valid_path(train_path_dict, train_index_list, anatomy)
    body_valid_path_dict['img_path_list'], body_valid_path_dict['label'] = get_valid_path(valid_path_dict, valid_index_list, anatomy)
    body_test_path_dict['img_path_list'], body_test_path_dict['label'] = get_valid_path(test_path_dict, test_index_list, anatomy)

    train_path_dict = body_train_path_dict
    valid_path_dict = body_valid_path_dict
    test_path_dict = body_test_path_dict
    
elif anatomy == 'all':
    anatomy = 'head'
    train_indices, val_indices, test_indices = IndexManager('code/index', anatomy).load()
    for key, value in path_dict[anatomy].items():
        try:
            train_path_dict[key] = value[train_indices] if len(value) > 0 else np.array([])
            valid_path_dict[key] = value[val_indices] if len(value) > 0 else np.array([])
            test_path_dict[key] = value[test_indices] if len(value) > 0 else np.array([])
        except:
            print(key)
    train_index_list = [int(re.search(r'sino(\d+)_', path).group(1)) for path in train_path_dict['metalart_sino_path_list']]
    valid_index_list = [int(re.search(r'sino(\d+)_', path).group(1)) for path in valid_path_dict['metalart_sino_path_list']]
    test_index_list = [int(re.search(r'sino(\d+)_', path).group(1)) for path in test_path_dict['metalart_sino_path_list']]



    head_train_path_dict['img_path_list'], head_train_path_dict['label'] = get_valid_path(train_path_dict, train_index_list, anatomy)
    head_valid_path_dict['img_path_list'], head_valid_path_dict['label'] = get_valid_path(valid_path_dict, valid_index_list, anatomy)
    head_test_path_dict['img_path_list'], head_test_path_dict['label'] = get_valid_path(test_path_dict, test_index_list, anatomy)


    anatomy = 'body'
    train_indices, val_indices, test_indices = IndexManager('code/index', anatomy).load()
    train_path_dict = {}
    valid_path_dict = {}
    test_path_dict = {}
    for key, value in path_dict[anatomy].items():
        try:
            train_path_dict[key] = value[train_indices] if len(value) > 0 else np.array([])
            valid_path_dict[key] = value[val_indices] if len(value) > 0 else np.array([])
            test_path_dict[key] = value[test_indices] if len(value) > 0 else np.array([])
        except:
            print(key)
    train_index_list = [int(re.search(r'sino(\d+)_', path).group(1)) for path in train_path_dict['metalart_sino_path_list']]
    valid_index_list = [int(re.search(r'sino(\d+)_', path).group(1)) for path in valid_path_dict['metalart_sino_path_list']]
    test_index_list = [int(re.search(r'sino(\d+)_', path).group(1)) for path in test_path_dict['metalart_sino_path_list']]

    body_train_path_dict['img_path_list'], body_train_path_dict['label'] = get_valid_path(train_path_dict, train_index_list, anatomy)
    body_valid_path_dict['img_path_list'], body_valid_path_dict['label'] = get_valid_path(valid_path_dict, valid_index_list, anatomy)
    body_test_path_dict['img_path_list'], body_test_path_dict['label'] = get_valid_path(test_path_dict, test_index_list, anatomy)

    for key in ['img_path_list']:
        train_path_dict[key] = merge_dicts_sequentially(head_train_path_dict[key], body_train_path_dict[key])
        valid_path_dict[key] = merge_dicts_sequentially(head_valid_path_dict[key], body_valid_path_dict[key])
        test_path_dict[key] = merge_dicts_sequentially(head_test_path_dict[key], body_test_path_dict[key])

    key = 'label'
    train_path_dict[key] = head_train_path_dict[key] + body_train_path_dict[key]
    valid_path_dict[key] = head_valid_path_dict[key] + body_valid_path_dict[key]
    test_path_dict[key] = head_test_path_dict[key] + body_test_path_dict[key]


train_dataset = CTMarEnhancementDataset(train_path_dict, transform=train_transforms)
valid_dataset = CTMarEnhancementDataset(valid_path_dict, transform=test_transforms)
test_dataset = CTMarEnhancementDataset(test_path_dict, transform=test_transforms)

train_dataloader = DataLoader(train_dataset, batch_size=conf.batch_size, shuffle=True, num_workers=conf.num_workers, pin_memory=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=conf.batch_size, shuffle=False, num_workers=conf.num_workers)
test_dataloader = DataLoader(test_dataset, batch_size=conf.batch_size, shuffle=False, num_workers=conf.num_workers)


# 모델별 파라미터 설정
model_kwargs = {
    'img_size': 512,
    'in_channels': 3,
    'out_channels': 3,
}

if model_type == 'uformer':
    model_kwargs['embed_dim'] = args.embed_dim
    model_kwargs['win_size'] = 8
    model_kwargs['mlp_ratio'] = 4.
    arch = 'UFormer'
    network = f'UFormer_dim{args.embed_dim}'
elif model_type == 'unet':
    model_kwargs['arch'] = 'resnet34'
    arch = 'UNet'
    network = 'UNet_resnet34'
elif model_type == 'nafnet':
    model_kwargs['width'] = args.width
    model_kwargs['middle_blk_num'] = args.middle_blk_num
    arch = 'NAFNet'
    network = f'NAFNet_w{args.width}_m{args.middle_blk_num}'
elif model_type == 'restormer':
    model_kwargs['dim'] = args.dim
    model_kwargs['num_blocks'] = [4, 6, 6, 8]
    arch = 'Restormer'
    network = f'Restormer_dim{args.dim}'

# residual 여부를 network 이름에 추가
learning_mode = 'residual' if use_residual else 'direct'
network = f'{network}_{learning_mode}'

# 모델 생성 - create_model 함수 사용 (models.py)
backbone = create_model(
    model_type=model_type,
    anatomy='enhancement',  # enhancement는 3채널
    **model_kwargs
)

# Residual learning 여부에 따라 모델 래핑
if use_residual:
    print("✓ Using Residual Learning: output = input + model(input)")
    model = ResidualEnhancementModel(backbone)
else:
    print("✓ Using Direct Prediction: output = model(input)")
    # Direct prediction용 wrapper
    class DirectPredictionModel(nn.Module):
        """Direct prediction wrapper (no residual connection)"""
        def __init__(self, model):
            super(DirectPredictionModel, self).__init__()
            self.model = model
        
        def forward(self, x):
            output = self.model(x)
            residual = output - x  # 비교를 위한 residual 계산 (학습에는 사용 안함)
            return output, residual
    
    model = DirectPredictionModel(backbone)

lightning_model = ImageEnhancement(model, lr=conf.learning_rate, use_residual=use_residual)

current_time = datetime.now().strftime('%Y-%m-%d')
checkpoint_name, dirpath = create_checkpoint_dir(experiment_type, anatomy, network, arch, current_time, loss_metric=loss_metric)
logger = setup_logger(checkpoint_name, logger_type='wandb')  # 또는 logger_type='wandb'
early_stop = setup_early_stopping(monitor_metric='valid_loss', patience=conf.patience, mode='min')
cbs_loss = create_model_checkpoint(dirpath, monitor_metric='valid_loss', mode='min')
wandb_logger = setup_logger(f"{anatomy}_{experiment_type}_{arch}_{network}", logger_type='wandb')



# In[ ]:


trainer = pl.Trainer(callbacks=[cbs_loss, 
                                early_stop,
                                ], 
                     accelerator='gpu', 
                     devices=1,
                     num_sanity_val_steps=0,
                     max_epochs=conf.epoch, 
                     val_check_interval=conf.val_check_interval,
                     log_every_n_steps=conf.log_every_n_steps,
                     logger=wandb_logger,
                     )
trainer.fit(lightning_model, train_dataloader, valid_dataloader)

