#!/usr/bin/env python
# coding: utf-8
# UNet 기반의 모델 학습


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
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
import segmentation_models_pytorch as smp

# Image processing
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Utilities
import pytz
from icecream import ic
from datetime import datetime

# Custom imports
from utils.dataset import (
    CTMarDLDataset,
    IndexManager,
    load_data_list,
)
from utils.utils import create_checkpoint_dir, setup_logger, setup_early_stopping, create_model_checkpoint
from utils.models import DLbasedMARModel

# Data management
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

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
    batch_size: int = 8
    num_workers: int = 8
    epoch: int = 50
    learning_rate: float = 1e-3
    patience: int = 5
    log_every_n_steps: int = 10
    val_check_interval: float = 1.0

conf = TrainingConfig()


import yaml
param_yaml_path = code_dir / 'param.yaml'
with open(param_yaml_path, 'r') as f:
    param = yaml.safe_load(f)


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--anatomy', type=str, default='body', choices=['head', 'body'])
parser.add_argument('--arch', type=str, default='resnet101', choices=smp.encoders.get_encoder_names())
args = parser.parse_args()

anatomy = args.anatomy
arch = args.arch
experiment_type = 'MAR'

train_transforms = A.Compose([
    ToTensorV2(p=1, transpose_mask=True),
    ],
)
test_transforms = A.Compose([
    ToTensorV2(p=1, transpose_mask=True),
    ],
)


# 프로젝트 루트 디렉토리 설정
project_root = code_dir.parent
dataset_path = project_root / 'dataset'
path_dict = load_data_list(dataset_path, import_new_gt=True)[anatomy]

# 필수 키 검증 (데이터 분할 전)
required_keys = [
    "metalart_img_path_list",
    "nometal_img_path_list"
]

missing_keys = [key for key in required_keys if key not in path_dict]
if missing_keys:
    print(f"Error: Missing required keys in path_dict for '{anatomy}': {missing_keys}")
    print(f"Available keys: {list(path_dict.keys())}")
    print(f"Please check if the dataset is properly set up.")
    sys.exit(1)

index_path = code_dir / 'index'
train_indices, val_indices, test_indices = IndexManager(index_path, anatomy).load()
train_path_dict = {}
val_path_dict = {}
test_path_dict = {}
for key, value in path_dict.items():
    train_path_dict[key] = value[train_indices] if len(value) > 0 else np.array([])
    val_path_dict[key] = value[val_indices] if len(value) > 0 else np.array([])
    test_path_dict[key] = value[test_indices] if len(value) > 0 else np.array([])


train_dataset = CTMarDLDataset(train_path_dict, anatomy, train_transforms)
val_dataset = CTMarDLDataset(val_path_dict, anatomy, test_transforms)
test_dataset = CTMarDLDataset(test_path_dict, anatomy, test_transforms)

train_dataloader = DataLoader(train_dataset, batch_size=conf.batch_size, shuffle=True, num_workers=conf.num_workers, pin_memory=True)
valid_dataloader = DataLoader(val_dataset, batch_size=conf.batch_size, shuffle=False, num_workers=conf.num_workers)
test_dataloader = DataLoader(test_dataset, batch_size=conf.batch_size, shuffle=False, num_workers=conf.num_workers)

network = 'unet'


model = smp.Unet(
    encoder_name=arch,
    encoder_depth=5,
    encoder_weights=None,
    decoder_use_batchnorm=True,
    decoder_channels=(256, 128, 64, 32, 16),
    in_channels=1,
    classes=1,
)
criterion = nn.MSELoss()
mar_model = DLbasedMARModel(model, criterion, lr=conf.learning_rate, anatomy=anatomy)


# 설정 및 구성
current_time = datetime.now().strftime('%Y-%m-%d')
checkpoint_name, dirpath = create_checkpoint_dir(experiment_type, anatomy, network, arch, current_time, loss_metric='mse')
logger = setup_logger(checkpoint_name, logger_type='wandb')
early_stop = setup_early_stopping(monitor_metric='valid_loss', patience=conf.patience, mode='min')
cbs_loss = create_model_checkpoint(dirpath, monitor_metric='valid_loss', mode='min')
wandb_logger = setup_logger(f"{anatomy}_{experiment_type}_{arch}_{network}", logger_type='wandb')


trainer = pl.Trainer(callbacks=[cbs_loss, 
                                early_stop,
                                ], 
                     accelerator=device, 
                     devices=1,
                     num_sanity_val_steps=0,
                     max_epochs=conf.epoch, 
                     val_check_interval=conf.val_check_interval,
                     log_every_n_steps=conf.log_every_n_steps,
                     logger=wandb_logger,
                     )
trainer.fit(mar_model, train_dataloader, valid_dataloader)