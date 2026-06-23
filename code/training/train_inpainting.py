#!/usr/bin/env python
# coding: utf-8

import os
import sys
import random
from datetime import datetime
from pathlib import Path

# code 디렉토리를 Python path에 추가
code_dir = Path(__file__).parent.parent
sys.path.insert(0, str(code_dir))

import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pytz
from dataclasses import dataclass
from torch.utils.data import DataLoader
from utils.dataset import (
    CTMarSinogramDataset,
    IndexManager,
    load_data_list,
)
from utils.utils import create_checkpoint_dir, setup_logger, setup_early_stopping, create_model_checkpoint
from utils.models import SinogramInpainting, create_model


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

@dataclass
class TrainingConfig:
    batch_size: int = 4
    num_workers: int = 4
    epoch: int = 200
    learning_rate: float = 1e-3
    patience: int = 5
    log_every_n_steps: int = 10
    val_check_interval: float = 1.0

conf = TrainingConfig()

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--anatomy', type=str, default='body', choices=['head', 'body'])
parser.add_argument('--model_type', type=str, default='unet', 
                    choices=['unet', 'uformer', 'nafnet', 'restormer', 'swinunet'],
                    help='Model architecture to use')
parser.add_argument('--arch', type=str, default='resnet34',
                    help='Encoder architecture for UNet (e.g., resnet34, resnet101)')
# 모델별 하이퍼파라미터
parser.add_argument('--embed_dim', type=int, default=96,
                    help='Embedding dimension for transformer models (uformer, swinunet)')
parser.add_argument('--img_size', type=int, default=512,
                    help='Input image size for transformer models')
parser.add_argument('--window_size', type=int, default=8,
                    help='Window size for transformer models')
args = parser.parse_args()

anatomy = args.anatomy
model_type = args.model_type
experiment_type = 'inpainting'

dataset_path = Path('dataset')
path_dict = load_data_list(dataset_path, import_new_gt=True)[anatomy]

# 필수 키 검증 (데이터 분할 전)
required_keys = [
    "metalart_img_path_list",
    "metalart_sino_path_list", 
    "metalonlymask_img_path_list",
    "metalonlymask_sino_path_list",
    "metalinfo_path_list",
    "nometal_img_path_list",
    "nometal_sino_path_list"
]

missing_keys = [key for key in required_keys if key not in path_dict]
if missing_keys:
    print(f"Error: Missing required keys in path_dict for '{anatomy}': {missing_keys}")
    print(f"Available keys: {list(path_dict.keys())}")
    print(f"Please check if the dataset is properly set up.")
    sys.exit(1)

train_transforms = A.Compose([
    ToTensorV2(),
])
test_transforms = A.Compose([
    ToTensorV2(),
])

# IndexManager 경로 설정 (code 디렉토리 기준)
index_path = code_dir / 'index'
train_indices, val_indices, test_indices = IndexManager(str(index_path), anatomy).load()
train_path_dict = {}
val_path_dict = {}
test_path_dict = {}
for key, value in path_dict.items():
    train_path_dict[key] = value[train_indices] if len(value) > 0 else np.array([])
    val_path_dict[key] = value[val_indices] if len(value) > 0 else np.array([])
    test_path_dict[key] = value[test_indices] if len(value) > 0 else np.array([])
    
train_dataset = CTMarSinogramDataset(train_path_dict, type=anatomy, transform=train_transforms, task='inpainting', gen_random_mask=True)
valid_dataset = CTMarSinogramDataset(val_path_dict, type=anatomy, transform=test_transforms, task='inpainting', gen_random_mask=False)
test_dataset = CTMarSinogramDataset(test_path_dict, type=anatomy, transform=test_transforms, task='inpainting', gen_random_mask=False)

train_dataloader = DataLoader(train_dataset, batch_size=conf.batch_size, shuffle=True, num_workers=conf.num_workers, pin_memory=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=conf.batch_size, shuffle=False, num_workers=conf.num_workers)
test_dataloader = DataLoader(test_dataset, batch_size=conf.batch_size, shuffle=False, num_workers=conf.num_workers)

# create_model을 사용하여 모델 생성
print(f"\n{'='*70}")
print(f"Creating {model_type.upper()} model for Sinogram Inpainting")
print(f"Anatomy: {anatomy}")
print(f"{'='*70}\n")

# 모델별 설정
model_kwargs = {
    'in_channels': 2,  # Inpainting: input + mask
    'out_channels': 1,
}

if model_type == 'unet':
    model_kwargs['arch'] = args.arch
elif model_type in ['uformer', 'swinunet']:
    # Sinogram inpainting에서는 padding 후 1024x1024가 됨
    # 원본 (1000, 900) + padding (24, 124) = (1024, 1024)
    actual_img_size = 1024
    model_kwargs['img_size'] = actual_img_size
    model_kwargs['embed_dim'] = args.embed_dim
    model_kwargs['window_size'] = args.window_size
    if model_type == 'uformer':
        model_kwargs['win_size'] = args.window_size
    if model_type == 'swinunet':
        model_kwargs['depths'] = [2, 2, 6, 2]
        model_kwargs['num_heads'] = [3, 6, 12, 24]
    print(f"⚠️  Note: Using img_size={actual_img_size} for Transformer models (due to padding in SinogramInpainting)")
elif model_type == 'nafnet':
    model_kwargs['width'] = 32
    model_kwargs['middle_blk_num'] = 12
elif model_type == 'restormer':
    model_kwargs['dim'] = 48
    model_kwargs['num_blocks'] = [4, 6, 6, 8]

# 모델 생성
inpainting_base_model = create_model(
    model_type=model_type,
    anatomy=anatomy,
    **model_kwargs
)

optimizer = torch.optim.AdamW(inpainting_base_model.parameters(), lr=conf.learning_rate)
criterion = nn.MSELoss()
model = SinogramInpainting(inpainting_base_model, optimizer, criterion)

# 체크포인트 및 로거 설정
current_time = datetime.now().strftime('%Y-%m-%d')
arch_name = args.arch if model_type == 'unet' else model_type
checkpoint_name, dirpath = create_checkpoint_dir(experiment_type, anatomy, model_type, arch_name, current_time)
early_stop = setup_early_stopping(monitor_metric='valid_loss', patience=conf.patience, mode='min')
cbs_loss = create_model_checkpoint(dirpath, monitor_metric='valid_loss', mode='min')
wandb_logger = setup_logger(f"{anatomy}_{experiment_type}_{model_type}_{arch_name}", logger_type='wandb')

trainer = pl.Trainer(
    callbacks=[cbs_loss, early_stop],
    accelerator=device,
    devices=1,
    num_sanity_val_steps=0,
    max_epochs=conf.epoch,
    val_check_interval=conf.val_check_interval,
    log_every_n_steps=conf.log_every_n_steps,
    logger=wandb_logger,
)
trainer.fit(model, train_dataloader, valid_dataloader)

