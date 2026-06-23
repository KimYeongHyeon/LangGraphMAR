#!/usr/bin/env python
# coding: utf-8
# 공식 구현체를 사용한 MAR 학습 (NAFNet, Restormer 등)

"""
이 스크립트는 공식 GitHub 저장소의 모델들을 사용합니다.

필수 설치:
pip install basicsr timm

모델 저장소 클론 (권장):
mkdir -p code/models/external && cd code/models/external
git clone https://github.com/megvii-research/NAFNet.git
git clone https://github.com/swz30/Restormer.git

주의사항:
1. basicsr 패치 필요 (torchvision 0.24.0 호환성)
   - 파일: .venv/lib/python3.12/site-packages/basicsr/data/degradations.py
   - Line 8: functional_tensor → functional로 변경
   
2. Python 패키지 구조 생성:
   touch code/models/external/NAFNet/basicsr/{,models/,models/archs/}__init__.py
   touch code/models/external/Restormer/basicsr/{,models/,models/archs/}__init__.py

3. HAT, SwinIR은 호환성 문제로 현재 지원하지 않음

테스트된 모델:
- ✅ NAFNet (head, body)
- ✅ peo (head, body)
"""

import os
import sys
import random
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass

# CUDA 메모리 단편화 방지 (Restormer 같은 큰 모델용)
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

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

# Custom imports
from utils.dataset import (
    CTMarDLDataset,
    IndexManager,
    load_data_list,
)
from utils.utils import create_checkpoint_dir, setup_logger, setup_early_stopping, create_model_checkpoint
from utils.models import DLbasedMARModel, create_model

# Data management
from torch.utils.data import DataLoader

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
    batch_size: int = 1  # Restormer dim=48 메모리 부족으로 1로 설정
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


# ==================== Main Training Code ====================

import argparse
parser = argparse.ArgumentParser(description='MAR Training with Official Model Implementations')
parser.add_argument('--anatomy', type=str, default='body', choices=['head', 'body'])
parser.add_argument('--model', type=str, default='nafnet', 
                    choices=['unet', 'nafnet', 'restormer'],
                    help='Model type (HAT and SwinIR removed due to compatibility issues)')

# UNet parameters
parser.add_argument('--arch', type=str, default='resnet34', 
                    choices=smp.encoders.get_encoder_names(),
                    help='UNet encoder architecture')

# NAFNet parameters
parser.add_argument('--width', type=int, default=32,
                    help='NAFNet width (default: 32, test: 16)')
parser.add_argument('--middle_blk_num', type=int, default=12,
                    help='NAFNet middle block number (default: 12, test: 6)')

# Restormer parameters
parser.add_argument('--dim', type=int, default=48,
                    help='Restormer dimension (default: 48, test: 32)')

# Multi-GPU parameters
parser.add_argument('--devices', type=int, default=1,
                    help='Number of GPUs to use (default: 1)')
parser.add_argument('--strategy', type=str, default='auto',
                    choices=['auto', 'ddp', 'ddp_spawn', 'dp'],
                    help='Training strategy for multi-GPU (default: auto, recommended: ddp)')

args = parser.parse_args()

anatomy = args.anatomy
model_type = args.model
experiment_type = 'MAR'

train_transforms = A.Compose([
    ToTensorV2(p=1, transpose_mask=True),
])
test_transforms = A.Compose([
    ToTensorV2(p=1, transpose_mask=True),
])

# 데이터 로드
project_root = code_dir.parent
dataset_path = project_root / 'dataset'
print(dataset_path)
path_dict = load_data_list(dataset_path, import_new_gt=True)[anatomy]

required_keys = ["metalart_img_path_list", "nometal_img_path_list"]
missing_keys = [key for key in required_keys if key not in path_dict]
if missing_keys:
    print(f"Error: Missing required keys: {missing_keys}")
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

train_dataloader = DataLoader(train_dataset, batch_size=conf.batch_size, shuffle=True, 
                              num_workers=conf.num_workers, pin_memory=True)
valid_dataloader = DataLoader(val_dataset, batch_size=conf.batch_size, shuffle=False, 
                             num_workers=conf.num_workers)
test_dataloader = DataLoader(test_dataset, batch_size=conf.batch_size, shuffle=False, 
                            num_workers=conf.num_workers)

# 모델 생성
model = create_model(
    model_type=model_type,
    anatomy=anatomy,
    arch=args.arch,
    width=args.width,
    middle_blk_num=args.middle_blk_num,
    dim=args.dim,
)

criterion = nn.MSELoss()
mar_model = DLbasedMARModel(model, criterion, lr=conf.learning_rate, anatomy=anatomy)

# 설정 및 구성
current_time = datetime.now().strftime('%Y-%m-%d')

if model_type == 'unet':
    model_id = args.arch
elif model_type == 'nafnet':
    model_id = f'w{args.width}_m{args.middle_blk_num}'
elif model_type == 'restormer':
    model_id = f'dim{args.dim}'
else:
    model_id = 'default'

checkpoint_name, dirpath = create_checkpoint_dir(experiment_type, anatomy, model_type, 
                                                 model_id, current_time, loss_metric='mse')
early_stop = setup_early_stopping(monitor_metric='valid_loss', patience=conf.patience, mode='min')
cbs_loss = create_model_checkpoint(dirpath, monitor_metric='valid_loss', mode='min')
wandb_logger = setup_logger(f"{anatomy}_{experiment_type}_{model_type}_{model_id}", logger_type='wandb')

trainer = pl.Trainer(
    callbacks=[cbs_loss, early_stop],
    accelerator=device,
    devices=args.devices,
    strategy=args.strategy if args.devices > 1 else 'auto',
    num_sanity_val_steps=0,
    max_epochs=conf.epoch,
    val_check_interval=conf.val_check_interval,
    log_every_n_steps=conf.log_every_n_steps,
    logger=wandb_logger,
    accumulate_grad_batches=8,  # effective batch_size = 1 * 8 = 8
)

print("\n" + "="*60)
print("Starting Training...")
print("="*60 + "\n")

trainer.fit(mar_model, train_dataloader, valid_dataloader)

print("\n" + "="*60)
print("Training Complete!")
print("="*60 + "\n")

# GPU 메모리 정리 (다음 실험을 위해)
del mar_model, trainer
torch.cuda.empty_cache()
print("✓ GPU memory cleared")


