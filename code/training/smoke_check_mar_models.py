#!/usr/bin/env python
# coding: utf-8
# MAR model smoke-check script using official implementations (NAFNet, HAT, SwinIR, Restormer)

"""
이 스크립트는 MAR 모델들의 전체 파이프라인을 빠르게 점검합니다:
- 10개 데이터로만 학습 (train: 7, val: 2, test: 1)
- 1 epoch만 실행
- 모델 저장/로드 검증
- wandb 업로드 확인
"""

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
from utils.models import DLbasedMARModel

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
# is set; otherwise the run proceeds without remote experiment logging.
_wandb_api_key = os.getenv("WANDB_API_KEY")
if _wandb_api_key:
    wandb.login(key=_wandb_api_key)

@dataclass
class TrainingConfig:
    batch_size: int = 1  # 메모리 절약을 위해 1로 감소
    num_workers: int = 2  # 적은 워커로 테스트
    epoch: int = 1  # 1 epoch만
    learning_rate: float = 1e-3
    patience: int = 5
    log_every_n_steps: int = 1  # 더 자주 로그
    val_check_interval: float = 1.0

conf = TrainingConfig()

import yaml
with open('code/param.yaml', 'r') as f:
    param = yaml.safe_load(f)


# ==================== Model Loader Functions ====================

def load_nafnet_from_github(width=32, middle_blk_num=12, img_channel=1):
    """NAFNet 공식 구현을 로드하는 함수"""
    try:
        # 방법 1: 로컬 NAFNet 저장소에서 임포트
        nafnet_path = code_dir / 'models' / 'external' / 'NAFNet' / 'basicsr' / 'models' / 'archs'
        if nafnet_path.exists():
            sys.path.insert(0, str(code_dir / 'models' / 'external' / 'NAFNet'))
            from basicsr.models.archs.NAFNet_arch import NAFNet
            print(f"✓ Loaded NAFNet from local repository: {nafnet_path}")
        else:
            # 방법 2: 설치된 basicsr에서 임포트
            from basicsr.archs.nafnet_arch import NAFNet
            print("✓ Loaded NAFNet from installed basicsr package")
        
        model = NAFNet(
            img_channel=img_channel,
            width=width,
            middle_blk_num=middle_blk_num,
            enc_blk_nums=[2, 2, 4, 8],
            dec_blk_nums=[2, 2, 2, 2]
        )
        return model
        
    except ImportError as e:
        print(f"✗ Failed to load NAFNet: {e}")
        print("\nInstallation guide:")
        print("  pip install basicsr timm")
        raise


def load_hat_from_github(img_size=512, embed_dim=180, depths=[6,6,6,6,6,6], 
                         num_heads=[6,6,6,6,6,6], window_size=16, img_channel=1):
    """HAT 공식 구현을 로드하는 함수"""
    try:
        # 로컬 HAT 저장소에서 임포트 (수정된 방식)
        hat_root = code_dir / 'models' / 'external' / 'HAT'
        if hat_root.exists():
            # HAT 루트를 sys.path에 추가
            if str(hat_root) not in sys.path:
                sys.path.insert(0, str(hat_root))
            # __init__.py를 우회하고 직접 임포트
            from hat.archs.hat_arch import HAT
            print(f"✓ Loaded HAT from local repository: {hat_root}")
        else:
            # basicsr에서 시도
            from basicsr.archs.hat_arch import HAT
            print("✓ Loaded HAT from installed basicsr package")
        
        model = HAT(
            img_size=img_size,
            in_chans=img_channel,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            compress_ratio=3,
            squeeze_factor=30,
            conv_scale=0.01,
            overlap_ratio=0.5,
            mlp_ratio=2.,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.1,
            ape=False,
            patch_norm=True,
            use_checkpoint=False,
            img_range=1.,
            upsampler='',
            resi_connection='1conv'
        )
        return model
        
    except ImportError as e:
        print(f"✗ Failed to load HAT: {e}")
        print("\nInstallation guide:")
        print("  pip install basicsr timm")
        raise


def load_swinir_from_github(img_size=512, window_size=8, img_range=1., 
                            depths=[6,6,6,6,6,6], embed_dim=180, 
                            num_heads=[6,6,6,6,6,6], img_channel=1):
    """SwinIR 공식 구현을 로드하는 함수"""
    try:
        # 로컬 SwinIR 저장소에서 임포트
        swinir_path = code_dir / 'models' / 'external' / 'SwinIR' / 'models'
        if swinir_path.exists():
            sys.path.insert(0, str(swinir_path.parent))
            from models.network_swinir import SwinIR
            print(f"✓ Loaded SwinIR from local repository: {swinir_path}")
        else:
            from basicsr.archs.swinir_arch import SwinIR
            print("✓ Loaded SwinIR from installed basicsr package")
        
        model = SwinIR(
            img_size=img_size,
            patch_size=1,
            in_chans=img_channel,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=2,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.1,
            norm_layer=nn.LayerNorm,
            ape=False,
            patch_norm=True,
            use_checkpoint=False,
            upscale=1,
            img_range=img_range,
            upsampler='',
            resi_connection='1conv'
        )
        return model
        
    except ImportError as e:
        print(f"✗ Failed to load SwinIR: {e}")
        print("\nInstallation guide:")
        print("  pip install basicsr timm")
        raise


def load_restormer_from_github(inp_channels=1, out_channels=1, dim=48, 
                               num_blocks=[4,6,6,8], num_refinement_blocks=4):
    """Restormer 공식 구현을 로드하는 함수"""
    try:
        restormer_path = code_dir / 'models' / 'external' / 'Restormer' / 'basicsr' / 'models' / 'archs'
        if restormer_path.exists():
            sys.path.insert(0, str(code_dir / 'models' / 'external' / 'Restormer'))
            from basicsr.models.archs.restormer_arch import Restormer
            print(f"✓ Loaded Restormer from local repository: {restormer_path}")
        else:
            from basicsr.archs.restormer_arch import Restormer
            print("✓ Loaded Restormer from installed basicsr package")
        
        model = Restormer(
            inp_channels=inp_channels,
            out_channels=out_channels,
            dim=dim,
            num_blocks=num_blocks,
            num_refinement_blocks=num_refinement_blocks,
            heads=[1,2,4,8],
            ffn_expansion_factor=2.66,
            bias=False,
            LayerNorm_type='WithBias',
            dual_pixel_task=False
        )
        return model
        
    except ImportError as e:
        print(f"✗ Failed to load Restormer: {e}")
        print("\nInstallation guide:")
        print("  pip install basicsr timm")
        raise


# ==================== Model Factory ====================

def create_model(model_type, anatomy, **kwargs):
    """
    공식 구현체를 사용하여 모델 생성
    
    Args:
        model_type: 'nafnet', 'hat', 'swinir', 'restormer'
        anatomy: 'head' or 'body'
        **kwargs: 모델별 추가 파라미터
    """
    print(f"\n{'='*60}")
    print(f"Creating {model_type.upper()} model for {anatomy}")
    print(f"{'='*60}\n")
    
    if model_type == 'nafnet':
        width = kwargs.get('width', 32)
        middle_blk_num = kwargs.get('middle_blk_num', 12)
        model = load_nafnet_from_github(
            width=width,
            middle_blk_num=middle_blk_num,
            img_channel=1
        )
        print(f"✓ NAFNet ready: width={width}, middle_blk_num={middle_blk_num}")
        
    elif model_type == 'hat':
        embed_dim = kwargs.get('embed_dim', 180)
        window_size = kwargs.get('window_size', 16)
        depths = kwargs.get('depths', [6,6,6,6,6,6])
        num_heads = kwargs.get('num_heads', [6,6,6,6,6,6])
        model = load_hat_from_github(
            img_size=512,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            img_channel=1
        )
        print(f"✓ HAT ready: embed_dim={embed_dim}, window_size={window_size}")
        
    elif model_type == 'swinir':
        embed_dim = kwargs.get('embed_dim', 180)
        window_size = kwargs.get('window_size', 8)
        depths = kwargs.get('depths', [6,6,6,6,6,6])
        num_heads = kwargs.get('num_heads', [6,6,6,6,6,6])
        model = load_swinir_from_github(
            img_size=512,
            window_size=window_size,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            img_channel=1
        )
        print(f"✓ SwinIR ready: embed_dim={embed_dim}, window_size={window_size}")
        
    elif model_type == 'restormer':
        dim = kwargs.get('dim', 48)
        num_blocks = kwargs.get('num_blocks', [4,6,6,8])
        model = load_restormer_from_github(
            inp_channels=1,
            out_channels=1,
            dim=dim,
            num_blocks=num_blocks,
            num_refinement_blocks=4
        )
        print(f"✓ Restormer ready: dim={dim}, num_blocks={num_blocks}")
        
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    print(f"\n{'='*60}\n")
    return model


def verify_model_checkpoint(checkpoint_path, model_type, anatomy):
    """체크포인트에서 모델을 로드하여 검증"""
    print(f"\n{'='*60}")
    print("Verifying Model Checkpoint")
    print(f"{'='*60}\n")
    
    try:
        print(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        print(f"✓ Checkpoint loaded successfully")
        
        # 체크포인트 정보 출력
        if 'epoch' in checkpoint:
            print(f"  - Epoch: {checkpoint['epoch']}")
        if 'global_step' in checkpoint:
            print(f"  - Global Step: {checkpoint['global_step']}")
        
        # state_dict 확인
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            print(f"  - Number of parameters: {len(state_dict)}")
            print(f"  - First few keys: {list(state_dict.keys())[:5]}")
            print(f"✓ State dict is valid")
        else:
            print("✗ No state_dict found in checkpoint")
            return False
        
        print(f"\n✓ Checkpoint verification PASSED for {model_type} ({anatomy})")
        return True
        
    except Exception as e:
        print(f"\n✗ Checkpoint verification FAILED: {e}")
        return False


# ==================== Main Testing Code ====================

import argparse
parser = argparse.ArgumentParser(description='MAR Model Testing Script')
parser.add_argument('--anatomy', type=str, default='body', choices=['head', 'body'])
parser.add_argument('--model', type=str, default='nafnet', 
                    choices=['nafnet', 'hat', 'swinir', 'restormer'])

# NAFNet parameters
parser.add_argument('--width', type=int, default=16)
parser.add_argument('--middle_blk_num', type=int, default=6)

# HAT/SwinIR parameters
parser.add_argument('--embed_dim', type=int, default=96)
parser.add_argument('--window_size', type=int, default=8)

# Restormer parameters
parser.add_argument('--dim', type=int, default=32)

args = parser.parse_args()

anatomy = args.anatomy
model_type = args.model
experiment_type = 'MAR_TEST'

print("\n" + "="*60)
print(f"Testing {model_type.upper()} on {anatomy.upper()} anatomy")
print("="*60 + "\n")

train_transforms = A.Compose([
    ToTensorV2(p=1, transpose_mask=True),
])
test_transforms = A.Compose([
    ToTensorV2(p=1, transpose_mask=True),
])

# 데이터 로드
dataset_path = Path('data')
path_dict = load_data_list(dataset_path, import_new_gt=True)[anatomy]

required_keys = ["metalart_img_path_list", "nometal_img_path_list"]
missing_keys = [key for key in required_keys if key not in path_dict]
if missing_keys:
    print(f"Error: Missing required keys: {missing_keys}")
    sys.exit(1)

# 전체 인덱스 로드
train_indices, val_indices, test_indices = IndexManager('code/index', anatomy).load()

# 10개로 제한 (train: 7, val: 2, test: 1)
train_indices = train_indices[:7]
val_indices = val_indices[:2]
test_indices = test_indices[:1]

print(f"\n데이터셋 크기:")
print(f"  - Train: {len(train_indices)} samples")
print(f"  - Val: {len(val_indices)} samples")
print(f"  - Test: {len(test_indices)} samples")

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
    width=args.width,
    middle_blk_num=args.middle_blk_num,
    embed_dim=args.embed_dim,
    window_size=args.window_size,
    dim=args.dim,
)

criterion = nn.MSELoss()
mar_model = DLbasedMARModel(model, criterion, lr=conf.learning_rate, anatomy=anatomy)

# 설정 및 구성
current_time = datetime.now().strftime('%Y-%m-%d')

if model_type == 'nafnet':
    model_id = f'w{args.width}_m{args.middle_blk_num}'
elif model_type in ['hat', 'swinir']:
    model_id = f'dim{args.embed_dim}_win{args.window_size}'
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
    devices=1,
    num_sanity_val_steps=0,
    max_epochs=conf.epoch,
    val_check_interval=conf.val_check_interval,
    log_every_n_steps=conf.log_every_n_steps,
    logger=wandb_logger,
)

print("\n" + "="*60)
print("Starting Training...")
print("="*60 + "\n")

try:
    trainer.fit(mar_model, train_dataloader, valid_dataloader)
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60 + "\n")
    
    # GPU 메모리 정리
    del mar_model, trainer
    torch.cuda.empty_cache()
    
    # 체크포인트 검증
    checkpoint_files = list(Path(dirpath).glob("*.ckpt"))
    if checkpoint_files:
        checkpoint_path = checkpoint_files[0]
        print(f"\nFound checkpoint: {checkpoint_path}")
        
        verification_passed = verify_model_checkpoint(checkpoint_path, model_type, anatomy)
        
        if verification_passed:
            print(f"\n{'='*60}")
            print(f"✓ TEST PASSED for {model_type.upper()} ({anatomy})")
            print(f"{'='*60}\n")
        else:
            print(f"\n{'='*60}")
            print(f"✗ TEST FAILED: Checkpoint verification failed")
            print(f"{'='*60}\n")
            sys.exit(1)
    else:
        print(f"\n✗ No checkpoint files found in {dirpath}")
        sys.exit(1)
        
except Exception as e:
    print(f"\n{'='*60}")
    print(f"✗ TEST FAILED with exception: {e}")
    print(f"{'='*60}\n")
    import traceback
    traceback.print_exc()
    sys.exit(1)

