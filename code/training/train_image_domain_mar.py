#!/usr/bin/env python
# coding: utf-8
# Image-domain MAR baseline training script (paper-standard settings)

"""
구성:
1. Charbonnier Loss (MSE 대신)
2. AdamW Optimizer (lr=2e-4, weight_decay=1e-4, betas=(0.9, 0.999))
3. CosineAnnealingLR Scheduler
4. Data Augmentation: Flip, Rotation (512x512 full image)
5. Early Stopping with valid_loss monitoring
"""

import os
import sys
import random
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass

# CUDA 메모리 단편화 방지
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# code 디렉토리를 Python path에 추가
code_dir = Path(__file__).parent.parent
sys.path.insert(0, str(code_dir))

import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import numpy as np
import argparse
import albumentations as A
from albumentations.pytorch import ToTensorV2

from utils.dataset import (
    CTMarDLDataset,
    IndexManager,
    load_data_list,
)
from utils.metric import BatchImageQualityEvaluator
from utils.utils import (
    setup_logger, 
    setup_early_stopping, 
    create_checkpoint_dir, 
    create_model_checkpoint,
)

device = "cuda" if torch.cuda.is_available() else "cpu"

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

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


# ==================== Loss Function ====================

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1 변형) - Restormer/NAFNet 논문에서 사용"""
    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps
    
    def forward(self, pred, target):
        diff = pred - target
        loss = torch.sqrt(diff * diff + self.eps * self.eps)
        return torch.mean(loss)


# ==================== Lightning Module ====================

class ImageDomainMARModel(pl.LightningModule):
    """Image-domain MAR model with paper-standard training settings."""
    
    def __init__(self, model, criterion, lr=2e-4, weight_decay=1e-4, 
                 max_epochs=50, anatomy="head"):
        super(ImageDomainMARModel, self).__init__()
        self.save_hyperparameters(ignore=['model', 'criterion'])
        self.model = model
        self.criterion = criterion
        self.anatomy = anatomy
        self.validation_outputs = []
        self.compute_batch_metrics = BatchImageQualityEvaluator()
        
    def forward(self, x):
        return self.model(x)
    
    def denormalize(self, x):
        return x * 2000 - 1000
    
    def _shared_step(self, batch, batch_idx, stage):
        images, labels, masks = batch['image'], batch['label'], batch['mask']
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        
        metrics = self.compute_batch_metrics(
            self.denormalize(labels), 
            self.denormalize(outputs), 
            masks, 
            self.anatomy
        )
        
        self.log(f'{stage}_loss', loss, on_step=True, on_epoch=True, 
                prog_bar=True, sync_dist=True)
        for metric_name, value in metrics.items():
            self.log(f'{stage}_{metric_name}', value, on_step=True, 
                    on_epoch=True, prog_bar=True, sync_dist=True)
        
        return loss
    
    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, 'train')
    
    def validation_step(self, batch, batch_idx):
        loss = self._shared_step(batch, batch_idx, 'valid')
        
        if batch_idx == 0:
            images, labels, masks = batch['image'], batch['label'], batch['mask']
            outputs = self(images)
            self.validation_outputs.append({
                'images': images[:4].detach().cpu(),
                'labels': labels[:4].detach().cpu(),
                'outputs': outputs[:4].detach().cpu(),
                'masks': masks[:4].detach().cpu()
            })
        return loss
    
    def on_validation_epoch_end(self):
        """Validation epoch 종료 시 wandb에 이미지 로깅"""
        if not self.validation_outputs:
            return
        
        sample = self.validation_outputs[-1]
        images = sample['images']
        labels = sample['labels']
        outputs = sample['outputs']
        masks = sample['masks']
        
        images_hu = self.denormalize(images)
        labels_hu = self.denormalize(labels)
        outputs_hu = self.denormalize(outputs)
        
        wandb_images = []
        for i in range(min(4, images.shape[0])):
            img = images_hu[i, 0].numpy()
            label = labels_hu[i, 0].numpy()
            output = outputs_hu[i, 0].numpy()
            mask = masks[i, 0].numpy()
            
            vmin, vmax = -1000, 1000
            img_norm = np.clip((img - vmin) / (vmax - vmin), 0, 1)
            label_norm = np.clip((label - vmin) / (vmax - vmin), 0, 1)
            output_norm = np.clip((output - vmin) / (vmax - vmin), 0, 1)
            mask_norm = mask
            
            wandb_images.append(
                wandb.Image(
                    np.hstack([img_norm, label_norm, output_norm, mask_norm]),
                    caption=f"Sample {i}: Input | GT | Pred | Mask"
                )
            )
        
        self.logger.experiment.log({
            "validation_samples": wandb_images,
            "global_step": self.global_step
        })
        
        self.validation_outputs.clear()
    
    def configure_optimizers(self):
        """Paper-standard optimizer and scheduler settings."""
        # AdamW with 논문 표준 설정
        optimizer = optim.AdamW(
            self.parameters(), 
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
            betas=(0.9, 0.999)  # 기본값 사용
        )
        
        # CosineAnnealingLR with warmup
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.max_epochs,
            eta_min=1e-6
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1
            }
        }


# ==================== Configuration ====================

@dataclass
class ImageDomainTrainingConfig:
    batch_size: int = 1  # Restormer 메모리 최적화
    num_workers: int = 8
    epoch: int = 50
    learning_rate: float = 2e-4  # NAFNet/Restormer 표준
    weight_decay: float = 1e-4
    patience: int = 10  # valid loss 진척 없으면 종료
    log_every_n_steps: int = 10
    val_check_interval: float = 1.0
    accumulate_grad_batches: int = 8  # effective batch = 1 * 8 = 8

conf = ImageDomainTrainingConfig()


# ==================== Model Loading ====================

def load_nafnet_from_github(width=32, middle_blk_num=12, img_channel=1):
    """NAFNet 공식 구현 로드"""
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
        raise


def load_restormer_from_github(inp_channels=1, out_channels=1, dim=48, 
                               num_blocks=[4,6,6,8], num_refinement_blocks=4):
    """Restormer 공식 구현 로드"""
    try:
        restormer_path = code_dir / 'models' / 'external' / 'Restormer' / 'basicsr' / 'models' / 'archs'
        if restormer_path.exists():
            sys.path.insert(0, str(code_dir / 'models' / 'external' / 'Restormer'))
            from basicsr.models.archs.restormer_arch import Restormer
            
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
            print(f"✓ Restormer loaded: dim={dim}, num_blocks={num_blocks}")
            return model
    except Exception as e:
        print(f"✗ Restormer loading failed: {e}")
        raise


def create_model(model_type, anatomy, **kwargs):
    """모델 생성 팩토리"""
    print(f"\n{'='*60}")
    print(f"Creating {model_type.upper()} model for {anatomy}")
    print(f"{'='*60}\n")
    
    if model_type == 'nafnet':
        width = kwargs.get('width', 32)
        middle_blk_num = kwargs.get('middle_blk_num', 12)
        model = load_nafnet_from_github(width=width, middle_blk_num=middle_blk_num, img_channel=1)
        
    elif model_type == 'restormer':
        dim = kwargs.get('dim', 48)
        num_blocks = kwargs.get('num_blocks', [4,6,6,8])
        model = load_restormer_from_github(
            inp_channels=1, out_channels=1, dim=dim, 
            num_blocks=num_blocks, num_refinement_blocks=4
        )
        
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    print(f"\n{'='*60}\n")
    return model


# ==================== Argument Parser ====================

parser = argparse.ArgumentParser()
parser.add_argument('--anatomy', type=str, required=True, choices=['head', 'body'])
parser.add_argument('--model', type=str, required=True, choices=['nafnet', 'restormer'])

# NAFNet parameters
parser.add_argument('--width', type=int, default=32)
parser.add_argument('--middle_blk_num', type=int, default=12)

# Restormer parameters
parser.add_argument('--dim', type=int, default=48)

args = parser.parse_args()

anatomy = args.anatomy
model_type = args.model
experiment_type = 'MAR_IMPROVED'


# ==================== Data Augmentation ====================

# 512x512 full image augmentation (Crop 제외)
train_transforms = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=90, p=0.3, border_mode=0),  # 90도 단위 회전
    ToTensorV2(p=1, transpose_mask=True),
])

test_transforms = A.Compose([
    ToTensorV2(p=1, transpose_mask=True),
])

print("\n" + "="*60)
print("Data Augmentation Settings:")
print("  - HorizontalFlip: 50%")
print("  - VerticalFlip: 50%")
print("  - Rotation (90°): 30%")
print("  - Image Size: 512×512 (No Crop)")
print("="*60 + "\n")


# ==================== Data Loading ====================

# 데이터 로드
dataset_path = Path('data')
path_dict = load_data_list(dataset_path, import_new_gt=True)[anatomy]

required_keys = ["metalart_img_path_list", "nometal_img_path_list"]
missing_keys = [key for key in required_keys if key not in path_dict]
if missing_keys:
    print(f"Error: Missing required keys: {missing_keys}")
    sys.exit(1)

train_indices, val_indices, test_indices = IndexManager('code/index', anatomy).load()
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

train_dataloader = DataLoader(
    train_dataset, 
    batch_size=conf.batch_size, 
    shuffle=True, 
    num_workers=conf.num_workers, 
    pin_memory=True
)
valid_dataloader = DataLoader(
    val_dataset, 
    batch_size=conf.batch_size, 
    shuffle=False, 
    num_workers=conf.num_workers
)
test_dataloader = DataLoader(
    test_dataset, 
    batch_size=conf.batch_size, 
    shuffle=False, 
    num_workers=conf.num_workers
)

print(f"\nDataset sizes:")
print(f"  Train: {len(train_dataset)}")
print(f"  Valid: {len(val_dataset)}")
print(f"  Test: {len(test_dataset)}")


# ==================== Model Creation ====================

model = create_model(
    model_type=model_type,
    anatomy=anatomy,
    width=args.width,
    middle_blk_num=args.middle_blk_num,
    dim=args.dim,
)

# Charbonnier Loss 사용
criterion = CharbonnierLoss(eps=1e-3)
print(f"\n✓ Using Charbonnier Loss (eps=1e-3)")

mar_model = ImageDomainMARModel(
    model, 
    criterion, 
    lr=conf.learning_rate,
    weight_decay=conf.weight_decay,
    max_epochs=conf.epoch,
    anatomy=anatomy
)


# ==================== Training Setup ====================

current_time = datetime.now().strftime('%Y-%m-%d')

if model_type == 'nafnet':
    model_id = f'w{args.width}_m{args.middle_blk_num}'
elif model_type == 'restormer':
    model_id = f'dim{args.dim}'
else:
    model_id = 'default'

checkpoint_name, dirpath = create_checkpoint_dir(
    experiment_type, anatomy, model_type, 
    model_id, current_time, loss_metric='charbonnier'
)

# Early stopping with stricter patience
early_stop = setup_early_stopping(
    monitor_metric='valid_loss', 
    patience=conf.patience, 
    mode='min'
)

cbs_loss = create_model_checkpoint(
    dirpath, 
    monitor_metric='valid_loss', 
    mode='min'
)

wandb_logger = setup_logger(
    f"{anatomy}_{experiment_type}_{model_type}_{model_id}", 
    logger_type='wandb'
)

trainer = pl.Trainer(
    callbacks=[cbs_loss, early_stop],
    accelerator=device,
    devices=1,
    num_sanity_val_steps=0,
    max_epochs=conf.epoch,
    val_check_interval=conf.val_check_interval,
    log_every_n_steps=conf.log_every_n_steps,
    logger=wandb_logger,
    accumulate_grad_batches=conf.accumulate_grad_batches,
)

print("\n" + "="*60)
print("Training Configuration:")
print(f"  Loss: Charbonnier")
print(f"  Optimizer: AdamW (lr={conf.learning_rate}, wd={conf.weight_decay})")
print(f"  Scheduler: CosineAnnealingLR")
print(f"  Batch Size: {conf.batch_size}")
print(f"  Gradient Accumulation: {conf.accumulate_grad_batches}")
print(f"  Effective Batch Size: {conf.batch_size * conf.accumulate_grad_batches}")
print(f"  Early Stopping Patience: {conf.patience}")
print("="*60 + "\n")

print("\n" + "="*60)
print("Starting Training...")
print("="*60 + "\n")

trainer.fit(mar_model, train_dataloader, valid_dataloader)

print("\n" + "="*60)
print("Training completed!")
print(f"Best model saved at: {dirpath}")
print("="*60 + "\n")

