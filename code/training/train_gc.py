#!/usr/bin/env python
# coding: utf-8



import os
import sys
import random
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass

# code 디렉토리를 Python path에 추가
code_dir = Path(__file__).parent.parent
sys.path.insert(0, str(code_dir))

import numpy as np
# Deep Learning
import torch
import torch.nn as nn
import pytorch_lightning as pl

# Image processing
import albumentations as A
from albumentations.pytorch import ToTensorV2

from torch.utils.data import ConcatDataset
from utils.dataset import CTMarClassificationDataset

# Utilities
import pytz
from datetime import datetime

# Custom imports
from utils.dataset import (
    IndexManager,
    load_data_list,
)
from utils.utils import MinMaxScaling
from utils.utils import create_checkpoint_dir, setup_logger, setup_early_stopping, create_model_checkpoint
from torchvision import models
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

@dataclass
class TrainingConfig:
    batch_size: int = 16
    num_workers: int = 16
    epoch: int = 100
    learning_rate: float = 1e-3
    patience: int = 5
    log_every_n_steps: int = 10
    val_check_interval: float = 1.0

conf = TrainingConfig()

anatomy = 'all'
experiment_type = 'groundchecking'
arch = 'resnet18'
network = 'unet'

dataset_path = Path('dataset')
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

train_transforms = A.Compose([
    A.Lambda(
        image=lambda img, **kwargs: np.repeat(img, 3, axis=-1),
    ),
    MinMaxScaling(p=1),
    A.HorizontalFlip(p=0.5),  
    ToTensorV2(),
])
test_transforms = A.Compose([
    A.Lambda(
        image=lambda img, **kwargs: np.repeat(img, 3, axis=-1),
    ),
    MinMaxScaling(p=1),
    ToTensorV2(),
])


dataset_path = Path('dataset')
path_dict = load_data_list(dataset_path)

# 필수 키 검증은 다음 단계에서 각 anatomy별로 수행

anatomy = 'head'

train_indices, val_indices, test_indices = IndexManager('index', anatomy).load()
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

train_path_dict['img_path_list'] = np.concatenate((train_path_dict['metalart_img_path_list'], train_path_dict['nometal_img_path_list']))
train_path_dict['label'] = np.array([1] * len(train_path_dict['metalart_img_path_list']) + [0] * len(train_path_dict['nometal_img_path_list']))
valid_path_dict['img_path_list'] = np.concatenate((valid_path_dict['metalart_img_path_list'], valid_path_dict['nometal_img_path_list']))
valid_path_dict['label'] = np.array([1] * len(valid_path_dict['metalart_img_path_list']) + [0] * len(valid_path_dict['nometal_img_path_list']))
test_path_dict['img_path_list'] = np.concatenate((test_path_dict['metalart_img_path_list'], valid_path_dict['nometal_img_path_list']))
test_path_dict['label'] = np.array([1] * len(test_path_dict['metalart_img_path_list']) + [0] * len(test_path_dict['nometal_img_path_list']))


head_train_dataset = CTMarClassificationDataset(train_path_dict, anatomy=anatomy, transform=train_transforms)
head_valid_dataset = CTMarClassificationDataset(valid_path_dict, anatomy=anatomy, transform=test_transforms)
head_test_dataset = CTMarClassificationDataset(test_path_dict, anatomy=anatomy, transform=test_transforms)
# head_train_dataset = ConcatDataset([head_train_dataset, head_test_dataset])

anatomy = 'body'

train_indices, val_indices, test_indices = IndexManager('index', anatomy).load()
train_path_dict = {}
valid_path_dict = {}
test_path_dict = {}
for key, value in path_dict[anatomy].items():
    train_path_dict[key] = value[train_indices] if len(value) > 0 else np.array([])
    valid_path_dict[key] = value[val_indices] if len(value) > 0 else np.array([])
    test_path_dict[key] = value[test_indices] if len(value) > 0 else np.array([])

train_path_dict['img_path_list'] = np.concatenate((train_path_dict['metalart_img_path_list'], train_path_dict['nometal_img_path_list']))
train_path_dict['label'] = np.array([1] * len(train_path_dict['metalart_img_path_list']) + [0] * len(train_path_dict['nometal_img_path_list']))
valid_path_dict['img_path_list'] = np.concatenate((valid_path_dict['metalart_img_path_list'], valid_path_dict['nometal_img_path_list']))
valid_path_dict['label'] = np.array([1] * len(valid_path_dict['metalart_img_path_list']) + [0] * len(valid_path_dict['nometal_img_path_list']))
test_path_dict['img_path_list'] = np.concatenate((test_path_dict['metalart_img_path_list'], test_path_dict['nometal_img_path_list']))
test_path_dict['label'] = np.array([1] * len(test_path_dict['metalart_img_path_list']) + [0] * len(test_path_dict['nometal_img_path_list']))

body_train_dataset = CTMarClassificationDataset(train_path_dict, anatomy=anatomy, transform=train_transforms)
body_valid_dataset = CTMarClassificationDataset(valid_path_dict, anatomy=anatomy, transform=test_transforms)
body_test_dataset = CTMarClassificationDataset(test_path_dict, anatomy=anatomy, transform=test_transforms)
# body_train_dataset = ConcatDataset([body_train_dataset, body_test_dataset])

train_dataset = ConcatDataset([head_train_dataset, body_train_dataset])
valid_dataset = ConcatDataset([head_valid_dataset, body_valid_dataset])
test_dataset = ConcatDataset([head_test_dataset, body_test_dataset])

train_dataloader = DataLoader(train_dataset, batch_size=conf.batch_size, shuffle=True, num_workers=conf.num_workers, pin_memory=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=conf.batch_size, shuffle=False, num_workers=conf.num_workers)
test_dataloader = DataLoader(test_dataset, batch_size=conf.batch_size, shuffle=False, num_workers=conf.num_workers)


# 설정 및 구성
anatomy = 'all'
arch = 'resnet18'
network = 'unet'



class ImageClassifier(pl.LightningModule):
    def __init__(self, model, criterion, optimizer=None, lr=1e-3, threshold=0.5):
        super(ImageClassifier, self).__init__()
        self.save_hyperparameters(ignore=['model', 'criterion'])
        self.model = model
        self.model.fc = nn.Linear(self.model.fc.in_features, 1) 
        self.criterion = criterion
        self.threshold = threshold
        self.optimizer = optimizer
        
    def forward(self, x):
        # Pass through the model and apply sigmoid for a probability-like output
        return self.model(x)
    def _shared_step(self, batch, batch_idx, stage):
        images, labels = batch['image'], batch['label']
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        probs = torch.sigmoid(outputs)
        predictions = (probs > self.threshold).float()
        acc = (predictions == labels).float().mean()
        
        self.log(f'{stage}_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f'{stage}_acc', acc, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss
    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, 'train')

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, batch_idx, 'valid')

    def test_step(self, batch, batch_idx):
        self._shared_step(batch, batch_idx, 'test')
    
    def configure_optimizers(self):
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5, 
            patience=2, 
            verbose=True
        )
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "valid_loss",
                "frequency": 1
            }
        }


# In[31]:

base_model = models.resnet18(pretrained=True) 

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(base_model.parameters(), lr=conf.learning_rate)
cls_model = ImageClassifier(base_model, criterion, optimizer=optimizer, lr=conf.learning_rate)

current_time = datetime.now().strftime('%Y-%m-%d')
checkpoint_name, dirpath = create_checkpoint_dir(experiment_type, anatomy, network, arch, current_time)
early_stop = setup_early_stopping(monitor_metric='valid_loss', patience=conf.patience, mode='min')
cbs_loss = create_model_checkpoint(dirpath, monitor_metric='valid_loss', mode='min')
wandb_logger = setup_logger(f"{anatomy}_{experiment_type}_{arch}_{network}", logger_type='wandb')

# Initialize trainer
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

# Start training
trainer.fit(cls_model, train_dataloader, valid_dataloader)

