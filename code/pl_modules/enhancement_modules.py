import numpy as np

import pytorch_lightning as pl
from torchmetrics.image import StructuralSimilarityIndexMeasure
from typing import Any
from pytorch_lightning.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from skimage.metrics import structural_similarity as SKssim
from skimage.metrics import peak_signal_noise_ratio as SKpsnr

import torch
import torch.nn.functional as F
from utils.metric import ImageQualityEvaluator
from loss.losses import *
from utils.ct import transform_image_unit_HU_to_mm, transform_image_unit_mm_to_HU
def init_loss():
    L1_loss= L1Loss(reduction='mean')#.cuda()
    D_loss = SSIM()#.cuda()
    E_loss = EdgeLoss()#.cuda()
    P_loss = PerceptualLoss({'conv1_2': 1, 'conv2_2': 1,'conv3_4': 1,'conv4_4': 1}, perceptual_weight = 1 ,criterion='mse')#.cuda()
    return L1_loss,P_loss,E_loss,D_loss

# 나중에 이름 변경 필요
class Pl_EnhacnementModel_Unet_in_HU(pl.LightningModule):
    def __init__(self, model, optimizer, criterion, **kwargs):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.ssim = StructuralSimilarityIndexMeasure(data_range=None)
        L1_loss,P_loss,E_loss,D_loss = init_loss()
        self.L1_loss = L1_loss
        self.P_loss = P_loss
        self.E_loss = E_loss
        self.D_loss = D_loss

        # self.save_hyperparameters() # multi gpu에서 에러발생 일으킴
    def forward(self, x):
        return self.model(x)
    def preprocessing_data(self, batch):
        """Apply padding and prepare masks."""
        input = F.pad(batch['image'], (0, 124, 0, 24))
        label = F.pad(batch['label'], (0, 124, 0, 24))
        mask = F.pad(batch['mask'], (0, 124, 0, 24))
        
        label = label * mask
        return input, label, mask
    def shared_step(self, batch, stage):
        input, label, mask = batch['image'], batch['label'], batch['mask']
        # input, label, mask = self.preprocessing_data(batch=batch)
        input = torch.concat([input, mask], dim=1)
        pred = self.forward(input)
        
        # a = 1.
        # loss = a*self.criterion(pred.squeeze(), label.squeeze()) + min_std(pred, label)*(1-a) 
        # loss = self.criterion(pred.squeeze(dim=1)[mask], label.squeeze(dim=1)[mask])

        # if criterion is SSIM:
        # loss = self.criterion(pred, label)
        # if criterion is MSE:
        # loss = self.criterion(pred, label)
        
        loss = self.L1_loss(pred, label) + self.D_loss(pred, label) + self.E_loss(pred, label) + self.P_loss(pred, label)[0]
        ssim_value = self.ssim(pred, label)
        self.log_metrics(stage, loss, ssim_value)
        return {'loss': loss, 'ssim': ssim_value}
        
    def log_metrics(self, stage, loss, ssim_value):
        """Log metrics for training and validation stages."""
        self.log(f"{stage}_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log(f"{stage}_ssim", ssim_value, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

    def on_train_start(self):
        self.ssim.to(self.device)

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        return self.shared_step(batch, stage='train')
    
    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        return self.shared_step(batch, stage='valid')
    
    # def on_epoch_end(self) -> None:
    #     if torch.cuda.is_available():
    #         torch.cuda.empty_cache()
    def configure_optimizers(self) -> Any:
        # return {
        #     'optimizer': self.optimizer,
        #     'lr_scheduler': {
        #         'scheduler': torch.optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=1e-5, step_size_up=500,
        #                                                        max_lr=1e-3, gamma=0.5, mode='exp_range'),
        #         'interval': 'epoch',
        #         'frequency': 1,
        #         "monitor": "valid_loss",
        #         'strict': True,
        #         # "cycle_momentum": False
        #     }
        # }
        return self.optimizer
class Pl_EnhacnementModel_3CH(pl.LightningModule):
    def __init__(self, model, optimizer, criterion, **kwargs):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.ssim = StructuralSimilarityIndexMeasure(data_range=None)
        L1_loss,P_loss,E_loss,D_loss = init_loss()
        self.L1_loss = L1_loss
        self.P_loss = P_loss
        self.E_loss = E_loss
        self.D_loss = D_loss

        # self.save_hyperparameters() # multi gpu에서 에러발생 일으킴
    def forward(self, x):
        return self.model(x)
    def preprocessing_data(self, batch):
        """Apply padding and prepare masks."""
        input = F.pad(batch['image'], (0, 124, 0, 24))
        label = F.pad(batch['label'], (0, 124, 0, 24))
        mask = F.pad(batch['mask'], (0, 124, 0, 24))
        
        label = label * mask
        return input, label, mask
    def shared_step(self, batch, stage):
        input, label, mask = batch['image'], batch['label'], batch['mask']
        # input, label, mask = self.preprocessing_data(batch=batch)
        label = transform_image_unit_HU_to_mm(label)
        input = transform_image_unit_HU_to_mm(input)
        mask = transform_image_unit_HU_to_mm(mask)
        non_channel = torch.zeros_like(input)
        input = torch.concat([input, mask, non_channel], dim=1)
        pred = self.forward(input)
        
        # a = 1.
        # loss = a*self.criterion(pred.squeeze(), label.squeeze()) + min_std(pred, label)*(1-a) 
        # loss = self.criterion(pred.squeeze(dim=1)[mask], label.squeeze(dim=1)[mask])

        # if criterion is SSIM:
        # loss = self.criterion(pred, label)
        # if criterion is MSE:
        # loss = self.criterion(pred, label)
        
        loss = self.L1_loss(pred, label) + self.D_loss(pred, label) + self.E_loss(pred, label) + self.P_loss(pred, label)[0]
        ssim_value = self.ssim(pred, label)
        self.log_metrics(stage, loss, ssim_value)
        return {'loss': loss, 'ssim': ssim_value}
        
    def log_metrics(self, stage, loss, ssim_value):
        """Log metrics for training and validation stages."""
        self.log(f"{stage}_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log(f"{stage}_ssim", ssim_value, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

    def on_train_start(self):
        self.ssim.to(self.device)

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        return self.shared_step(batch, stage='train')
    
    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        return self.shared_step(batch, stage='valid')
    
    # def on_epoch_end(self) -> None:
    #     if torch.cuda.is_available():
    #         torch.cuda.empty_cache()
    def configure_optimizers(self) -> Any:
        # return {
        #     'optimizer': self.optimizer,
        #     'lr_scheduler': {
        #         'scheduler': torch.optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=1e-5, step_size_up=500,
        #                                                        max_lr=1e-3, gamma=0.5, mode='exp_range'),
        #         'interval': 'epoch',
        #         'frequency': 1,
        #         "monitor": "valid_loss",
        #         'strict': True,
        #         # "cycle_momentum": False
        #     }
        # }
        return self.optimizer
    
class Pl_EnhacnementModel_Unet(pl.LightningModule):
    def __init__(self, model, optimizer, criterion, **kwargs):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.ssim = StructuralSimilarityIndexMeasure(data_range=None)
        L1_loss,P_loss,E_loss,D_loss = init_loss()
        self.L1_loss = L1_loss
        self.P_loss = P_loss
        self.E_loss = E_loss
        self.D_loss = D_loss

        # self.save_hyperparameters() # multi gpu에서 에러발생 일으킴
    def forward(self, x):
        return self.model(x)
    def preprocessing_data(self, batch):
        """Apply padding and prepare masks."""
        input = F.pad(batch['image'], (0, 124, 0, 24))
        label = F.pad(batch['label'], (0, 124, 0, 24))
        mask = F.pad(batch['mask'], (0, 124, 0, 24))
        
        label = label * mask
        return input, label, mask
    def shared_step(self, batch, stage):
        input, label, mask = batch['image'], batch['label'], batch['mask']
        # input, label, mask = self.preprocessing_data(batch=batch)
        label = transform_image_unit_HU_to_mm(label)
        input = transform_image_unit_HU_to_mm(input)
        mask = transform_image_unit_HU_to_mm(mask)

        input = torch.concat([input, mask], dim=1)
        pred = self.forward(input)
        
        # a = 1.
        # loss = a*self.criterion(pred.squeeze(), label.squeeze()) + min_std(pred, label)*(1-a) 
        # loss = self.criterion(pred.squeeze(dim=1)[mask], label.squeeze(dim=1)[mask])

        # if criterion is SSIM:
        # loss = self.criterion(pred, label)
        # if criterion is MSE:
        # loss = self.criterion(pred, label)
        
        loss = self.L1_loss(pred, label) + self.D_loss(pred, label) + self.E_loss(pred, label) + self.P_loss(pred, label)[0]
        ssim_value = self.ssim(pred, label)
        self.log_metrics(stage, loss, ssim_value)
        return {'loss': loss, 'ssim': ssim_value}
        
    def log_metrics(self, stage, loss, ssim_value):
        """Log metrics for training and validation stages."""
        self.log(f"{stage}_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log(f"{stage}_ssim", ssim_value, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

    def on_train_start(self):
        self.ssim.to(self.device)

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        return self.shared_step(batch, stage='train')
    
    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        return self.shared_step(batch, stage='valid')
    
    # def on_epoch_end(self) -> None:
    #     if torch.cuda.is_available():
    #         torch.cuda.empty_cache()
    def configure_optimizers(self) -> Any:
        # return {
        #     'optimizer': self.optimizer,
        #     'lr_scheduler': {
        #         'scheduler': torch.optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=1e-5, step_size_up=500,
        #                                                        max_lr=1e-3, gamma=0.5, mode='exp_range'),
        #         'interval': 'epoch',
        #         'frequency': 1,
        #         "monitor": "valid_loss",
        #         'strict': True,
        #         # "cycle_momentum": False
        #     }
        # }
        return self.optimizer
    
    
class Pl_EnhacnementModel_Unet_with_given_loss(pl.LightningModule):
    def __init__(self, model, optimizer, criterion, **kwargs):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.ssim = StructuralSimilarityIndexMeasure(data_range=None)

        # self.save_hyperparameters() # multi gpu에서 에러발생 일으킴
    def forward(self, x):
        return self.model(x)
    def preprocessing_data(self, batch):
        """Apply padding and prepare masks."""
        input = F.pad(batch['image'], (0, 124, 0, 24))
        label = F.pad(batch['label'], (0, 124, 0, 24))
        mask = F.pad(batch['mask'], (0, 124, 0, 24))
        
        label = label * mask
        return input, label, mask
    def shared_step(self, batch, stage):
        input, label, mask = batch['image'], batch['label'], batch['mask']
        # input, label, mask = self.preprocessing_data(batch=batch)
        
        input = torch.concat([input, mask], dim=1)
        pred = self.forward(input)
        loss = self.criterion(pred, label)        
        ssim_value = self.ssim(pred, label)
        self.log_metrics(stage, loss, ssim_value)
        return {'loss': loss, 'ssim': ssim_value}
        
    def log_metrics(self, stage, loss, ssim_value):
        """Log metrics for training and validation stages."""
        self.log(f"{stage}_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log(f"{stage}_ssim", ssim_value, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

    def on_train_start(self):
        self.ssim.to(self.device)

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        return self.shared_step(batch, stage='train')
    
    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        return self.shared_step(batch, stage='valid')
    
    # def on_epoch_end(self) -> None:
    #     if torch.cuda.is_available():
    #         torch.cuda.empty_cache()
    def configure_optimizers(self) -> Any:
        return {
            'optimizer': self.optimizer,
            'lr_scheduler': {
                'scheduler': torch.optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=1e-5, step_size_up=500,
                                                               max_lr=1e-3, gamma=0.5, mode='exp_range'),
                'interval': 'epoch',
                'frequency': 1,
                "monitor": "valid_loss",
                'strict': True,
                "cycle_momentum": False
            }
        }
        # return self.optimizer
class Pl_EnhacnementModel_Unet_wo_perceptualloss(pl.LightningModule):
    def __init__(self, model, optimizer, criterion, **kwargs):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.ssim = StructuralSimilarityIndexMeasure(data_range=None)
        L1_loss,_,E_loss,D_loss = init_loss()
        self.L1_loss = L1_loss
        self.E_loss = E_loss
        self.D_loss = D_loss

        # self.save_hyperparameters() # multi gpu에서 에러발생 일으킴
    def forward(self, x):
        return self.model(x)
    def preprocessing_data(self, batch):
        """Apply padding and prepare masks."""
        input = F.pad(batch['image'], (0, 124, 0, 24))
        label = F.pad(batch['label'], (0, 124, 0, 24))
        mask = F.pad(batch['mask'], (0, 124, 0, 24))
        
        label = label * mask
        return input, label, mask
    def shared_step(self, batch, stage):
        input, label, mask = batch['image'], batch['label'], batch['mask']
        # input, label, mask = self.preprocessing_data(batch=batch)
        
        input = torch.concat([input, mask], dim=1)
        pred = self.forward(input)
        
        # a = 1.
        # loss = a*self.criterion(pred.squeeze(), label.squeeze()) + min_std(pred, label)*(1-a) 
        # loss = self.criterion(pred.squeeze(dim=1)[mask], label.squeeze(dim=1)[mask])

        # if criterion is SSIM:
        # loss = self.criterion(pred, label)
        # if criterion is MSE:
        # loss = self.criterion(pred, label)
        
        loss = self.L1_loss(pred, label) + self.D_loss(pred, label) + self.E_loss(pred, label)
        ssim_value = self.ssim(pred, label)
        self.log_metrics(stage, loss, ssim_value)
        return {'loss': loss, 'ssim': ssim_value}
        
    def log_metrics(self, stage, loss, ssim_value):
        """Log metrics for training and validation stages."""
        self.log(f"{stage}_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log(f"{stage}_ssim", ssim_value, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

    def on_train_start(self):
        self.ssim.to(self.device)

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        return self.shared_step(batch, stage='train')
    
    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        return self.shared_step(batch, stage='valid')
    
    # def on_epoch_end(self) -> None:
    #     if torch.cuda.is_available():
    #         torch.cuda.empty_cache()
    def configure_optimizers(self) -> Any:
        return {
            'optimizer': self.optimizer,
            'lr_scheduler': {
                'scheduler': torch.optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=1e-5, step_size_up=500,
                                                               max_lr=1e-3, gamma=0.5, mode='exp_range'),
                'interval': 'epoch',
                'frequency': 1,
                "monitor": "valid_loss",
                'strict': True,
                "cycle_momentum": False
            }
        }
        # return self.optimizer