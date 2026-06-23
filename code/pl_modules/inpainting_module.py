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

class SinogramInpainting_Unet_Residual_with_given_criterion_with_adam(pl.LightningModule):
    def __init__(self, model, optimizer, criterion, **kwargs):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.ssim = StructuralSimilarityIndexMeasure(data_range=None)
        self.ImageQualityEvaluator = ImageQualityEvaluator()

        self.save_hyperparameters() # multi gpu에서 에러발생 일으킴
    def forward(self, x):
        return self.model(x)
    def preprocessing_data(self, batch):
        """Apply padding and prepare masks."""
        image = F.pad(batch['image'], (0, 124, 0, 24))
        label = F.pad(batch['label'], (0, 124, 0, 24))
        mask = F.pad(batch['mask'], (0, 124, 0, 24)).bool()
        
        label = label * mask
        return image, label, mask
    def shared_step(self, batch, stage):
        image_1ch, label, mask = self.preprocessing_data(batch=batch)
        image = torch.concat([image_1ch, mask], dim=1)
        pred = self.forward(image)
        
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
        return self.optimizer
class SinogramInpainting_Unet_Residual_with_given_criterion(pl.LightningModule):
    def __init__(self, model, optimizer, criterion, **kwargs):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.ssim = StructuralSimilarityIndexMeasure(data_range=None)
        self.ImageQualityEvaluator = ImageQualityEvaluator()

        self.save_hyperparameters() # multi gpu에서 에러발생 일으킴
    def forward(self, x):
        return self.model(x)
    def preprocessing_data(self, batch):
        """Apply padding and prepare masks."""
        image = F.pad(batch['image'], (0, 124, 0, 24))
        label = F.pad(batch['label'], (0, 124, 0, 24))
        mask = F.pad(batch['mask'], (0, 124, 0, 24)).bool()
        
        label = label * mask
        return image, label, mask
    def shared_step(self, batch, stage):
        image_1ch, label, mask = self.preprocessing_data(batch=batch)
        image = torch.concat([image_1ch, mask], dim=1)
        pred = self.forward(image)
        
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



def init_loss():
    L1_loss= L1Loss(reduction='mean').cuda()
    D_loss = SSIM().cuda()
    E_loss = EdgeLoss().cuda()
    P_loss = PerceptualLoss({'conv1_2': 1, 'conv2_2': 1,'conv3_4': 1,'conv4_4': 1}, perceptual_weight = 1 ,criterion='mse').cuda()
    return L1_loss,P_loss,E_loss,D_loss
# https://openaccess.thecvf.com/content_cvpr_2018/papers/Yu_Generative_Image_Inpainting_CVPR_2018_paper.pdf
# 나중에 이름 변경 필요
class SinogramInpainting_Unet_Residual(pl.LightningModule):
    def __init__(self, model, optimizer, criterion, **kwargs):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.ssim = StructuralSimilarityIndexMeasure(data_range=None)
        self.ImageQualityEvaluator = ImageQualityEvaluator()

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
        image = F.pad(batch['image'], (0, 124, 0, 24))
        label = F.pad(batch['label'], (0, 124, 0, 24))
        mask = F.pad(batch['mask'], (0, 124, 0, 24)).bool()
        
        label = label * mask
        return image, label, mask
    def shared_step(self, batch, stage):
        image_1ch, label, mask = self.preprocessing_data(batch=batch)
        image = torch.concat([image_1ch, mask], dim=1)
        pred = self.forward(image)
        
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
    
# class SinogramInpainting_Unet_Residual(pl.LightningModule):
#     def __init__(self, model, optimizer, criterion):
#         super().__init__()
#         self.model = model
#         self.optimizer = optimizer
#         self.criterion = criterion
#         self.ssim = StructuralSimilarityIndexMeasure(data_range=None)

#     def forward(self, x):
#         return self.model(x)

#     def shared_step(self, batch, stage):
#         input, label, mask = self.prepare_data(batch)
#         pred = self.forward(input)
#         loss = self.criterion(pred, label)
#         ssim_value = self.ssim(pred, label)
        
#         self.log_metrics(stage, loss, ssim_value)
#         return {'loss': loss, 'ssim': ssim_value}

#     def prepare_data(self, batch):
#         """Apply padding and prepare masks."""
#         input = F.pad(batch['input'], (0, 124, 0, 24))
#         label = F.pad(batch['label'], (0, 124, 0, 24))
#         mask = F.pad(batch['mask'], (0, 124, 0, 24)).unsqueeze(dim=1).bool()
        
#         label = label * mask
#         return input, label, mask

#     def log_metrics(self, stage, loss, ssim_value):
#         """Log metrics for training and validation stages."""
#         self.log(f"{stage}_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
#         self.log(f"{stage}_ssim", ssim_value, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

#     def training_step(self, batch, batch_idx):
#         return self.shared_step(batch, 'train')

#     def validation_step(self, batch, batch_idx):
#         return self.shared_step(batch, 'valid')

#     def configure_optimizers(self):
#         return self.optimizer

#     def on_train_start(self):
#         self.ssim.to(self.device)


class SinogramInpainting_Unet(pl.LightningModule):
    def __init__(self, model, optimizer, criterion, **kwargs):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.ssim = StructuralSimilarityIndexMeasure(data_range=None)

        # self.save_hyperparameters() # multi gpu에서 에러발생 일으킴
    def forward(self, x):
        return self.model(x)
    def shared_step(self, batch, stage):
        input = batch['input']
        label = batch['label']
        mask = batch['mask']
        
        input = F.pad(input, (0, 124, 0, 24, 0, 0, 0, 0)) # 1000x900 -> 1024x1024
        label = F.pad(label, (0, 124, 0, 24, 0, 0, 0, 0)) # 1000x900 -> 1024x1024
        mask = F.pad(mask, (0, 124, 0, 24))
        # add channel dimension
        mask = mask.unsqueeze(dim=1)
        mask = mask.bool()

        pred = self.forward(input)
        # a = 1.
        # loss = a*self.criterion(pred.squeeze(), label.squeeze()) + min_std(pred, label)*(1-a) 
        # loss = self.criterion(pred.squeeze(dim=1)[mask], label.squeeze(dim=1)[mask])

        # if criterion is SSIM:
        # loss = self.criterion(pred, label)
        # if criterion is MSE:
        loss = self.criterion(pred[mask], label[mask])
        
        ssim_value = self.ssim(pred, label)
        # mse_value = F.mse_loss(pred.squeeze(dim=1)[mask], label.squeeze(dim=1)[mask])
        # loss = 1 - ssim_value

        self.log(f"{stage}_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log(f"{stage}_ssim", ssim_value, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        

        return {'loss': loss, 'ssim': ssim_value}
        # tp, fp, fn, tn = smp.metrics.get_stats(torch.argmax(pred, 1), label.long(), mode='multiclass', num_classes=5)
        # iou  = smp.metrics.iou_score(tp, fp, fn, tn, reduction='macro-imagewise')
        # self.log(f"{stage}_iou", iou, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # self.log(f"{stage}_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # return {'loss': loss, 'iou': iou}
    
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
    
class SinogramInpainting_ResUnet(pl.LightningModule):
    def __init__(self, model, optimizer, criterion, **kwargs):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.ssim = StructuralSimilarityIndexMeasure(data_range=None)

        # self.save_hyperparameters() # multi gpu에서 에러발생 일으킴
    def forward(self, x):
        return self.model(x)
    def shared_step(self, batch, stage):
        input = batch['input']
        label = batch['label']
        # mask = batch['mask']
        
        input = F.pad(input, (0, 100, 0, 0, 0, 0, 0, 0)) # 1000x900 -> 1000x1000
        label = F.pad(label, (0, 100, 0, 0, 0, 0, 0, 0)) # 1000x900 -> 1000x1000
        
        # mask = F.pad(torch.tensor(mask, dtype=torch.float32), (0, 100, 0, 0)).to(bool) 
        pred = self.forward(input)
        # print(pred.shape, label.shape)
        # a = 1.
        # loss = a*self.criterion(pred.squeeze(), label.squeeze()) + min_std(pred, label)*(1-a) 
        # loss = self.criterion(pred.squeeze(dim=1)[mask], label.squeeze(dim=1)[mask])
        loss = self.criterion(pred.squeeze(dim=1), label.squeeze(dim=1))
        
        ssim_value = self.ssim(pred, label)
        loss = 1 - ssim_value

        self.log(f"{stage}_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log(f"{stage}_ssim", ssim_value, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        

        return {'loss': loss, 'ssim': ssim_value}
        # tp, fp, fn, tn = smp.metrics.get_stats(torch.argmax(pred, 1), label.long(), mode='multiclass', num_classes=5)
        # iou  = smp.metrics.iou_score(tp, fp, fn, tn, reduction='macro-imagewise')
        # self.log(f"{stage}_iou", iou, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # self.log(f"{stage}_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # return {'loss': loss, 'iou': iou}
    
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
                # "cycle_momentum": False
            }
        }
        # return self.optimizer
    
def get_random_mask(tgt_mtdiam):
    """
    Args:
        tgt_mtdiam(int): 직경; np.random.randint(5, 15)로 우선 생각중
    
    """
    import os
    _mat_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "AAPM_datachallenge", "simulation_scripts", "metal_masks.mat",
    )
    try:
        metal_file = loadmat(_mat_path)
    except Exception as e:
        raise FileNotFoundError(f"metal_masks.mat not found at {_mat_path}") from e
    metal_file = metal_file['tumor_imgs']/255
    metal_file[metal_file>=0.1] = 1
    metal_file[metal_file<0.1] = 0

def generate_random_metal(image, num_metal: int):
    assert num_metal >= 1

    if isinstance(image, torch.tensor):
        image  = image.numpy()
    
    width, height = image.shape[:2]

    mask = np.zeros_like(image.shape)
    x, y = 0, 0
    

    for i in range(num_metal):
        if image[x, y] in list(range(-1000, -900)):
            x = np.randint(0, width)
            y = np.randint(0, height)
        