import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import numpy as np

import pytorch_lightning as pl
from torchmetrics.image import StructuralSimilarityIndexMeasure
from typing import Any
from pytorch_lightning.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from skimage.metrics import structural_similarity as SKssim
from skimage.metrics import peak_signal_noise_ratio as SKpsnr
import segmentation_models_pytorch as smp

import torch
import torch.nn.functional as F
from utils.metric import ImageQualityEvaluator
from loss.losses import *
from torchmetrics.image import StructuralSimilarityIndexMeasure as ssim
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from utils.uformer import Uformer

from utils.metric import BatchImageQualityEvaluator
compute_batch_metrics = BatchImageQualityEvaluator()

import wandb

# Import external models for enhancement
try:
    from models.external.NAFNet_model.archs.NAFNet_arch import NAFNet
    NAFNET_AVAILABLE = True
except ImportError:
    NAFNET_AVAILABLE = False
    print("Warning: NAFNet not available. Install with: git clone https://github.com/megvii-research/NAFNet.git in code/models/external/")

try:
    from models.external.Restormer_model.archs.restormer_arch import Restormer
    RESTORMER_AVAILABLE = True
except ImportError:
    RESTORMER_AVAILABLE = False
    print("Warning: Restormer not available. Install with: git clone https://github.com/swz30/Restormer.git in code/models/external/")

try:
    from models.external.SwinUnet_model.archs.swinunet_arch import SwinUNet
    SWINUNET_AVAILABLE = True
except ImportError:
    SWINUNET_AVAILABLE = False
    print("Warning: SwinUNet not available. Install with: git clone https://github.com/HuCaoFighting/Swin-Unet.git in code/models/external/")


# ==================== Model Factory Functions ====================

def load_nafnet(width=32, middle_blk_num=12, img_channel=1):
    """
    NAFNet 공식 구현 로드
    
    Args:
        width: NAFNet width (default: 32)
        middle_blk_num: middle block number (default: 12)
        img_channel: input image channels (default: 1)
    
    Returns:
        NAFNet model
    """
    if not NAFNET_AVAILABLE:
        raise ImportError("NAFNet not available. Please install it first.")
    
    model = NAFNet(
        img_channel=img_channel,
        width=width,
        middle_blk_num=middle_blk_num,
        enc_blk_nums=[2, 2, 4, 8],
        dec_blk_nums=[2, 2, 2, 2]
    )
    print(f"✓ NAFNet ready: width={width}, middle_blk_num={middle_blk_num}")
    return model


def load_restormer(inp_channels=1, out_channels=1, dim=48, 
                   num_blocks=[4,6,6,8], num_refinement_blocks=4):
    """
    Restormer 공식 구현 로드
    
    Args:
        inp_channels: input channels (default: 1)
        out_channels: output channels (default: 1)
        dim: model dimension (default: 48)
        num_blocks: number of blocks per stage (default: [4,6,6,8])
        num_refinement_blocks: refinement blocks (default: 4)
    
    Returns:
        Restormer model
    """
    if not RESTORMER_AVAILABLE:
        raise ImportError("Restormer not available. Please install it first.")
    
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


def load_swinunet(img_size=224, in_channels=1, out_channels=1, 
                  embed_dim=96, depths=[2,2,2,2], num_heads=[3,6,12,24], 
                  window_size=7):
    """
    SwinUNet 공식 구현 로드
    
    Args:
        img_size: input image size (default: 224)
        in_channels: input channels (default: 1)
        out_channels: output channels (default: 1)
        embed_dim: embedding dimension (default: 96)
        depths: depth of each stage (default: [2,2,2,2])
        num_heads: number of attention heads per stage (default: [3,6,12,24])
        window_size: window size for attention (default: 7)
    
    Returns:
        SwinUNet model
    """
    if not SWINUNET_AVAILABLE:
        raise ImportError("SwinUNet not available. Please install it first.")
    
    model = SwinUNet(
        img_size=img_size,
        patch_size=4,
        in_chans=in_channels,
        num_classes=out_channels,
        embed_dim=embed_dim,
        depths=depths,
        depths_decoder=depths,
        num_heads=num_heads,
        window_size=window_size,
        mlp_ratio=4.,
        qkv_bias=True,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.1,
        ape=False,
        patch_norm=True,
        use_checkpoint=False,
        final_upsample="expand_first"
    )
    print(f"✓ SwinUNet loaded: img_size={img_size}, embed_dim={embed_dim}, depths={depths}")
    return model


def create_model(model_type, anatomy, **kwargs):
    """
    공식 구현체를 사용하여 모델 생성
    
    Args:
        model_type: 'unet', 'uformer', 'nafnet', 'restormer', 'swinunet'
        anatomy: 'head' or 'body' or 'enhancement'
        **kwargs: 모델별 추가 파라미터
            - in_channels: input channels (default: 1 for MAR, 3 for enhancement)
            - out_channels: output channels (default: 1 for MAR, 3 for enhancement)
            - UNet: arch (encoder name, default: 'resnet101')
            - Uformer: embed_dim (default: 16), win_size (default: 8), 
                      depths (default: [2,2,2,2,2,2,2,2,2]), img_size (default: 512)
            - NAFNet: width (default: 32), middle_blk_num (default: 12)
            - Restormer: dim (default: 48), num_blocks (default: [4,6,6,8])
            - SwinUNet: img_size (default: 224), embed_dim (default: 96), 
                       depths (default: [2,2,2,2]), num_heads (default: [3,6,12,24]), 
                       window_size (default: 7)
    
    Returns:
        Initialized model
    """
    print(f"\n{'='*60}")
    print(f"Creating {model_type.upper()} model for {anatomy}")
    print(f"{'='*60}\n")
    
    # 채널 설정 (enhancement는 3채널, MAR은 1채널)
    in_channels = kwargs.get('in_channels', 3 if anatomy == 'enhancement' else 1)
    out_channels = kwargs.get('out_channels', 3 if anatomy == 'enhancement' else 1)
    
    if model_type == 'unet':
        arch = kwargs.get('arch', 'resnet101')
        model = smp.Unet(
            encoder_name=arch,
            encoder_depth=5,
            encoder_weights=None,
            decoder_use_batchnorm=True,
            decoder_channels=(256, 128, 64, 32, 16),
            in_channels=in_channels,
            classes=out_channels,
        )
        print(f"✓ Created UNet model with {arch} encoder (in={in_channels}, out={out_channels})")
    
    elif model_type == 'uformer':
        img_size = kwargs.get('img_size', 512)
        embed_dim = kwargs.get('embed_dim', 16)
        win_size = kwargs.get('win_size', 8)
        mlp_ratio = kwargs.get('mlp_ratio', 4.)
        depths = kwargs.get('depths', [2, 2, 2, 2, 2, 2, 2, 2, 2])
        model = Uformer(
            img_size=img_size,
            embed_dim=embed_dim,
            depths=depths,
            win_size=win_size,
            mlp_ratio=mlp_ratio,
            token_projection='linear',
            token_mlp='leff',
            modulator=True,
            shift_flag=False
        )
        print(f"✓ Uformer ready: embed_dim={embed_dim}, win_size={win_size}, img_size={img_size}")
        
    elif model_type == 'nafnet':
        width = kwargs.get('width', 32)
        middle_blk_num = kwargs.get('middle_blk_num', 12)
        model = load_nafnet(
            width=width,
            middle_blk_num=middle_blk_num,
            img_channel=in_channels
        )
        print(f"✓ NAFNet ready: width={width}, middle_blk_num={middle_blk_num}, channels={in_channels}")
        
    elif model_type == 'restormer':
        dim = kwargs.get('dim', 48)
        num_blocks = kwargs.get('num_blocks', [4,6,6,8])
        model = load_restormer(
            inp_channels=in_channels,
            out_channels=out_channels,
            dim=dim,
            num_blocks=num_blocks,
            num_refinement_blocks=4
        )
        print(f"✓ Restormer ready: dim={dim}, num_blocks={num_blocks}, channels={in_channels}")
        
    elif model_type == 'swinunet':
        img_size = kwargs.get('img_size', 224)
        embed_dim = kwargs.get('embed_dim', 96)
        depths = kwargs.get('depths', [2,2,2,2])
        num_heads = kwargs.get('num_heads', [3,6,12,24])
        window_size = kwargs.get('window_size', 7)
        model = load_swinunet(
            img_size=img_size,
            in_channels=in_channels,
            out_channels=out_channels,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size
        )
        print(f"✓ SwinUNet ready: img_size={img_size}, embed_dim={embed_dim}, channels={in_channels}")
        
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose from 'unet', 'uformer', 'nafnet', 'restormer', 'swinunet'")
    
    print(f"\n{'='*60}\n")
    return model


# ==================== PyTorch Lightning Models ====================

# To remove
# def z_score(x, mean, std, **kwargs):
#     return (x - mean) / (std + 1e-8)
# def zscore_conditional(x, **kwargs):
#     print(kwargs)
#     key = kwargs.get("original_key", None)
#     print(key)
#     if key == "image":
#         return z_score(x, mean_image, std_image)
#     elif key == "label":
#         return z_score(x, mean_label, std_label)
#     else:
#         return x
# def denormalize(data, mean, std):
#     return data * (std + 1e-8) + mean
class ResidualEnhancementModel(nn.Module):
    def __init__(self,img_size=512, embed_dim=16, win_size=8, mlp_ratio=4.):
        super(ResidualEnhancementModel, self).__init__()
        depths=[2, 2, 2, 2, 2, 2, 2, 2, 2]
        self.model = Uformer(img_size=512, embed_dim=embed_dim,depths=depths,
                        win_size=win_size, mlp_ratio=mlp_ratio, token_projection='linear', token_mlp='leff', modulator=True, shift_flag=False)

    def forward(self, x):
        residual = self.model(x)
        enhanced = x + residual  # Enhanced 이미지 생성
        return enhanced, residual
    
class DLbasedMARModel(pl.LightningModule):
    """
    This class is used to train a model for the MAR task.
    """
    def __init__(self, model, criterion, optimizer=None, lr=1e-3, anatomy="head"):
        super(DLbasedMARModel, self).__init__()
        self.save_hyperparameters(ignore=['model', 'criterion'])
        self.model = model
        self.criterion = criterion
        self.optimizer_class = optimizer or optim.Adam
        self.anatomy = anatomy
        self.validation_outputs = []  # epoch별 샘플 이미지 저장용
    def forward(self, x):
        return self.model(x)
    def denormalize(self, x):
        x = x * 2000 - 1000
        return x
    def _shared_step(self, batch, batch_idx, stage):    
        images, labels, masks = batch['image'], batch['label'], batch['mask']
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        
        metrics = compute_batch_metrics(self.denormalize(labels), 
                                        self.denormalize(outputs), 
                                        masks, 
                                        self.anatomy)
        self.log(f'{stage}_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        for metric_name, value in metrics.items():
            self.log(f'{stage}_{metric_name}', value, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss
    
    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, 'train')
    def validation_step(self, batch, batch_idx):
        loss = self._shared_step(batch, batch_idx, 'valid')
        
        # 첫 배치만 저장 (메모리 절약)
        if batch_idx == 0:
            images, labels, masks = batch['image'], batch['label'], batch['mask']
            outputs = self(images)
            self.validation_outputs.append({
                'images': images[:4].detach().cpu(),  # 첫 4개만
                'labels': labels[:4].detach().cpu(),
                'outputs': outputs[:4].detach().cpu(),
                'masks': masks[:4].detach().cpu()
            })
        return loss
    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, 'test')
    
    def on_validation_epoch_end(self):
        """Validation epoch 종료 시 wandb에 이미지 로깅"""
        if not self.validation_outputs:
            return
        
        # 마지막 배치의 샘플만 사용
        sample = self.validation_outputs[-1]
        images = sample['images']
        labels = sample['labels']
        outputs = sample['outputs']
        masks = sample['masks']
        
        # denormalize (HU 단위로 변환)
        images_hu = self.denormalize(images)
        labels_hu = self.denormalize(labels)
        outputs_hu = self.denormalize(outputs)
        
        # wandb에 이미지 로깅
        wandb_images = []
        for i in range(min(4, images.shape[0])):  # 최대 4개
            # 단일 슬라이스 선택 (중간 슬라이스)
            img = images_hu[i, 0].numpy()  # [H, W]
            label = labels_hu[i, 0].numpy()
            output = outputs_hu[i, 0].numpy()
            mask = masks[i, 0].numpy()
            
            # 정규화 for visualization (-1000 ~ 1000 HU → 0 ~ 1)
            vmin, vmax = -1000, 1000
            img_norm = np.clip((img - vmin) / (vmax - vmin), 0, 1)
            label_norm = np.clip((label - vmin) / (vmax - vmin), 0, 1)
            output_norm = np.clip((output - vmin) / (vmax - vmin), 0, 1)
            
            # 에러 맵 계산
            error_map = np.abs(label_norm - output_norm)
            
            wandb_images.append(
                wandb.Image(
                    img_norm,
                    caption=f"Sample {i+1}: Input",
                )
            )
            wandb_images.append(
                wandb.Image(
                    label_norm,
                    caption=f"Sample {i+1}: Ground Truth",
                )
            )
            wandb_images.append(
                wandb.Image(
                    output_norm,
                    caption=f"Sample {i+1}: Prediction",
                )
            )
            wandb_images.append(
                wandb.Image(
                    error_map,
                    caption=f"Sample {i+1}: Error Map",
                )
            )
        
        # WandB에 로깅
        if self.logger:
            self.logger.experiment.log({
                "validation_samples": wandb_images,
                "epoch": self.current_epoch
            })
        
        # 메모리 정리
        self.validation_outputs.clear()

    def configure_optimizers(self):
        optimizer = self.optimizer_class(self.parameters(), lr=self.hparams.lr, betas=(0.5, 0.9))
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5, 
            patience=10
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                 "monitor": "valid_loss",
                "frequency": 1
            }
        }
        

# # To remove
# def combined_loss(pred, target, ratio=0.7):
#     ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
#     mse_loss = nn.MSELoss()(pred, target)
#     ssim_loss = 1 - ssim(pred, target)
#     return mse_loss * ratio + (1-ratio) * ssim_loss  # SSIM 손실 포함

class ImageEnhancement(pl.LightningModule):
    def __init__(self, model, lr=5e-4, use_residual=True):
        super(ImageEnhancement, self).__init__()
        self.model = model
        self.criterion = self.combined_loss
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
        self.use_residual = use_residual
        self.save_hyperparameters(ignore=['model'])
        
    def combined_loss(self, pred, target, ssim_fn, alpha=0.7):
        l1 = nn.L1Loss()(pred, target)
        ssim_loss = 1 - ssim_fn(pred, target)
        return alpha * l1 + (1-alpha) *ssim_loss

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch, batch_idx, stage):
        images, labels, original = batch['image'], batch['label'], batch['original']
        enhanced, residual = self(images)
        
        if self.use_residual:
            # Residual learning: loss = |residual - labels|
            loss = self.criterion(residual, labels, ssim_fn=self.ssim)
        else:
            # Direct prediction: loss = |enhanced - original|
            loss = self.criterion(enhanced, original, ssim_fn=self.ssim)
       
        ssim_val = self.ssim(enhanced, original)
        
        self.log(f'{stage}_loss', loss, on_step=True, on_epoch=True, prog_bar=True, )
        self.log(f'{stage}_ssim', ssim_val.detach(), on_step=True, on_epoch=True, prog_bar=True, metric_attribute='ssim')
        return loss
    

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, 'train')

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, batch_idx, 'valid')

    def test_step(self, batch, batch_idx):
        self._shared_step(batch, batch_idx, 'test')
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-4, weight_decay=0)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
    
class SinogramInpainting(pl.LightningModule):
    def __init__(self, model, optimizer, criterion, **kwargs):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.ssim = StructuralSimilarityIndexMeasure(data_range=None)
        self.image_quality_evaluator = ImageQualityEvaluator()
        # Note: self.save_hyperparameters() commented out due to multi-GPU errors
    
    def forward(self, x):
        return self.model(x)
    
    def preprocess_batch(self, batch):
        """Apply padding to input, label, and mask, and prepare the masked label."""
        input_padded = F.pad(batch['input'], (0, 124, 0, 24))
        label_padded = F.pad(batch['label'], (0, 124, 0, 24))
        mask_padded = F.pad(batch['mask'], (0, 124, 0, 24)).unsqueeze(dim=1).bool()
        label_masked = label_padded * mask_padded
        return input_padded, label_masked, mask_padded

    def step(self, batch, stage):
        """Shared logic for training and validation steps."""
        input_padded, label_masked, mask_padded = self.preprocess_batch(batch)
        predictions = self.forward(input_padded)
        loss = self.criterion(predictions, label_masked)
        ssim_value = self.ssim(predictions, label_masked)
        self.log_metrics(stage, loss, ssim_value)
        return {'loss': loss, 'ssim': ssim_value}
        
    def log_metrics(self, stage, loss, ssim_value):
        """Log metrics for training and validation stages."""
        self.log(f"{stage}_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log(f"{stage}_ssim", ssim_value, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

    def on_train_start(self):
        """Ensure metrics like SSIM are moved to the correct device."""
        self.ssim.to(self.device)

    def training_step(self, batch, batch_idx):
        return self.step(batch, stage='train')
    
    def validation_step(self, batch, batch_idx):
        return self.step(batch, stage='valid')
    
    def configure_optimizers(self):
        """Define optimizer configuration."""
        return self.optimizer

class ImageClassifier(pl.LightningModule):
    def __init__(self, model, criterion, threshold=0.5):
        super(ImageClassifier, self).__init__()
        self.model = model
        self.model.fc = nn.Linear(self.model.fc.in_features, 1) 
        self.criterion = criterion
        self.threshold = threshold
        
    def forward(self, x):
        # Pass through the model and apply sigmoid for a probability-like output
        return torch.sigmoid(self.model(x))

    def training_step(self, batch, batch_idx):
        images, labels = batch['image'], batch['label']
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch['image'], batch['label']
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        # acc = (outputs.argmax(dim=1) == labels).float().mean()
        predictions = (outputs > self.threshold).float()
        acc = (predictions == labels).float().mean()
        self.log('valid_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('valid_acc', acc, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        return optimizer