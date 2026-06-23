import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import segmentation_models_pytorch as smp
from pathlib import Path

from albumentations.core.transforms_interface import DualTransform
from utils.models import (
    SinogramInpainting, 
    ImageClassifier, 
    ResidualEnhancementModel, 
    ImageEnhancement, 
)
from utils.uformer import Uformer

from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

def load_models(load_enhancement=True):
    """
    모델 로드 함수
    Args:
        load_enhancement (bool): enhancement 모델 로드 여부 (기본값: True)
    Returns:
        model_dict (dict): 로드된 모델들의 딕셔너리
            - inpainting: 해부학적 부위별 인페인팅 모델 딕셔너리
            - gc: 분류 모델
            - enhancement: 해부학적 부위별 enhancement 모델 딕셔너리 (load_enhancement=True인 경우)
    """
    # 프로젝트 루트 경로 계산 (utils.py가 code/utils/에 있으므로 2단계 상위)
    project_root = Path(__file__).parent.parent.parent
    checkpoints_dir = project_root / 'checkpoints'
    
    classification_base_model = models.resnet18(pretrained=True)
    inpainting_base_model = smp.Unet(
        encoder_name='tu-efficientnet_b4',
        encoder_weights=None,
        in_channels=2,
        classes=1,
    )

    anatomy_inpainting_models = {}
    anatomy_enhancement_models = {}

    # 각 해부학적 부위별 모델 로드
    for anatomy_part in ['head', 'body']:
        # # 체크포인트 디렉토리 경로 설정
        # inpainting_checkpoint_dir = f'./checkpoints/inpainting_model_old/inpainting_{anatomy_part}'
        
        # # 디렉토리 내 모든 체크포인트 파일 가져오기
        # checkpoint_files = glob.glob(os.path.join(inpainting_checkpoint_dir, '*'))
        
        # # 가장 최근 체크포인트 파일 선택 (시간 기준과 이름 기준)
        # latest_checkpoint_by_time = max(checkpoint_files, key=os.path.getctime)
        # latest_checkpoint_by_name = max(checkpoint_files, key=os.path.basename)
        
        latest_checkpoint_by_name = str(checkpoints_dir / f'inpainting_{anatomy_part}.ckpt')
        
        # 선택된 체크포인트에서 모델 로드
        inpainting_model = SinogramInpainting.load_from_checkpoint(
            latest_checkpoint_by_name,
            model=inpainting_base_model,
            optimizer=None,
            criterion=None
        )
        
        inpainting_model.eval()
        anatomy_inpainting_models[anatomy_part] = inpainting_model
    
        # Enhancement 모델은 선택적으로 로드
        if load_enhancement:
            enhancement_base_model = ResidualEnhancementModel()
            latest_checkpoint_by_name = str(checkpoints_dir / f'enhancement_{anatomy_part}.ckpt')
            enhancement_model = ImageEnhancement.load_from_checkpoint(
                checkpoint_path=latest_checkpoint_by_name,
                model=enhancement_base_model
            )
            enhancement_model.eval()
            anatomy_enhancement_models[anatomy_part] = enhancement_model


    gc_model = ImageClassifier.load_from_checkpoint(
        checkpoint_path=str(checkpoints_dir / 'gc.ckpt'), 
        model=classification_base_model,
        criterion=nn.BCEWithLogitsLoss()
    )
    gc_model.eval()

    
    model_dict = {
        "inpainting": anatomy_inpainting_models,
        "gc": gc_model,
    }
    
    # Enhancement 모델을 로드한 경우에만 딕셔너리에 추가
    if load_enhancement:
        model_dict["enhancement"] = anatomy_enhancement_models
    
    return model_dict

def create_checkpoint_dir(experiment_type, anatomy, network, arch, current_time, loss_metric='mse'):
    checkpoint_name = f'{anatomy}_{network}_{arch}_{experiment_type}_{current_time}'
    dirpath = os.path.join('checkpoints', experiment_type, f'{checkpoint_name}_loss_{loss_metric}')
    return checkpoint_name, dirpath

def setup_logger(checkpoint_name, logger_type='tensorboard', project_name="CTMAR"):
    if logger_type == 'tensorboard':
        return TensorBoardLogger("lightning_logs", name=checkpoint_name)
    elif logger_type == 'wandb':
        return WandbLogger(name=checkpoint_name, project=project_name)
    else:
        raise ValueError(f"Unsupported logger type: {logger_type}")

def create_model_checkpoint(dirpath, monitor_metric='valid_loss', mode='min', filename_pattern='{epoch:02d}-{valid_loss_epoch:.4f}', top_k=10):
    return ModelCheckpoint(
        dirpath=dirpath,
        filename=filename_pattern,
        monitor=monitor_metric,
        mode=mode,
        save_top_k=top_k
    )

def setup_early_stopping(monitor_metric='valid_loss', patience=30, mode='min', verbose=True):
    return EarlyStopping(
        monitor=monitor_metric,
        patience=patience,
        mode=mode,
        verbose=verbose
    )
    
    
def pad_to_divisible(array: np.ndarray, divisor: int = 32) -> np.ndarray:
    """
    Pads the input numpy array so that its height and width are divisible by 32.

    Args:
        array (np.ndarray): The input numpy array with shape (H, W) or (H, W, C), where H is the height,
                            W is the width, and C is the number of channels.

    Returns:
        np.ndarray: The padded numpy array.

    Raises:
        ValueError: If the input array does not have 2 or 3 dimensions.
    """
    if divisor in [0, 1]:
        return array
    if array.ndim not in [2, 3]:
        raise ValueError("Input array must have 2 or 3 dimensions (H, W) or (H, W, C)")

    # Get the original height and width
    height, width = array.shape[:2]

    # Calculate the padding needed to make height and width divisible by 32
    pad_height = (divisor - height % divisor) % divisor
    pad_width = (divisor - width % divisor) % divisor

    # Apply padding (divided equally on both sides of the height and width)
    padding = ((pad_height // 2, pad_height - pad_height // 2), (pad_width // 2, pad_width - pad_width // 2))

    # If the array has a channel dimension, add a dummy padding for it
    if array.ndim == 3:
        padding += ((0, 0),)

    return np.pad(array, padding, mode='constant', constant_values=0)

def unpad_to_original(array: np.ndarray, original_height: int, original_width: int) -> np.ndarray:
    """
    Unpads the input numpy array to its original size.

    Args:
        array (np.ndarray): The input numpy array with padding.
        original_height (int): The original height of the array before padding.
        original_width (int): The original width of the array before padding.

    Returns:
        np.ndarray: The unpadded numpy array.

    Raises:
        ValueError: If the input array does not have 2 or 3 dimensions.
    """
    if array.ndim not in [2, 3]:
        raise ValueError("Input array must have 2 or 3 dimensions (H, W) or (H, W, C)")

    current_height, current_width = array.shape[:2]

    # Check if the current dimensions are larger than or equal to the original dimensions
    if current_height < original_height or current_width < original_width:
        raise ValueError("Current dimensions are smaller than the original dimensions")

    # Calculate the start and end indices for slicing
    start_height = (current_height - original_height) // 2
    end_height = start_height + original_height
    start_width = (current_width - original_width) // 2
    end_width = start_width + original_width

    # Slice the array to get the original size
    if array.ndim == 2:
        return array[start_height:end_height, start_width:end_width]
    elif array.ndim == 3:
        return array[start_height:end_height, start_width:end_width, :]

# Custom transform to convert single-channel grayscale to three-channel RGB
def grayscale_to_rgb(image: np.ndarray, **kwargs) -> np.ndarray:
    """ 
    Convert single-channel grayscale to three-channel RGB
    """
    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=-1)

    if image.shape[2] == 1:  # Ensure it's a single-channel image 
        image = np.repeat(image, 3, axis=2)  # Repeat the single channel three times
    return image


# To remove
class MinMaxScaling(DualTransform):
    def __init__(self, always_apply=False, p=1.0):
        super(MinMaxScaling, self).__init__(always_apply, p)

    def apply(self, image, **params):
        # Apply min-max scaling to the image
        min_val = image.min()
        max_val = image.max()
        if max_val - min_val != 0:
            image = (image - min_val) / (max_val - min_val)
        return image

    def apply_to_mask(self, mask, **params):
        # Apply min-max scaling to the mask
        min_val = mask.min()
        max_val = mask.max()
        if max_val - min_val != 0:
            mask = (mask - min_val) / (max_val - min_val)
        return mask
if __name__ == '__main__':
    image = np.random.randint(0, 255, (100, 100, 3))
    padded_image = pad_to_divisible(image, 32)
    unpadded_image = unpad_to_original(padded_image, 100, 100)
    assert (image == unpadded_image).all(), "Unpadded image is not equal to the original image"

    image = np.random.randint(0, 255, (100, 100))
    padded_image = pad_to_divisible(image, 32)
    unpadded_image = unpad_to_original(padded_image, 100, 100)
    assert (image == unpadded_image).all(), "Unpadded image is not equal to the original image"