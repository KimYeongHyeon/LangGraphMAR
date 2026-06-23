import numpy as np
import torch
import torch

def merge_masked_prediction_batch(image, pred, mask):
    """
    Calculates the complete prediction for a batch of images by adding the masked predicted values to the images.

    Args:
        image (torch.Tensor): The batch of image tensors.
        pred (torch.Tensor): The batch of predicted tensors.
        mask (torch.Tensor): The batch of mask tensors.

    Returns:
        torch.Tensor: The batch of complete prediction tensors.

    """
    
    if image.shape != pred.shape:
        raise ValueError("image and predicted tensors must have the same shape.")
    
    # Ensure mask is aligned in shape and device with image and pred
    # Assuming mask might need broadcasting to match the number of channels in image and pred
    if mask.dim() == 3:  # if mask is (N, H, W), unsqueeze to (N, 1, H, W)
        mask = mask.unsqueeze(1)
    
    mask = mask.to(pred.device)
    
    # Apply the mask to the predictions
    masked_pred = pred * mask
    
    # Sum the original image and the masked predictions
    complete_pred = image + masked_pred
    
    return complete_pred

def merge_masked_prediction(image, pred, mask):
    """
    Calculates the complete prediction by adding the image and predicted values.

    Args:
        image (torch.Tensor): The image tensor.
        pred (torch.Tensor): The predicted tensor.

    Returns:
        torch.Tensor: The complete prediction tensor.

    """
    
    image = image.cpu().numpy()[0][0]
    pred = pred.cpu().detach().numpy()[0][0]
    if image.shape != pred.shape:
        raise ValueError("image and predicted tensors must have the same shape.")
    
    pred = pred * mask.squeeze().cpu().numpy()
    pred = image + pred
    return pred
