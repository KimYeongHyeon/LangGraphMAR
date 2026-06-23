import torch
import torch.nn as nn


class MaskedMSELoss(nn.Module):
    def __init__(self):
        """
        Initializes the MaskedMSELoss class which computes the Mean Squared Error loss
        on masked regions of the input tensors.
        """
        super(MaskedMSELoss, self).__init__()

    def forward(self, input, target, mask):
        """
        Forward pass for the masked MSE loss calculation.
        
        Args:
        - input (torch.Tensor): Predicted tensor.
        - target (torch.Tensor): Ground truth tensor.
        - mask (torch.Tensor): Mask tensor where 1 indicates the region of interest and 0 indicates ignored region.
        
        Returns:
        - torch.Tensor: Computed masked MSE loss.
        """
        # Ensure input, target, and mask have the same dimensions
        assert input.size() == target.size() == mask.size(), "Input, target, and mask must have the same dimensions"
        
        # Calculate the error
        diff = input - target
        
        # Apply the mask (only consider the regions where mask == 1)
        masked_diff = diff * mask
        
        # Calculate MSE loss on masked regions
        loss = torch.sum(masked_diff ** 2) / torch.sum(mask)
        
        return loss
