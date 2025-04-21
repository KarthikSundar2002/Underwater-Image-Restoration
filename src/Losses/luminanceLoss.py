import torch
import torch.nn as nn
import torch.nn.functional as F

class LuminanceLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(LuminanceLoss, self).__init__()
        self.reduction = reduction

        self.rgb_to_y_coeffs = torch.tensor([0.299, 0.587, 0.114],
                                            dtype=torch.float32).view(1, 3, 1, 1)
    def forward(self, x, target):

        self.rgb_to_y_coeffs = self.rgb_to_y_coeffs.to(x.device)

        x_luminance = torch.sum(x * self.rgb_to_y_coeffs, dim=1, keepdim=True)
        target_luminance = torch.sum(target * self.rgb_to_y_coeffs, dim=1, keepdim=True)

        loss = F.mse_loss(x_luminance, target_luminance, reduction=self.reduction)

        return loss