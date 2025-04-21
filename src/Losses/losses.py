import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from pytorch_msssim import MS_SSIM
from focal_frequency_loss import FocalFrequencyLoss as FFL
from timm.utils import NativeScaler


class LossFunction():
    def __init__(self, loss_name, device):
        self.loss_name = loss_name

        if loss_name in ["L1","L1withColor"]:
            self.criterion = torch.nn.L1Loss()
        if loss_name in ["L1withColor"]:
            self.colorLoss = ColorLoss().to(device)
        if loss_name in ["L2"]:
            self.L2_loss = torch.nn.MSELoss()
        if loss_name in ["mix","bigMix","charbonnier","fflCharbonnier"]:
            self.charbonnier_loss = CharbonnierLoss().to(device)
        if loss_name in ["mix", "bigMix", "perceptual"]:
            self.perceptual_loss = VGGPerceptualLoss().to(device)
        if loss_name in ["mix", "bigMix", "gradient"]:
            self.gradient_loss = Gradient_Loss().to(device)
        if loss_name in ["mix", "bigMix"]:
            self.ms_ssim_loss = MS_SSIM(win_size=11, win_sigma=1.5, data_range=1, size_average=True, channel=3).to(device)
        if loss_name in ["fflCharbonnier"]:
            self.ffl = FFL(loss_weight=1.0,alpha=1.0).to(device)

    def getloss(self, predicted_data, truth_data):
        if self.loss_name == "L1":
            loss = self.criterion(predicted_data, truth_data)
            loss = loss / (truth_data.shape[0] * truth_data.shape[1])
        elif self.loss_name == "L1withColor":
            loss = 0.75 * self.colorLoss(predicted_data, truth_data) + 0.25 * self.criterion(predicted_data, truth_data)
            loss = loss / (truth_data.shape[0] * truth_data.shape[1])
        elif self.loss_name == "L2":
            loss = self.L2_loss(predicted_data, truth_data)
            loss = loss / (truth_data.shape[0] * truth_data.shape[1])
        # Calculate loss
        elif self.loss_name == "charbonnier":
            loss = self.charbonnier_loss(predicted_data, truth_data)
        elif self.loss_name == "perceptual":
            loss = self.perceptual_loss(predicted_data, truth_data)

        elif self.loss_name == "gradient":
            loss = self.gradient_loss(predicted_data, truth_data)

        elif self.loss_name == "mix":
            loss = 0.03 * self.charbonnier_loss(predicted_data, truth_data) + 0.025 * self.perceptual_loss(predicted_data,
                                                                                        truth_data) + 0.02 * self.gradient_loss(
                predicted_data, truth_data) + 0.01 * (1 - self.ms_ssim_loss(predicted_data, truth_data))
        #loss = 0.03*charbonnier_loss(ref_imgs,outputs) +0.025*perceptual_loss(outputs,ref_imgs)+0.02*gradient_loss(outputs,ref_imgs)+0.01*(1-ms_ssim_loss(outputs,ref_imgs))
        #loss = criterion(outputs, ref_imgs)

        elif self.loss_name == "bigMix":
            loss = 0.4 * self.charbonnier_loss(predicted_data, truth_data) + 0.25 * self.perceptual_loss(predicted_data,
                                                                                  truth_data) + 0.25 * self.gradient_loss(
            predicted_data, truth_data) + 0.1 * (1 - self.ms_ssim_loss(predicted_data, truth_data))
            #loss = 0.03*charbonnier_loss(ref_imgs,outputs) +0.025*perceptual_loss(outputs,ref_imgs)+0.02*gradient_loss(outputs,ref_imgs)+0.01*(1-ms_ssim_loss(outputs,ref_imgs))
            #loss = criterion(outputs, ref_imgs)
        elif self.loss_name == "fflCharbonnier":
            loss = self.ffl(predicted_data, truth_data) + self.charbonnier_loss(predicted_data, truth_data)
        else:
            raise ValueError(f"Unsupported loss: {self.loss_name}")

        loss = torch.clamp(loss, min=0.0, max=100.0)
        return loss

class Gradient_Loss(nn.Module):
    def __init__(self):
        super(Gradient_Loss, self).__init__()

        kernel_g = [[[0,1,0],[1,-4,1],[0,1,0]],
                    [[0,1,0],[1,-4,1],[0,1,0]],
                    [[0,1,0],[1,-4,1],[0,1,0]]]
        kernel_g = torch.FloatTensor(kernel_g).unsqueeze(0).permute(1, 0, 2, 3)
        self.weight_g = nn.Parameter(data=kernel_g, requires_grad=False)

    def forward(self, x,xx):
        grad = 0
        y = x
        yy = xx
        gradient_x = F.conv2d(y, self.weight_g,groups=3)
        gradient_xx = F.conv2d(yy,self.weight_g,groups=3)
        l = nn.L1Loss()
        a = l(gradient_x,gradient_xx)
        grad = grad + a
        return grad
class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, loss_weight=1.0, reduction='mean', eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss

class ColorLoss(nn.Module):
    def __init__(self):
        super(ColorLoss, self).__init__()

    def forward(self,y_pred,y_true):

        #
        # avg_pred = torch.mean(y_pred, dim=(2, 3))
        # avg_true = torch.mean(y_true, dim=(2, 3))
        # avg = avg_pred - avg_true
        #
        # elementwise_loss = avg ** 2
        # loss = torch.mean(elementwise_loss)
        #
        diff = y_pred - y_true
        perchannelloss = diff ** 2
        channelsLoss = torch.mean(perchannelloss, dim=(2, 3))
        loss = torch.mean(channelsLoss)
        return loss
    
class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, inp, target, feature_layers=[0, 1, 2, 3], style_layers=[]):
        # if inp.shape[1] != 3:
        #     inp = inp.repeat(1, 3, 1, 1)
        #     target = target.repeat(1, 3, 1, 1)
        inp = (inp-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            inp = self.transform(inp, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = inp
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += torch.nn.functional.l1_loss(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        return loss

