import os

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt

from src.Models import SpectralTransformer


def ProcessImageUsingModel(device, fileToTest, model, saveName):
    img = cv2.imread(fileToTest)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_array = np.array(rgb)
    input_tensor = torch.from_numpy(img_array)
    input_tensor = input_tensor.permute(2, 0, 1)
    input_tensor = input_tensor.unsqueeze(0)
    input_tensor = input_tensor.float()
    input_tensor = input_tensor.to(device)
    result = model(input_tensor)
    result_cpu = result.detach().cpu()
    if result_cpu.dim() == 4 and result_cpu.shape[0] == 1:
        result_squeezed = result_cpu.squeeze(0)  # Shape: [C, H, W]
    else:
        result_squeezed = result_cpu  # Keep as is if not 4D with batch=1
    # 3. Permute dimensions: CHW -> HWC
    if result_squeezed.dim() == 3:  # Only permute if it has C, H, W dims
        result_hwc = result_squeezed.permute(1, 2, 0)  # Shape: [H, W, C]
    else:
        result_hwc = result_squeezed  # Keep as is if not 3D (e.g., grayscale output)
    # 4. Convert to NumPy array
    result_numpy = result_hwc.numpy()
    result_numpy = np.clip(result_numpy, 0, 1)
    result_numpy = (result_numpy * 255).astype(np.uint8)

    plt.imshow(result_numpy, interpolation='nearest',cmap = plt.cm.Spectral)
    os.makedirs("Images", exist_ok=True)

    plt.savefig(f"Images/{saveName}.png")
    return img_array


def loadModelFromWeights(device, pthFileLocation):
    model = SpectralTransformer()
    model.load_state_dict(torch.load(pthFileLocation, weights_only=True)["model_state_dict"])
    model.to(device)
    return model