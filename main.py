# Copyright (c) EEEM071, University of Surrey
import os
import os.path as osp
import sys
import time
import warnings
import cv2
import src.ModelTrainer as mt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from args import argument_parser, dataset_kwargs, optimizer_kwargs, lr_scheduler_kwargs
from src.Models.SpectralTransformer import SpectralTransformer
from src import Models

from src.utils.loggers import Logger
from src.utils.wandb_logger import WandBLogger
import src.DataManipulation.DataManager as dataManager

parser = argument_parser()
args = parser.parse_args()

def main():
    global args, wandb_logger
    print(args.evaluate)

    if not args.use_avai_gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_devices

    use_gpu = torch.cuda.is_available()
    if args.use_cpu:
        use_gpu = False

    outputdirectory = "logs/" + "/arch-" + str(args.arch) + "/optimizer-" + str(args.optim) + "/maxEpoch-" + str(args.max_epoch) + "/lr-" + str(args.lr)  + "/batchSize-" + str(args.train_batch_size) + "/perspective-" + str(args.randomPerspective) + "-rotate-" + str(args.randomRotate)
    args.save_dir = outputdirectory
    log_name = "log_test.txt" if args.evaluate else "log_train.txt"
    sys.stdout = Logger(osp.join(args.save_dir, log_name))
    print("==========")
    print("Start time:{}".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
    print("==========")
    print(f"==========\nArgs:{args}\n==========")

    wandb_logger = WandBLogger(args)

    if use_gpu:
        print(f"Currently using GPU {args.gpu_devices}")
        cudnn.benchmark = True
    else:
        warnings.warn("Currently using CPU, however, GPU is highly recommended")

    if not args.evaluate:
        print("Initializing image data manager")
        rawImageDirectory = "../data/kaggle/manipulated/uieb-dataset-raw"
        referenceImageDirectory = "../data/kaggle/manipulated/uieb-dataset-reference"

        dm = dataManager.DataManager()

        dm.setDownloadedLocations(
            rawDataDirectory=rawImageDirectory,
            remasteredDataDirectory=referenceImageDirectory
        )
        dm.download()
        dm.preProcess()
        dm.dataAugment()

        print("Starting training")
        print(f"Raw Data Directory: {dm.currentRawDataDirectory}")
        print(f"Reference Image Directory: {dm.currentReferenceDataDirectory}")

        trainer = mt.ModelTrainer(dm.currentRawDataDirectory, dm.currentReferenceDataDirectory)
        trainer.train(args.arch ,args.max_epoch, args.lr)
    else:
        pthFileLocation = "best_spectral_transformer.pth"
        fileToTest = "../data/kaggle/manipulated/uieb-dataset-raw/2_img_.png"

        device = "cuda" if use_gpu else "cpu"

        model = SpectralTransformer()
        model.load_state_dict(torch.load(pthFileLocation, weights_only=True)["model_state_dict"])

        img = cv2.imread(fileToTest)
        img_array = np.array(img)

        input_tensor = torch.from_numpy(img_array)
        input_tensor = input_tensor.permute(2, 0, 1)
        input_tensor = input_tensor.unsqueeze(0)
        input_tensor = input_tensor.float()

        model.to(device)
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

        from matplotlib import pyplot as plt
        plt.imshow(result_numpy, interpolation='nearest')
        plt.show()

        plt.imshow(img_array, interpolation='nearest')
        plt.show()

if __name__ == "__main__":
    main()
