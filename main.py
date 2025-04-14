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
from matplotlib import pyplot as plt

from src.utils.Visualiser import loadModelFromWeights, ProcessImageUsingModel
from src.utils.loggers import Logger
import src.DataManipulation.DataManager as dataManager

parser = argument_parser()
args = parser.parse_args()

def main():
    global args
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

    if use_gpu:
        cudnn.benchmark = True
    else:
        warnings.warn("Currently using CPU, however, GPU is highly recommended")

    if not args.evaluate:
        print("Initializing image data manager")
        # rawImageDirectory = "../data/kaggle/manipulated/uieb-dataset-raw"
        # referenceImageDirectory = "../data/kaggle/manipulated/uieb-dataset-reference"
        rawImageDirectory = "./uw_data/uw_data/train/a"
        referenceImageDirectory = "./uw_data/uw_data/train/b"
        dm = dataManager.DataManager()

        dm.setDownloadedLocations(
            rawDataDirectory=rawImageDirectory,
            remasteredDataDirectory=referenceImageDirectory
        )
        # dm.download()
        # dm.preProcess()
        #dm.dataAugment() - this does NOT work - missmatch in the dataloader with increased number of raw images

        print("Starting training")
        print(f"Raw Data Directory: {dm.currentRawDataDirectory}")
        print(f"Reference Image Directory: {dm.currentReferenceDataDirectory}")

        trainer = mt.ModelTrainer(dm.currentRawDataDirectory, dm.currentReferenceDataDirectory)
        trainer.train(args, args.arch ,args.max_epoch, args.lr)
    else:
        pthFileLocation = "best_spectral_transformer.pth"
        fileToTest = "../data/kaggle/manipulated/uieb-dataset-raw/2_img_.png"

        device = "cuda" if use_gpu else "cpu"

        model = loadModelFromWeights(device, pthFileLocation)

        imarray = ProcessImageUsingModel(device, fileToTest, model)

        plt.imshow(imarray, interpolation='nearest', cmap = plt.cm.Spectral)
        plt.savefig("ReferenceImage.png")

if __name__ == "__main__":
    main()
