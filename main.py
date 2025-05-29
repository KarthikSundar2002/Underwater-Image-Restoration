# Copyright (c) EEEM071, University of Surrey
import os
import os.path as osp
import sys
import time
import warnings
import src.ModelTrainer as mt
import torch
import torch.backends.cudnn as cudnn
from args import argument_parser
from matplotlib import pyplot as plt

from src.utils.Visualiser import loadModelFromWeights, ProcessImageUsingModel
from src.utils.loggers import Logger
import src.DataManipulation.DataManager as dataManager

parser = argument_parser()
args = parser.parse_args()

def main():
    print(torch.__version__)
    global args
    print(args.evaluate)

    if not args.use_avai_gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_devices

    use_gpu = torch.cuda.is_available()
    if args.use_cpu:
        use_gpu = False

    outputdirectory = "logs/" + "/arch-" + str(args.arch) + "/optimizer-" + str(args.optim) + "/loss-"+str(args.lossf)+"/maxEpoch-" + str(args.max_epoch) + "/lr-" + str(args.lr)  + "/batchSize-" + str(args.train_batch_size) + "/perspective-" + str(args.randomPerspective) + "-rotate-" + str(args.randomRotate)
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
        #rawImageDirectory = "data/uw_data/uw_data/manipulated/a"
        #referenceImageDirectory = "data/uw_data/uw_data/manipulated/b"
        rawImageDirectory = "uw_data/uw_data/train/a"
        referenceImageDirectory = "uw_data/uw_data/train/b"
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
        test_dir = "uw_data/uw_data/test/a"
        test_ref = "uw_data/uw_data/test/b"
        trainer = mt.ModelTrainer(dm.currentRawDataDirectory, dm.currentReferenceDataDirectory,test_dir, test_ref)
        trainer.train(args, args.arch ,args.max_epoch, args.lr)
    else:
        print("Evaluating...")
        input_dir = "image_in/"
        output_dir = "image_out/"
        model_path = "fflMix-0.0003-NewBigModel-1748292271.4851427-Wavelet/best_spectral_transformer_128.pth"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if not os.path.exists(input_dir):
            os.makedirs(input_dir)

        device = "cuda" if use_gpu else "cpu"

        model = loadModelFromWeights(device, model_path, args, args.arch)

        input_files = os.listdir(input_dir)

        print(f'Number of input images: {len(input_files)}')
        for file in input_files:
            print(f'Processing {file}')
        # fileToTest = "data/kaggle/manipulated/uieb-dataset-raw/2_img_.png"
            imarray = ProcessImageUsingModel(device, input_dir + file, model, output_dir, file)
            # plt.imshow(imarray, interpolation='nearest', cmap = plt.cm.Spectral)
            # plt.savefig("ReferenceImage.png")


if __name__ == "__main__":
    main()
