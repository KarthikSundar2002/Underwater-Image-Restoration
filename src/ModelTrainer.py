import pathlib as pl
import os
import datetime

import torch
import importlib

from pytorch_msssim import MS_SSIM

from src.losses import CharbonnierLoss, Gradient_Loss, VGGPerceptualLoss, ColorLoss
from src import Models
from src.DataManipulation.DataLoader import get_dataloaders
from src.Models.SpectralTransformer import SpectralTransformer
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import StepLR
import time
from tqdm import tqdm

from src.utils.wandb_logger import WandBLogger
from src.utils.Visualiser import ProcessImageUsingModel
from timm.utils import NativeScaler

class ModelTrainer:
    def __init__(self, inputDirectory, referenceDirectory):
        self.inputDir = inputDirectory
        self.referenceDir = referenceDirectory


    def train(self, args, arch="SpectroFormer", num_epochs=10, learning_rate=0.001, device="cuda" if torch.cuda.is_available() else "cpu"):
        wandb_logger = WandBLogger(args)
        if not device =="cuda":
            print(f"WARNING, NOT USING CUDA. Using device: {device}")

        print("Preparing data loaders...batch size" + str(args.train_batch_size))
        train_loader, test_loader = get_dataloaders(self.inputDir, self.referenceDir, args.train_batch_size)

        print("Initializing model...")
        model = Models.init_model(
            name=arch,
            #use_dwt=args.use_dwt,
        )
        model = model.to(device)

        gradient_loss = Gradient_Loss().to(device)
        charbonnier_loss = CharbonnierLoss().to(device)
        perceptual_loss = VGGPerceptualLoss().to(device)
        ms_ssim_loss = MS_SSIM(win_size=11, win_sigma=1.5, data_range=1, size_average=True, channel=3).to(device)
        loss_scaler = NativeScaler()
        criterion = torch.nn.L1Loss()
        L2_loss = torch.nn.MSELoss()
        colorLoss = ColorLoss().to(device)


        if args.optim == "adam":
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        elif args.optim == "adamw":
            optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer: {args.optimizer}")

        wandb_logger.watch_model(model)
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch - num_epochs) / float(num_epochs + 1)
            return lr_l
        scheduler = StepLR(optimizer, step_size=10, gamma=0.95)
        print(f"Starting training for {num_epochs} epochs...")
        best_loss = float('inf')
        Training_start_time = time.time()
        # fileToTest = "tiny_data/train/a/6_img_.png"
        # with torch.no_grad():
        #     ProcessImageUsingModel('cuda', fileToTest, model, f"{Training_start_time}",f"Model Output without Training", wandb_logger)
        fileToTest = "../data/kaggle/manipulated/uieb-dataset-raw/6_img_.png"
        directory = f"{args.lossf}-{args.lr}-{args.arch}-{Training_start_time}//"


        with torch.no_grad():
            ProcessImageUsingModel('cuda', fileToTest, model, directory, f"Model Output without Training", wandb_logger)
        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0
            start_time = time.time()

            for i, (raw_imgs, ref_imgs) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")):
                raw_imgs = raw_imgs.to(device)
                ref_imgs = ref_imgs.to(device)

                optimizer.zero_grad()
                outputs = model(raw_imgs)

                loss = charbonnier_loss(outputs, ref_imgs)
                if args.lossf == "L1":
                    loss = criterion(outputs, ref_imgs)
                if args.lossf == "L1withColor":
                    loss =  0.75 * colorLoss(outputs,ref_imgs) + 0.25 * criterion(outputs, ref_imgs)
                elif args.lossf == "L2":
                    loss = L2_loss(outputs, ref_imgs)
                #TODO: Finalise on loss function and remove the split here when done.
                elif args.lossf == "charbonnier":
                    loss = charbonnier_loss(outputs, ref_imgs)
                elif args.lossf == "perceptual":
                    loss = perceptual_loss(outputs, ref_imgs)
                elif args.lossf == "gradient":
                    loss = gradient_loss(outputs, ref_imgs)
                elif args.lossf == "mix":
                    loss = 0.03*charbonnier_loss(outputs,ref_imgs) +0.025*perceptual_loss(outputs,ref_imgs)+0.02*gradient_loss(outputs,ref_imgs)+0.01*(1-ms_ssim_loss(outputs,ref_imgs))
                elif args.lossf == "bigMix":
                    loss = 0.4 * charbonnier_loss(outputs, ref_imgs) + 0.25 * perceptual_loss(outputs,
                                                                                                ref_imgs) + 0.25 * gradient_loss(
                        outputs, ref_imgs) + 0.1 * (1 - ms_ssim_loss(outputs, ref_imgs))
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                #scheduler.step()

                epoch_loss += loss.item()

                # Print progress every 10 batches
                if (i + 1) % 1 == 0:
                    print(f"Batch {i + 1}/{len(train_loader)}, Loss: {loss.item():.6f}, Norm: {norm:.6f}")

                metrics = wandb_logger.format_train_metrics(
                    loss.item(),
                    optimizer.param_groups[0]["lr"],
                )
                wandb_logger.log_train_metrics(metrics, epoch, i, len(train_loader))


            avg_epoch_loss = epoch_loss / len(train_loader)

            epoch_time = time.time() - start_time
            print(f"Epoch {epoch + 1}/{num_epochs} completed in {epoch_time:.2f}s, Avg Loss: {avg_epoch_loss:.6f}")
            if (epoch + 1) % 4 == 0:

                # Validation phase
                model.eval()
                val_loss = 0
                with torch.no_grad():
                    for raw_imgs, ref_imgs in test_loader:
                        raw_imgs = raw_imgs.to(device)
                        ref_imgs = ref_imgs.to(device)

                        outputs = model(raw_imgs)
                        if args.lossf == "L1":
                            loss = criterion(outputs, ref_imgs)
                        if args.lossf == "L1withColor":
                            loss = 0.75 * colorLoss(outputs, ref_imgs) + 0.25 * criterion(outputs, ref_imgs)
                        elif args.lossf == "L2":
                            loss = L2_loss(outputs, ref_imgs)
                        # Calculate loss
                        elif args.lossf == "charbonnier":
                            loss = charbonnier_loss(outputs, ref_imgs)
                        elif args.lossf == "perceptual":
                            loss = perceptual_loss(outputs, ref_imgs)

                        elif args.lossf == "gradient":
                            loss = gradient_loss(outputs, ref_imgs)

                        elif args.lossf == "mix":
                            loss = 0.03 * charbonnier_loss(outputs, ref_imgs) + 0.025 * perceptual_loss(outputs,
                                                                                                        ref_imgs) + 0.02 * gradient_loss(
                                outputs, ref_imgs) + 0.01 * (1 - ms_ssim_loss(outputs, ref_imgs))
                        #loss = 0.03*charbonnier_loss(ref_imgs,outputs) +0.025*perceptual_loss(outputs,ref_imgs)+0.02*gradient_loss(outputs,ref_imgs)+0.01*(1-ms_ssim_loss(outputs,ref_imgs))
                        #loss = criterion(outputs, ref_imgs)

                        elif args.lossf == "bigMix":
                            loss = 0.4 * charbonnier_loss(outputs, ref_imgs) + 0.25 * perceptual_loss(outputs,
                                                                                                  ref_imgs) + 0.25 * gradient_loss(
                            outputs, ref_imgs) + 0.1 * (1 - ms_ssim_loss(outputs, ref_imgs))
                    #loss = 0.03*charbonnier_loss(ref_imgs,outputs) +0.025*perceptual_loss(outputs,ref_imgs)+0.02*gradient_loss(outputs,ref_imgs)+0.01*(1-ms_ssim_loss(outputs,ref_imgs))
                    #loss = criterion(outputs, ref_imgs)

                        val_loss +=  loss.item()

                avg_val_loss = val_loss / len(test_loader)
                print(f"Validation Loss: {avg_val_loss:.6f}")
                metrics = wandb_logger.format_test_metrics(avg_val_loss,epoch_time)
                wandb_logger.log_test_metrics(metrics)

                fileToTest = "data/kaggle/manipulated/uieb-dataset-raw/6_img_.png"
                best_loss_epoch = avg_val_loss < best_loss
                # Save model if it's the best so far
                if best_loss_epoch:
                    best_loss = avg_val_loss
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': best_loss,
                    }, 'best_spectral_transformer.pth')
                    print(f"Model saved with loss: {best_loss:.6f}")
                    with torch.no_grad():
                        #ProcessImageUsingModel('cuda', fileToTest, model,"Best" )
                        ProcessImageUsingModel('cuda', fileToTest, model, directory ,f"Epoch {epoch}_ Best True", wandb_logger)

                else:
                    torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': avg_val_loss,
                    }, 'latest_spectroformer.pth')
                    with torch.no_grad():
                        ProcessImageUsingModel('cuda', fileToTest, model, directory ,f"Epoch {epoch}_ Best False", wandb_logger)

        print("Training completed!")
        wandb_logger.finish()

        return model


    def evaluate(self, args, model_path='best_spectral_transformer.pth',
                 device="cuda" if torch.cuda.is_available() else "cpu"):
        from src.Models.SpectralTransformer import SpectralTransformer
        import numpy as np
        from skimage.metrics import peak_signal_noise_ratio as psnr
        from skimage.metrics import structural_similarity as ssim
        import matplotlib.pyplot as plt

        _, test_loader = get_dataloaders(self.rawDataDirectory, self.remasteredDataDirectory, args.test_batch_size)

        model = SpectralTransformer().to(device)
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        psnr_values = []
        ssim_values = []

        results_dir = 'evaluation_results'
        os.makedirs(results_dir, exist_ok=True)

        with torch.no_grad():
            for i, (raw_img, ref_img) in enumerate(test_loader):
                raw_img = raw_img.to(device)
                ref_img = ref_img.to(device)

                enhanced_img = model(raw_img)

                enhanced_np = enhanced_img.cpu().squeeze().permute(1, 2, 0).numpy()
                enhanced_np = np.clip(enhanced_np, 0, 1)  # Clip values to [0, 1]

                ref_np = ref_img.cpu().squeeze().permute(1, 2, 0).numpy()
                raw_np = raw_img.cpu().squeeze().permute(1, 2, 0).numpy()

                curr_psnr = psnr(ref_np, enhanced_np)
                curr_ssim = ssim(ref_np, enhanced_np, multichannel=True)
                psnr_values.append(curr_psnr)
                ssim_values.append(curr_ssim)

                # Print progress
                if (i + 1) % 20 == 0:
                    print(f"Processed {i + 1}/{len(test_loader)} test images")

        avg_psnr = np.mean(psnr_values)
        avg_ssim = np.mean(ssim_values)

        print(f"Evaluation Results:")
        print(f"Average PSNR: {avg_psnr:.2f} dB")
        print(f"Average SSIM: {avg_ssim:.4f}")

        with open(f'{results_dir}/metrics.txt', 'w') as f:
            f.write(f"Average PSNR: {avg_psnr:.2f} dB\n")
            f.write(f"Average SSIM: {avg_ssim:.4f}\n")

        return avg_psnr, avg_ssim



