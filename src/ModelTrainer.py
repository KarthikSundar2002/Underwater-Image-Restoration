import pathlib as pl
import os
import datetime

import torch
import importlib

from pytorch_msssim import MS_SSIM

from src.losses import CharbonnierLoss, Gradient_Loss, VGGPerceptualLoss
from src import Models
from src.DataManipulation.DataLoader import get_dataloaders
from src.Models.SpectralTransformer import SpectralTransformer
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import time
from tqdm import tqdm

from src.utils.wandb_logger import WandBLogger
from src.utils.Visualiser import ProcessImageUsingModel


class ModelTrainer:
    def __init__(self, inputDirectory, referenceDirectory):
        self.inputDir = inputDirectory
        self.referenceDir = referenceDirectory


    def train(self, args, arch="SpectroFormer", num_epochs=10, learning_rate=0.001, device="cuda" if torch.cuda.is_available() else "cpu"):
        """
        Train the Spectral Transformer model on UIEB dataset

        Args:
            num_epochs (int): Number of training epochs
            learning_rate (float): Learning rate for optimizer
            device (str): Device to run training on ('cuda' or 'cpu')
        """
        wandb_logger = WandBLogger(args)
        # Set device
        print(f"Using device: {device}")

        # Get dataloaders
        print("Preparing data loaders...")
        # For additional training data, you can use augmented directories too

        #raw_aug_dir = os.path.join(os.path.dirname(self.rawDataDirectory), "augmented_raw")
        #ref_aug_dir = os.path.join(os.path.dirname(self.remasteredDataDirectory),
        #                           "augmented_remastered")

        # Get train and test loaders using the original data
        train_loader, test_loader = get_dataloaders(self.inputDir, self.referenceDir, args.train_batch_size)

        # Initialize the model
        print("Initializing model...")
        model = Models.init_model(
            name=arch,
        )
        model = model.to(device)
        wandb_logger.watch_model(model)

        # Define loss function and optimizer
        gradient_loss = Gradient_Loss().to(device)
        charbonnier_loss = CharbonnierLoss().to(device)
        perceptual_loss = VGGPerceptualLoss().to(device)
        ms_ssim_loss = MS_SSIM(win_size=11, win_sigma=1.5, data_range=1, size_average=True, channel=3).to(device)

        criterion = torch.nn.L1Loss()  # L1 loss is commonly used for image reconstruction
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        # scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0.00003)

        wandb_logger.watch_model(model)
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch - num_epochs) / float(num_epochs + 1)
            return lr_l
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
        # Training loop
        print(f"Starting training for {num_epochs} epochs...")
        best_loss = float('inf')

        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0
            start_time = time.time()

            # Training phase
            for i, (raw_imgs, ref_imgs) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")):
                # Move data to device
                raw_imgs = raw_imgs.to(device)
                ref_imgs = ref_imgs.to(device)

                # Forward pass
                optimizer.zero_grad()
                outputs = model(raw_imgs)

                incremental_loss = 0
                if args.lossf == "L1":
                    loss = criterion(outputs, ref_imgs)
                    incremental_loss = loss.item()
                # Calculate loss
                elif args.lossf == "charbonnier":
                    loss = charbonnier_loss(outputs, ref_imgs)
                    incremental_loss = loss
                elif args.lossf == "perceptual":
                    loss = perceptual_loss(outputs, ref_imgs)
                    incremental_loss = loss
                elif args.lossf == "gradient":
                    loss = gradient_loss(outputs, ref_imgs)
                    incremental_loss = loss
                elif args.lossf == "mix":
                    loss = 0.03*charbonnier_loss(outputs,ref_imgs) +0.025*perceptual_loss(outputs,ref_imgs)+0.02*gradient_loss(outputs,ref_imgs)+0.01*(1-ms_ssim_loss(outputs,ref_imgs))
                    incremental_loss = loss

                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                scheduler.step()

                epoch_loss += incremental_loss

                # Print progress every 10 batches
                if (i + 1) % 1 == 0:
                    print(f"Batch {i + 1}/{len(train_loader)}, Loss: {incremental_loss:.6f}")

                metrics = wandb_logger.format_train_metrics(
                    incremental_loss,
                    optimizer.param_groups[0]["lr"],

                )
                wandb_logger.log_train_metrics(metrics, epoch, i, len(train_loader))


            avg_epoch_loss = epoch_loss / len(train_loader)
            epoch_time = time.time() - start_time
            print(f"Epoch {epoch + 1}/{num_epochs} completed in {epoch_time:.2f}s, Avg Loss: {avg_epoch_loss:.6f}")

            # Validation phase
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for raw_imgs, ref_imgs in test_loader:
                    raw_imgs = raw_imgs.to(device)
                    ref_imgs = ref_imgs.to(device)

                    outputs = model(raw_imgs)
                    #loss = 0.03*charbonnier_loss(ref_imgs,outputs) +0.025*perceptual_loss(outputs,ref_imgs)+0.02*gradient_loss(outputs,ref_imgs)+0.01*(1-ms_ssim_loss(outputs,ref_imgs))
                    loss = criterion(outputs, ref_imgs)
                    val_loss += incremental_loss

            avg_val_loss = val_loss / len(test_loader)
            print(f"Validation Loss: {avg_val_loss:.6f}")
            metrics = wandb_logger.format_test_metrics(avg_val_loss,epoch_time)
            wandb_logger.log_test_metrics(metrics)

            fileToTest = "../data/kaggle/manipulated/uieb-dataset-raw/6_img_.png"
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
                    ProcessImageUsingModel('cuda', fileToTest, model,"Best" )
                    ProcessImageUsingModel('cuda', fileToTest, model, f"Epoch {epoch}_ Best True")

            else:
                torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_val_loss,
                }, 'latest_spectroformer.pth')
                with torch.no_grad():
                    ProcessImageUsingModel('cuda', fileToTest, model, f"Epoch {epoch}_ Best False")

        print("Training completed!")
        wandb_logger.finish()

        return model


    def evaluate(self, args, model_path='best_spectral_transformer.pth',
                 device="cuda" if torch.cuda.is_available() else "cpu"):
        """
        Evaluate the trained model on the test dataset

        Args:
            model_path (str): Path to the saved model
            device (str): Device to run evaluation on ('cuda' or 'cpu')
        """
        from src.Models.SpectralTransformer import SpectralTransformer
        import numpy as np
        from skimage.metrics import peak_signal_noise_ratio as psnr
        from skimage.metrics import structural_similarity as ssim
        import matplotlib.pyplot as plt

        # Get test dataloader
        _, test_loader = get_dataloaders(self.rawDataDirectory, self.remasteredDataDirectory, args.test_batch_size)

        # Load model
        model = SpectralTransformer().to(device)
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        # Metrics
        psnr_values = []
        ssim_values = []

        # Create directory for saving results
        results_dir = 'evaluation_results'
        os.makedirs(results_dir, exist_ok=True)

        with torch.no_grad():
            for i, (raw_img, ref_img) in enumerate(test_loader):
                raw_img = raw_img.to(device)
                ref_img = ref_img.to(device)

                # Generate enhanced image
                enhanced_img = model(raw_img)

                # Convert tensors to numpy arrays for metric calculation
                enhanced_np = enhanced_img.cpu().squeeze().permute(1, 2, 0).numpy()
                enhanced_np = np.clip(enhanced_np, 0, 1)  # Clip values to [0, 1]

                ref_np = ref_img.cpu().squeeze().permute(1, 2, 0).numpy()
                raw_np = raw_img.cpu().squeeze().permute(1, 2, 0).numpy()

                # Calculate metrics
                curr_psnr = psnr(ref_np, enhanced_np)
                curr_ssim = ssim(ref_np, enhanced_np, multichannel=True)

                psnr_values.append(curr_psnr)
                ssim_values.append(curr_ssim)

                # Save some sample images (every 10th image)
                if i % 10 == 0:
                    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                    axes[0].imshow(raw_np)
                    axes[0].set_title('Raw Underwater Image')
                    axes[0].axis('off')

                    axes[1].imshow(enhanced_np)
                    axes[1].set_title(f'Enhanced Image (PSNR: {curr_psnr:.2f}, SSIM: {curr_ssim:.4f})')
                    axes[1].axis('off')

                    axes[2].imshow(ref_np)
                    axes[2].set_title('Reference Image')
                    axes[2].axis('off')

                    plt.tight_layout()
                    plt.savefig(f'{results_dir}/sample_{i}.png')
                    plt.close()

                # Print progress
                if (i + 1) % 20 == 0:
                    print(f"Processed {i + 1}/{len(test_loader)} test images")

        # Calculate and print average metrics
        avg_psnr = np.mean(psnr_values)
        avg_ssim = np.mean(ssim_values)

        print(f"Evaluation Results:")
        print(f"Average PSNR: {avg_psnr:.2f} dB")
        print(f"Average SSIM: {avg_ssim:.4f}")

        # Save metrics to file
        with open(f'{results_dir}/metrics.txt', 'w') as f:
            f.write(f"Average PSNR: {avg_psnr:.2f} dB\n")
            f.write(f"Average SSIM: {avg_ssim:.4f}\n")

        return avg_psnr, avg_ssim



