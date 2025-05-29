import os

import torch

from src.Losses.losses import LossFunction
from src import Models
from src.DataManipulation.DataLoader import get_dataloaders
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, MultiStepLR, ReduceLROnPlateau
import time
from tqdm import tqdm

from src.utils.wandb_logger import WandBLogger
from src.utils.Visualiser import ProcessImageUsingModel, save_from_tensor

from pytorch_msssim import ssim
def torchPSNR(tar_img, prd_img):
    imdff = torch.clamp(prd_img, 0, 1) - torch.clamp(tar_img, 0, 1)
    rmse = (imdff**2).mean().sqrt()
    ps = 20*torch.log10(1/rmse)
    return ps

def torchSSIM(tar_img, prd_img):
    return ssim(tar_img, prd_img, data_range=1.0, size_average=True)

class ModelTrainer:
    def __init__(self, inputDirectory, referenceDirectory, testInputDirectory, testReferenceDirectory,):
        self.inputDir = inputDirectory
        self.referenceDir = referenceDirectory
        self.testInputDir = testInputDirectory
        self.testReferenceDir = testReferenceDirectory

    def train(self, args, arch="SpectroFormer", num_epochs=10, learning_rate=0.001, device="cuda" if torch.cuda.is_available() else "cpu"):
        if not device =="cuda":
            print(f"WARNING, NOT USING CUDA. Using device: {device}")

        print("Preparing data loaders...batch size" + str(args.train_batch_size))
        train_loader, test_loader = get_dataloaders(self.inputDir, self.referenceDir,self.testInputDir,self.testReferenceDir ,args.train_batch_size)

        print("Initializing model...")
        model = Models.init_model(name=arch, use_dwt=args.use_dwt)
        model = model.to(device)
        wandb_logger = WandBLogger(args)
        wandb_logger.watch_model(model)
        lossfunction = LossFunction(args.lossf, device)
        optimizer = self.getOptimizer(args, learning_rate, model)

        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name, param.data.shape)

        #scheduler = StepLR(optimizer, step_size=1000, gamma=0.995)
        #scheduler = CosineAnnealingLR(optimizer, num_epochs, 0.000001)
        #scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.25)
        scheduler = MultiStepLR(optimizer, milestones=[1,100,250], gamma=0.25)
        best_loss = float('inf')

        Training_start_time = time.time()
        fileToTest = "uw_data/uw_data/train/a/6_img_.png"
        directory = f"{args.lossf}-{args.lr}-{args.arch}-{Training_start_time}-{args.use_dwt}//"

        # with torch.no_grad():#Write out image based on model initialization only
        #     ProcessImageUsingModel('cuda', fileToTest, model, directory, f"Model Output without Training", wandb_logger)

        print(f"Starting training for {num_epochs} epochs...")
        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0
            start_time = time.time()

            for batch, (raw_imgs, ref_imgs) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")):
                #print(f"Input Image Shape {raw_imgs.shape}",)
                #save_from_tensor(directory,f"input_img_{epoch}",raw_imgs)

                raw_imgs = raw_imgs.to(device)
                ref_imgs = ref_imgs.to(device)
                
                optimizer.zero_grad()
                outputs = model(raw_imgs)

                #print(f"Output Shape: {outputs.shape}")
                if args.lossf != "fflMix":
                    loss = lossfunction.getloss(outputs, ref_imgs)
                else:
                    loss, charb_loss, perc_loss, grad_loss, ffl_loss, ssim_loss = lossfunction.getloss(outputs, ref_imgs)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                epoch_loss += loss.item()

                # Print progress every 10 batches
                if (batch + 1) % 1 == 0:
                    print(f"Batch {batch + 1}/{len(train_loader)}, Loss: {loss.item():.6f}, Norm: {norm:.6f}")

                metrics = wandb_logger.format_train_metrics(
                    loss.item(),
                    optimizer.param_groups[0]["lr"],
                )
                wandb_logger.log_train_metrics(metrics, epoch, batch, len(train_loader))
                if args.lossf == "fflMix":
                    perc_loss_metric = wandb_logger.format_loss_metrics(
                        perc_loss.item(),
                        "Charbonnier",
                    )
                    wandb_logger.log_train_metrics(perc_loss_metric, epoch, batch, len(train_loader))
                    perc_loss_metric = wandb_logger.format_loss_metrics(
                        perc_loss.item(),
                        "Perceptual",
                    )
                    wandb_logger.log_train_metrics(perc_loss_metric, epoch, batch, len(train_loader))
                    grad_loss_metric = wandb_logger.format_loss_metrics(
                        grad_loss.item(),
                        "Gradient Loss",
                    )
                    wandb_logger.log_train_metrics(grad_loss_metric, epoch, batch, len(train_loader))
                    ffl_loss_metric = wandb_logger.format_loss_metrics(
                        ffl_loss.item(),
                        "FFL Loss",
                    )
                    wandb_logger.log_train_metrics(ffl_loss_metric, epoch, batch, len(train_loader))
                    ssim_loss_metric = wandb_logger.format_loss_metrics(
                        ssim_loss.item(),
                        "MS_SSIM Loss",
                    )
                    wandb_logger.log_train_metrics(ssim_loss_metric, epoch, batch, len(train_loader))
            avg_epoch_loss = epoch_loss / len(train_loader)
            epoch_time = time.time() - start_time
            scheduler.step()
            print(f"Epoch {epoch + 1}/{num_epochs} completed in {epoch_time:.2f}s, Avg Loss: {avg_epoch_loss:.6f}")

            if (epoch + 1) % 1 == 0:
                # Validation phase
                model.eval()
                val_loss = 0
                psnr = 0
                ssim = 0
                with torch.no_grad():
                    for raw_imgs, ref_imgs in test_loader:
                        raw_imgs = raw_imgs.to(device)
                        ref_imgs = ref_imgs.to(device)

                        outputs = model(raw_imgs)
                        if args.lossf != "fflMix":
                            loss = lossfunction.getloss(outputs, ref_imgs)
                        else:
                            loss, charb_loss, perc_loss, grad_loss, ffl_loss, ssim_loss = lossfunction.getloss(outputs, ref_imgs)
                        psnr += torchPSNR(ref_imgs,outputs)
                        ssim += torchSSIM(ref_imgs,outputs)
                        val_loss +=  loss.item()

                avg_val_loss = val_loss / len(test_loader)
                #scheduler.step(avg_val_loss)
                print(f"Validation Loss: {avg_val_loss:.6f}")
                avg_psnr = psnr / len(test_loader)
                avg_ssim = ssim / len(test_loader)
                metrics = wandb_logger.format_test_metrics(avg_val_loss,avg_psnr,avg_ssim,epoch_time)
                wandb_logger.log_test_metrics(metrics)

                self.SaveModel(avg_val_loss, best_loss, directory, epoch, model, optimizer)

        print("Training completed!")
        wandb_logger.finish()

        return model

    def SaveModel(self, avg_val_loss, best_loss, directory, epoch, model, optimizer):
        best_loss_epoch = avg_val_loss < best_loss
        # Save model if it's the best so far
        fileToTest = "uw_data/uw_data/train/a/6_img_.png"
        if not os.path.exists(directory):
            os.makedirs(directory)
        if best_loss_epoch:
            best_loss = avg_val_loss

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, directory + f'best_spectral_transformer_{epoch}.pth')
            print(f"Model saved with loss: {best_loss:.6f}")
            with torch.no_grad():
                # ProcessImageUsingModel('cuda', fileToTest, model,"Best" )
                ProcessImageUsingModel('cuda', fileToTest, model, directory, f"Epoch {epoch}_ Best True")

        else:
            
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': avg_val_loss,
                        }, directory + 'latest_spectroformer.pth')
            with torch.no_grad():
                ProcessImageUsingModel('cuda', fileToTest, model, directory, f"Epoch {epoch}_ Best False")

    def getOptimizer(self, args, learning_rate, model):
        if args.optim == "adam":
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        elif args.optim == "adamw":
            optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer: {args.optimizer}")
        return optimizer

    def evaluate(self, args, model_path='best_spectral_transformer.pth',
                 device="cuda" if torch.cuda.is_available() else "cpu"):
        from src.Models.SpectralTransformer import SpectralTransformer
        import numpy as np
        from skimage.metrics import peak_signal_noise_ratio as psnr
        from skimage.metrics import structural_similarity as ssim

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



