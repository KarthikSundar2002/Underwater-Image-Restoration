import glob
import pathlib
import pathlib as pl

import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image

# PyTorch Dataset for UIEB
class UIEBDataset(Dataset):
    def __init__(self, raw_dir, ref_dir, transform=None):
        """
        Args:
            raw_dir (str): Path to raw images.
            ref_dir (str): Path to reference images.
            transform (callable, optional): Image transformations.
        """
        self.raw_images = sorted(os.listdir(raw_dir))
        self.ref_images = sorted(os.listdir(ref_dir))
        self.raw_dir = raw_dir
        self.ref_dir = ref_dir
        self.transform = transform

    def __len__(self):
        return len(self.raw_images)

    def __getitem__(self, idx):
        raw_path = os.path.join(self.raw_dir, self.raw_images[idx])
        ref_path = os.path.join(self.ref_dir, self.ref_images[idx])

        raw_img = Image.open(raw_path).convert("RGB")
        ref_img = Image.open(ref_path).convert("RGB")

        if self.transform:
            raw_img = self.transform(raw_img)
            ref_img = self.transform(ref_img)

        return raw_img, ref_img  # Input and Ground Truth

# Function to get DataLoaders
def get_dataloaders(raw_dir, ref_dir, batch_size=16, num_workers=4):

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
    ])

    dataset = UIEBDataset(raw_dir, ref_dir, transform=transform)
    train_size = int(0.8 * len(dataset))  # 80% train, 20% test
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader


class DataManager:
    def __init__(self, fileExtension=".png"):
        self.rawDataDirectory = ""
        self.remasteredDataDirectory = ""
        self.fileExtension = fileExtension
        self.fileList = []

        rawFilePaths = glob.glob(self.rawDataDirectory + '/*/*' + self.fileExtension)
        rawFilePaths.sort()

        remasteredFilePaths = glob.glob(self.remasteredDataDirectory + '/*/*' + self.fileExtension)
        remasteredFilePaths.sort()

        if len(rawFilePaths) != len(remasteredFilePaths):
            raise Exception("Different number of raw images and remastered images")

        for i in range (len(rawFilePaths)):
            if pl.Path(remasteredFilePaths[i]).name == pl.Path(rawFilePaths[i]).name:
                self.fileList.append((rawFilePaths[i], remasteredFilePaths[i]))
            else:
                raise Exception("filename not matching at pos" + str(i))

    def download(self):
        import src.DataManipulation.DownloaderKaggle as dk
        import importlib
        importlib.reload(dk)

        downloaderReference = dk.DownloaderKaggle("larjeck/uieb-dataset-reference")
        self.remasteredDataDirectory = downloaderReference.downloadFiles()

        downloaderRaw = dk.DownloaderKaggle("larjeck/uieb-dataset-raw")
        self.rawDataDirectory = downloaderRaw.downloadFiles()

    def setDownloadedLocations(self, rawDataDirectory, remasteredDataDirectory):
        self.rawDataDirectory = rawDataDirectory
        self.remasteredDataDirectory = remasteredDataDirectory

    def preProcess(self):
        #todo: preprocess steps
        self.__resizeFiles(self.rawDataDirectory)
        self.__resizeFiles(self.remasteredDataDirectory)

    def split(self):
        ...

    def dataAugment(self):
        """
        Apply data augmentation to training images.
        Creates augmented versions of both raw and remastered images.
        """
        # Make sure to import modules at the top of the function
        import importlib
        import pathlib as pl
        import os
        import sys

        # Create augmentation directories FIRST
        raw_augmented_dir = os.path.join(os.path.dirname(self.rawDataDirectory), "augmented_raw")
        remastered_augmented_dir = os.path.join(os.path.dirname(self.remasteredDataDirectory), "augmented_remastered")

        # Then handle module imports
        # Try to reload if module exists
        if 'src.DataManipulation.DataAugmentor' in sys.modules:
            importlib.reload(sys.modules['src.DataManipulation.DataAugmentor'])

        # Import the DataAugmentor class
        from src.DataManipulation.DataAugmentor import DataAugmentor

        # Now create augmentors using the directories defined above
        print("Running Data Augmentor for Raw Images")
        raw_augmentor = DataAugmentor(
            sourceDirectory=self.rawDataDirectory,
            targetDirectory=raw_augmented_dir,
            imageFileExtension=self.fileExtension
        )

        # Create augmentor for remastered images
        print("Running Data Augmentor for remastered Images")
        remastered_augmentor = DataAugmentor(
            sourceDirectory=self.remasteredDataDirectory,
            targetDirectory=remastered_augmented_dir,
            imageFileExtension=self.fileExtension
        )

        # Apply augmentations (create 4 variants of each image)
        print("Generating augmented versions of raw images...")
        raw_augmentor.apply_augmentations(num_augmentations=4)
        raw_augmentor.save_augmented_images()

        print("Generating augmented versions of remastered images...")
        remastered_augmentor.apply_augmentations(num_augmentations=4)
        remastered_augmentor.save_augmented_images()

        print("Data augmentation completed.")

    def train(self, num_epochs=10, learning_rate=0.001, device="cuda" if torch.cuda.is_available() else "cpu"):
        """
        Train the Spectral Transformer model on UIEB dataset

        Args:
            num_epochs (int): Number of training epochs
            learning_rate (float): Learning rate for optimizer
            device (str): Device to run training on ('cuda' or 'cpu')
        """
        # Import the model
        from src.Models.SpectralTransformer import mymodel
        import torch.optim as optim
        import time
        from tqdm import tqdm

        # Set device
        print(f"Using device: {device}")

        # Get dataloaders
        print("Preparing data loaders...")
        # Use the raw and augmented directories
        raw_dir = self.rawDataDirectory
        ref_dir = self.remasteredDataDirectory

        # For additional training data, you can use augmented directories too
        raw_aug_dir = os.path.join(os.path.dirname(self.rawDataDirectory), "augmented_raw")
        ref_aug_dir = os.path.join(os.path.dirname(self.remasteredDataDirectory), "augmented_remastered")

        # Get train and test loaders using the original data
        train_loader, test_loader = get_dataloaders(raw_dir, ref_dir, batch_size=8)

        # Initialize the model
        print("Initializing model...")
        model = mymodel().to(device)

        # Define loss function and optimizer
        criterion = torch.nn.L1Loss()  # L1 loss is commonly used for image reconstruction
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

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

                # Calculate loss
                loss = criterion(outputs, ref_imgs)

                # Backward pass and optimize
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

                # Print progress every 10 batches
                if (i + 1) % 10 == 0:
                    print(f"Batch {i + 1}/{len(train_loader)}, Loss: {loss.item():.6f}")

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
                    loss = criterion(outputs, ref_imgs)
                    val_loss += loss.item()

            avg_val_loss = val_loss / len(test_loader)
            print(f"Validation Loss: {avg_val_loss:.6f}")

            # Save model if it's the best so far
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_loss,
                }, 'best_spectral_transformer.pth')
                print(f"Model saved with loss: {best_loss:.6f}")

        print("Training completed!")
        return model

    def evaluate(self, model_path='best_spectral_transformer.pth',
                 device="cuda" if torch.cuda.is_available() else "cpu"):
        """
        Evaluate the trained model on the test dataset

        Args:
            model_path (str): Path to the saved model
            device (str): Device to run evaluation on ('cuda' or 'cpu')
        """
        from src.Models.SpectralTransformer import mymodel
        import numpy as np
        from skimage.metrics import peak_signal_noise_ratio as psnr
        from skimage.metrics import structural_similarity as ssim
        import matplotlib.pyplot as plt

        # Get test dataloader
        _, test_loader = get_dataloaders(self.rawDataDirectory, self.remasteredDataDirectory, batch_size=1)

        # Load model
        model = mymodel().to(device)
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

    def __resizeFiles(self, directory):
        import src.DataManipulation.ImageManipulator as im
        import importlib
        import pathlib as pl
        importlib.reload(im)

        path = pl.Path(directory)
        outputDirectory = directory.replace(path.parts[3], "manipulated")

        imageManipulator = im.ImageManipulator(directory, outputDirectory)
        imageManipulator.resizeImages(256, 256)
        imageManipulator.saveToDisk()

        print("Resized images in " + directory)


