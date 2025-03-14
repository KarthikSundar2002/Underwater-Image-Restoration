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
        import importlib
        import pathlib as pl
        import os

        # Create augmentation directories
        raw_augmented_dir = os.path.join(os.path.dirname(self.rawDataDirectory), "augmented_raw")
        remastered_augmented_dir = os.path.join(os.path.dirname(self.remasteredDataDirectory), "augmented_remastered")

        # Import the DataAugmentor class
        # Assuming it's saved in the same directory as ImageManipulator
        from src.DataManipulation.DataAugmentor import DataAugmentor

        # Create augmentor for raw images
        raw_augmentor = DataAugmentor(
            sourceDirectory=self.rawDataDirectory,
            targetDirectory=raw_augmented_dir,
            imageFileExtension=self.fileExtension
        )

        # Create augmentor for remastered images
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

    def train(self):
        ...

    def evaluate(self):
        ...

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


