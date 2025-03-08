import glob
import pathlib
import pathlib as pl


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


