import glob
import pathlib as pl
import src.DataManipulation.ImageManipulator as im
from src.DataManipulation.DataAugmentor import DataAugmentor
import os

class DataManager:
    def __init__(self, fileExtension=".png"):
        self.currentRawDataDirectory = ""
        self.currentReferenceDataDirectory = ""
        self.fileExtension = fileExtension
        self.fileList = []

        self.useExistingFiles = False

    def download(self):
        if not self.useExistingFiles:
            import src.DataManipulation.DownloaderKaggle as dk
            import importlib
            importlib.reload(dk)

            downloaderReference = dk.DownloaderKaggle("larjeck/uieb-dataset-reference")
            self.currentReferenceDataDirectory = downloaderReference.downloadFiles()

            downloaderRaw = dk.DownloaderKaggle("larjeck/uieb-dataset-raw")
            self.currentRawDataDirectory = downloaderRaw.downloadFiles()

    def setDownloadedLocations(self, rawDataDirectory, remasteredDataDirectory):

        self.currentRawDataDirectory = rawDataDirectory
        self.currentReferenceDataDirectory = remasteredDataDirectory

        rawFilePaths = glob.glob(self.currentRawDataDirectory + '/*' + self.fileExtension)

        if len(rawFilePaths) > 0:
            self.useExistingFiles = True

    def preProcess(self):
        #todo: preprocess steps
        if not self.useExistingFiles:
            self.currentRawDataDirectory = self.__resizeFiles(self.currentRawDataDirectory)
            self.currentReferenceDataDirectory = self.__resizeFiles(self.currentReferenceDataDirectory)

    def split(self):
        ...
    #todo: this would be much faster to run on GPU
    '''Accepted input directory, returns outputDirectory with augmented images'''
    def dataAugment(self):
        if not self.useExistingFiles:
            inputDirectory = self.currentRawDataDirectory
            outputDir = os.path.join(os.path.dirname(inputDirectory), "augmented_raw")

            # Now create augmentors using the directories defined above
            print("Running Data Augmentor for Raw Images")
            raw_augmentor = DataAugmentor(
                sourceDirectory=inputDirectory,
                targetDirectory= outputDir,
                imageFileExtension=self.fileExtension
            )

            print("Generating augmented versions of raw images...")
            raw_augmentor.apply_augmentations(num_augmentations=4)
            raw_augmentor.save_augmented_images()

            print("Data augmentation completed.")
            self.currentRawDataDirectory = outputDir

    def __resizeFiles(self, directory):

        outputDirectory = self.getManipulatedDir(directory)

        imageManipulator = im.ImageManipulator(directory, outputDirectory)
        imageManipulator.resizeImages(256,256)
        imageManipulator.saveToDisk()

        print("Resized images in " + directory)
        return outputDirectory

    def getManipulatedDir(self, directory):
        path = pl.Path(directory)
        outputDirectory = directory.replace(path.parts[3], "manipulated")
        return outputDirectory
