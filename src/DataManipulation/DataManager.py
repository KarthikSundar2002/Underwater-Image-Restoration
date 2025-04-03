import glob
import pathlib as pl
import src.DataManipulation.ImageManipulator as im
from src.DataManipulation.DataAugmentor import DataAugmentor
import os

class DataManager:
    def __init__(self, fileExtension=".png"):
        self.currentRawDataDirectory = ""
        self.referenceDataDirectory = ""
        self.fileExtension = fileExtension
        self.fileList = []

        #rawFilePaths = glob.glob(self.currentRawDataDirectory + '/*/*' + self.fileExtension)
        #rawFilePaths.sort()

        #remasteredFilePaths = glob.glob(self.referenceDataDirectory + '/*/*' + self.fileExtension)
        #remasteredFilePaths.sort()

        # if len(rawFilePaths) != len(remasteredFilePaths):
        #     raise Exception("Different number of raw images and remastered images")
        #
        # for i in range (len(rawFilePaths)):
        #     if pl.Path(remasteredFilePaths[i]).name == pl.Path(rawFilePaths[i]).name:
        #         self.fileList.append((rawFilePaths[i], remasteredFilePaths[i]))
        #     else:
        #         raise Exception("filename not matching at pos" + str(i))

    def download(self):
        import src.DataManipulation.DownloaderKaggle as dk
        import importlib
        importlib.reload(dk)

        downloaderReference = dk.DownloaderKaggle("larjeck/uieb-dataset-reference")
        self.referenceDataDirectory = downloaderReference.downloadFiles()

        downloaderRaw = dk.DownloaderKaggle("larjeck/uieb-dataset-raw")
        self.currentRawDataDirectory = downloaderRaw.downloadFiles()

    #todo: get rid of this method - add the fields to class init
    def setDownloadedLocations(self, rawDataDirectory, remasteredDataDirectory):
        self.currentRawDataDirectory = rawDataDirectory
        self.referenceDataDirectory = remasteredDataDirectory

    def preProcess(self):
        #todo: preprocess steps
        self.currentRawDataDirectory = self.__resizeFiles(self.currentRawDataDirectory)
        self.referenceDataDirectory = self.__resizeFiles(self.referenceDataDirectory)

    def split(self):
        ...
    #todo: this would be much faster to run on GPU
    '''Accepted input directory, returns outputDirectory with augmented images'''
    def dataAugment(self):
        inputDirectory = self.currentRawDataDirectory
        outputDir = os.path.join(os.path.dirname(inputDirectory), "augmented_raw")

        # Now create augmentors using the directories defined above
        print("Running Data Augmentor for Raw Images")
        raw_augmentor = DataAugmentor(
            sourceDirectory=inputDirectory,
            targetDirectory= outputDir,
            imageFileExtension=self.fileExtension
        )

        # Apply augmentations (create 4 variants of each image)
        print("Generating augmented versions of raw images...")
        raw_augmentor.apply_augmentations(num_augmentations=4)
        raw_augmentor.save_augmented_images()

        print("Data augmentation completed.")
        self.currentRawDataDirectory = outputDir

    def __resizeFiles(self, directory):

        outputDirectory = self.getManipulatedDir(directory)

        imageManipulator = im.ImageManipulator(directory, outputDirectory)
        imageManipulator.resizeImages(256, 256)
        imageManipulator.saveToDisk()

        print("Resized images in " + directory)
        return outputDirectory

    def getManipulatedDir(self, directory):
        path = pl.Path(directory)
        outputDirectory = directory.replace(path.parts[3], "manipulated")
        return outputDirectory
