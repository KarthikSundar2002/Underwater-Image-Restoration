import os
import numpy as np
import cv2
import glob

#will hold the images in memory, use with a WITH statement to avoid memory issues
#for this reason will not be most efficient on larger datasets
#better approach would be to queue operations
class ImageManipulator:
    def __init__(self, sourceDirectory, targetDirectory, imageFileExtension = ".png"):
        self.sourceDirectory = sourceDirectory
        self.targetDirectory = targetDirectory
        self.imageFileExtension = imageFileExtension

        #dictionary of file path, image data to allow multiple operations before saving
        self.images = {}
        filePaths = glob.glob(self.sourceDirectory + '/*/*' + self.imageFileExtension)
        for filePath in filePaths:
            self.images.update({filePath: cv2.imread(filePath)})

    #resize the images from the source directory and save in the target directory, defaulting to 256 x 256
    def resizeImages(self, height = 256, width = 256):
        for filePath, img in self.images.items():
            img_array = np.array(img)
            self.images.update({filePath: cv2.resize(img_array, (height,width), interpolation=cv2.INTER_CUBIC)})
        return

    def saveToDisk(self):
        if not os.path.exists(self.targetDirectory):
            os.makedirs(self.targetDirectory)

        for filePath, img in self.images.items():
            cv2.imwrite(self.targetDirectory + '/' + os.path.basename(filePath), img)


