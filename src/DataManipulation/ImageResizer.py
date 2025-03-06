import os
import numpy as np
import cv2
import glob

#warning, uses fixed heigh and width, does not maintain aspect ratio, does not add padding
#todo: provide options for other resizing methods
class ImageResizer:
    def __init__(self, originalDirectory, resizedDirectory, height, width):
        self.originalDirectory = originalDirectory
        self.resizedDirectory = resizedDirectory
        self.height = height
        self.width = width

    def processFiles(self):
        if not os.path.exists(self.resizedDirectory):
            os.makedirs(self.resizedDirectory)

        files = glob.glob(self.originalDirectory +'/*/*.png')
        number_of_channels = 3
        for file in files:
            img = cv2.imread(file)
            img_array = np.array(img)
            resizedImg = cv2.resize(img_array, (self.height, self.width))
            cv2.imwrite(self.resizedDirectory + '/' + os.path.basename(file), resizedImg)
            print(img.shape)

        return