#Class to download a dataset from kaggle.
#Needs api key - https://www.kaggle.com/settings/account
#Returns directory of downloaded files

import json
import os.path

import kagglehub
import shutil

class DownloaderKaggle:
    def __init__(self,
                 KaggleName,
                 pathToAuthJson = 'Secrets/Secret/kaggle.json',
                 destinationDirectory = "../data/kaggle/"):
        self._datasetName = KaggleName
        self._pathToAuthJson = pathToAuthJson
        self._destinationDirectory = destinationDirectory

    def downloadFiles(self: str) -> str:

        # Open and read the JSON auth file

        with open(self._pathToAuthJson, 'r') as file:
            data = json.load(file)
        kagglehub.auth.set_kaggle_credentials(data['username'], data['key'])

        tempPath = kagglehub.dataset_download(self._datasetName)
        destinationPath = self._destinationDirectory + self._datasetName
        if os.path.exists(destinationPath):
            shutil.rmtree(destinationPath)
        os.makedirs(destinationPath)
        dest = shutil.copytree(tempPath, destinationPath, dirs_exist_ok = True)
        print("Downloaded " + self._datasetName)
        return dest