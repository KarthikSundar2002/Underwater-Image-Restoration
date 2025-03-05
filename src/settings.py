#single place to manage the key settings
class Settings:
    def __init__(self):

        self._dataDirectory = "./Data/"
        self._batchDirectory =  "Batched/"

    def dataDirectory(self: str) -> str:
        return self._dataDirectory

    def batchDirectory(self: str) -> str:
        return self._dataDirectory + self._batchDirectory
