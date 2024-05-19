import os
import torch

class Setup(object):
    """
    Setup class to initialize and hold configuration settings.

    Attributes:
        root (str): The root directory of the project. Defaults to the current working directory.
        whole_size (tuple of int): The target size (height, width) to which images will be resized.
        BATCH (int): The batch size for the DataLoader.
    """
    def __init__(self) -> None:

        self.root = os.getcwd()
        self.whole_size = (32,32)
        self.BATCH = 8