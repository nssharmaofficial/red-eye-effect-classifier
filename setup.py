import os
import torch

class Setup(object):
    """
    Setup class to initialize and hold configuration settings.

    Attributes:
        root (str): The root directory of the project. Defaults to the current working directory.
        whole_size (tuple of int): The target size (height, width) to which images will be resized.
        BATCH (int): The batch size for the DataLoader.
        DEVICE (torch.device): The device to run the model on.
        LR (float): The learning rate for the optimizer.
        EPOCHS (int): The number of epochs for training the model.
    """
    def __init__(self) -> None:

        self.root = os.getcwd()
        self.whole_size = (32,32)
        self.BATCH = 8
        self.DEVICE = torch.device("cuda:0")
        self.LR = 0.001
        self.EPOCHS = 30