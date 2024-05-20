import os
import torch

class Config(object):
    """
    Config class to initialize and hold configuration settings.

    Attributes:
        root (str): The root directory of the project. Defaults to the current working directory.
        BATCH (int): The batch size for the DataLoader.
        DEVICE (torch.device): The device to run the model on.
        LR (float): The target learning rate for the optimizer.
        WARMUP_LR (float): The beggining learning rate which will be gradually increased to LR
        WARMUP_EPOCHS (int): The number of epochs for gradually increasing the learning rate.
        EPOCHS (int): The number of epochs for training the model.
    """
    def __init__(self) -> None:

        self.root = os.getcwd()
        self.BATCH = 8
        self.DEVICE = torch.device("cuda:0")
        self.LR = 0.01
        self.WARMUP_LR = self.LR*0.1
        self.WARMUP_EPOCHS = 5
        self.EPOCHS = 30