import torch.nn as nn
import torch.nn.functional as F
from dataset import get_paths, get_data_loader, Dataset
from config import Config


class CNN(nn.Module):
    """
    Convolutional Neural Network (CNN) for classifying 'normal' and 'red' eye images.

    The network consists of four convolutional layers followed by two fully connected layers.
    Each convolutional layer is followed by batch normalization and a LeakyReLU activation function.
    A dropout layer is added before the final fully connected layer to prevent overfitting.

    Attributes:
        conv1 (nn.Sequential): First convolutional layer block.
        conv2 (nn.Sequential): Second convolutional layer block.
        conv3 (nn.Sequential): Third convolutional layer block.
        conv4 (nn.Sequential): Fourth convolutional layer block.
        fc1 (nn.Linear): First fully connected layer.
        fc2 (nn.Linear): Second fully connected layer (output layer).
        dropout (nn.Dropout): Dropout layer with a probability of 0.5.
    """
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 8, 4, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, 4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.fc1 = nn.Linear(64 * 2 * 2, 32)
        self.fc2 = nn.Linear(32, 2) 
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        """
        Defines the forward pass of the CNN.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, 32, 32).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 2).
        """
        # print('\nOriginal: ', x.size())
        x = self.conv1(x)
        # print('Conv1: ', x.size())
        x = self.conv2(x)
        # print('Conv2: ', x.size())
        x = self.conv3(x)
        # print('Conv3: ', x.size())
        x = self.conv4(x)
        # print('Conv4: ', x.size())

        x = x.view(x.size(0), -1)
        # print('OutConv: ', x.size())

        x = F.leaky_relu(self.fc1(x))
        # print('Lin1: ', x.size())
        x = self.dropout(x)
        x = self.fc2(x)
        # print('Lin2: ', x.size())
        return x

if __name__ == '__main__':
    """
    Main script to initialize the setup, load datasets, create DataLoader,
    instantiate the CNN model, and display the number of trainable parameters 
    and the output size for a batch of images.
    """

    config = Config()

    normal_train_paths, red_train_paths, normal_test_paths, red_test_paths = get_paths()

    train_dataset = Dataset(red_train_paths, normal_train_paths, type="train")
    train_loader = get_data_loader(train_dataset, batch_size=config.BATCH)

    imgs, labels = next(iter(train_loader))

    cnn = CNN()
    print(f'Number of trainable parameters in CNN: {sum(p.numel() for p in cnn.parameters() if p.requires_grad)}')
    output = cnn.forward(imgs)

    # Print info
    print('\nBatch size: ', config.BATCH)
    print('Images size: ', imgs.size())         # (batch, 3, 32, 32)
    print('CNN output size: ', output.size())   # (batch, 2)
