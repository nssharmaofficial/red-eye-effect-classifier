import os
import gc
import torch
import torch.utils.data
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import _LRScheduler
from dataset import denormalize, get_paths, get_data_loader, Dataset
from matplotlib import pyplot as plt
from model import CNN
from config import Config

def calculate_accuracy(output: torch.Tensor, labels: torch.Tensor):
    """
    Calculates the accuracy of the model's predictions.

    Args:
        output (torch.Tensor): The model's output logits of shape (batch_size, 2).
        labels (torch.Tensor): The true labels of shape (batch_size, 1).

    Returns:
        float: The accuracy of the model's predictions.
    """
    _, preds = torch.max(output, dim=1)

    # Ensure labels are the same shape as preds
    labels = labels.view(-1)

    correct = torch.sum(preds == labels).item()
    accuracy = correct / labels.size(0)
    return accuracy*100

class GradualWarmupScheduler(_LRScheduler):
    """
    Gradual Warmup Scheduler class.

    Gradually increases the learning rate (LR) from a small value to a target value
    over a specified number of epochs. After the warmup period, the LR follows the
    specified after_scheduler.

    Args:
        optimizer (torch.optim.Optimizer): Wrapped optimizer.
        multiplier (float): The target LR is the initial LR multiplied by this value.
        total_epoch (int): Number of epochs for the warmup phase.

    Methods:
        get_lr: Calculates the LR for the current epoch.
        step: Updates the LR at the end of each epoch.
    """
    def __init__(self, optimizer: torch.optim.Optimizer, multiplier: float, total_epoch: int):
        self.multiplier = multiplier
        self.total_epoch = total_epoch
        self.finished = False
        super(GradualWarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        """
        Computes the learning rate for the current epoch.

        If the current epoch is within the warmup period, the learning rate is calculated as a linear
        interpolation between the initial learning rate and the target learning rate. After the warmup
        period, the learning rate is set to the target learning rate.

        Returns:
            list: List of learning rates for each parameter group.
        """
        if self.last_epoch < self.total_epoch:
            return [base_lr * (1 + self.last_epoch / self.total_epoch *
                               (self.multiplier - 1)) for base_lr in self.base_lrs]
        return [base_lr * self.multiplier for base_lr in self.base_lrs]

    def step(self, epoch=None):
        """
        Updates the learning rate.

        This method should be called at the end of each epoch to update the learning rate.
        It internally calls the _LRScheduler's step method to update the learning rate.

        Args:
            epoch (int): The current epoch number.
        """
        super(GradualWarmupScheduler, self).step(epoch)

def train(model: torch.nn.Module,
          device: torch.device,
          train_loader: DataLoader,
          criterion: torch.nn.CrossEntropyLoss,
          optimizer: torch.optim.Adam,
          epoch: int,
          total_epochs: int):
    """
    Train the model for one epoch.

    Args:
        model (nn.Module): The CNN model.
        device (torch.device): The device to run the model on.
        train_loader (DataLoader): DataLoader for the training dataset.
        criterion (nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        scheduler (_LRScheduler): Learning rate scheduler.
        epoch (int): Current epoch number.
        total_epochs (int): Total number of epochs.

    Returns:
        float: The average training loss for the epoch.
    """
    model.train()
    training_batch_losses = []
    for i, batch in enumerate(train_loader):
        imgs, labels = batch[0].to(device), batch[1].to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels.view(-1))
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            acc = calculate_accuracy(outputs, labels)
            print("T_Epoch: [%d/%d], Step: [%d/%d], Loss: %.3f, Acc: %.3f "
                  % (epoch + 1, total_epochs, i, len(train_loader), loss.item(), acc))

        training_batch_losses.append(loss.item())

    avg_train_loss = sum(training_batch_losses) / len(training_batch_losses)
    return avg_train_loss

def evaluate(model: torch.nn.Module,
             device: torch.device,
             val_loader: DataLoader,
             criterion: torch.nn.CrossEntropyLoss,
             epoch: int,
             total_epochs: int):
    """
    Evaluate the model.

    Args:
        model (nn.Module): The CNN model.
        device (torch.device): The device to run the model on.
        val_loader (DataLoader): DataLoader for the validation dataset.
        criterion (nn.Module): Loss function.
        epoch (int): Current epoch number.
        total_epochs (int): Total number of epochs.

    Returns:
        float: The average validation loss for the epoch.
        list of tuples: List of (image, true label, predicted label) for visualization.
    """
    model.eval()
    val_batch_losses = []
    val_images = []
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            imgs, labels = batch[0].to(device), batch[1].to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels.view(-1))
            if i % 10 == 0:
                acc = calculate_accuracy(outputs, labels)
                print("V_Epoch: [%d/%d], Step: [%d/%d], Loss: %.3f, Acc: %.3f "
                      % (epoch + 1, total_epochs, i, len(val_loader), loss.item(), acc))

            val_batch_losses.append(loss.item())

            # Collect images and labels for visualization
            _, preds = torch.max(outputs, 1)
            for img, label, pred in zip(imgs.cpu(), labels.cpu(), preds.cpu()):
                val_images.append((img, label.item(), pred.item()))

    avg_val_loss = sum(val_batch_losses) / len(val_batch_losses)
    return avg_val_loss, val_images

def plot_and_save_images(images: torch.Tensor, output_dir: str, grid_size=(5, 10)):
    """
    Plot and save images in a grid with true and predicted labels.

    Args:
        images (list of tuples): List of (image, true label, predicted label).
        output_dir (str): Directory to save the plot.
        grid_size (tuple of int): Grid size for plotting (rows, columns).
    """
    fig, axes = plt.subplots(*grid_size, figsize=(15, 15))
    axes = axes.flatten()
    for ax, (img, true_label, pred_label) in zip(axes, images):
        img = denormalize(img)
        color = 'green' if true_label == pred_label else 'red'
        ax.imshow(img)
        ax.set_title(f"True: {true_label}, Pred: {pred_label}", color=color)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'CNN-B{config.BATCH}-L-{config.LR}-E{config.EPOCHS}-validation_images.png'))
    plt.close()

if __name__ == '__main__':
    """
    This script performs the following steps:
    1. Initializes the setup and device configuration.
    2. Loads the training and validation datasets and creates DataLoaders.
    3. Sets up the CNN model, loss function, optimizer, and learning rate scheduler.
    4. Trains the model for a specified number of epochs, evaluates it, and prints the
       learning rate, training, and validation metrics.
    5. Saves the model checkpoints and plots the training and validation losses.
    6. Plots and saves the validation images with true and predicted labels at the end of training.
    """
    config = Config()
    device = config.DEVICE

    checkpoints_dir = "checkpoints"
    plots_dir = "training_output_images"
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    print('Loading dataset...')
    normal_train_paths, red_train_paths, normal_test_paths, red_test_paths = get_paths()
    train_dataset = Dataset(red_train_paths, normal_train_paths)
    train_loader = get_data_loader(train_dataset, batch_size=config.BATCH)
    val_dataset = Dataset(red_test_paths, normal_test_paths)
    val_loader = get_data_loader(val_dataset, batch_size=config.BATCH)

    print('Setting up the model...')
    cnn = CNN().to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=list(cnn.parameters()), lr=config.WARMUP_LR)
    scheduler = GradualWarmupScheduler(optimizer,
                                       multiplier=config.LR / config.WARMUP_LR,
                                       total_epoch=config.WARMUP_EPOCHS)

    print("Beginning training...")
    training_losses, val_losses = [], []
    all_val_images = []

    for epoch in range(config.EPOCHS):
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch {epoch+1}/{config.EPOCHS} started. Current learning rate: {current_lr:.4f}')

        avg_train_loss = train(cnn, device, train_loader, criterion, optimizer, epoch, config.EPOCHS)
        avg_val_loss, val_images = evaluate(cnn, device, val_loader, criterion, epoch, config.EPOCHS)

        training_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        if epoch+1 == config.EPOCHS:
            all_val_images.extend(val_images)

        torch.cuda.empty_cache()
        gc.collect()

        scheduler.step()

        # save model after every epoch
        torch.save(cnn.state_dict(), f"{checkpoints_dir}/CNN-B{config.BATCH}-LR-{config.LR}-E{epoch+1}.pt")

    plt.plot(training_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('CNN Training')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f"{plots_dir}/CNN-B{config.BATCH}-L-{config.LR}-E{config.EPOCHS}.png")

    # Plot and save validation images with true and predicted labels
    plot_and_save_images(all_val_images, plots_dir, grid_size=(10, 10))