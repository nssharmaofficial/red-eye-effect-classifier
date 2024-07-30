import os
import matplotlib.pyplot as plt
import torch
import torch.utils.data as data
from PIL import Image
from config import Config
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torch.utils.data.sampler import WeightedRandomSampler


def get_paths():
    """
    Retrieves file paths for training and test datasets for 'normal' and 'red' eye images.

    This function navigates through the directory structure defined in the Config object to
    gather the file paths for the images used in training and testing the model. It segregates
    the images into 'normal' and 'red' categories for both training and test datasets.

    Returns:
        tuple: Four lists containing the paths for:
            - normal_train_paths (list of str): Paths to 'normal' eye training images.
            - red_train_paths (list of str): Paths to 'red' eye training images.
            - normal_test_paths (list of str): Paths to 'normal' eye test images.
            - red_test_paths (list of str): Paths to 'red' eye test images.
    """
    config = Config()

    normal_train_paths = []
    red_train_paths = []
    normal_test_paths = []
    red_test_paths = []

    normal_train_folder = os.path.join(config.root, 'data', 'train', 'normal')
    red_train_folder = os.path.join(config.root, 'data', 'train', 'red')
    normal_test_folder = os.path.join(config.root, 'data', 'test', 'normal')
    red_test_folder = os.path.join(config.root, 'data', 'test', 'red')

    # get training set
    for img_name in os.listdir(normal_train_folder):
        img_path = os.path.join(normal_train_folder, img_name)
        normal_train_paths.append(img_path)
    for img_name in os.listdir(red_train_folder):
        img_path = os.path.join(red_train_folder, img_name)
        red_train_paths.append(img_path)

    # get test set
    for img_name in os.listdir(normal_test_folder):
        img_path = os.path.join(normal_test_folder, img_name)
        normal_test_paths.append(img_path)
    for img_name in os.listdir(red_test_folder):
        img_path = os.path.join(red_test_folder, img_name)
        red_test_paths.append(img_path)

    return normal_train_paths, red_train_paths, normal_test_paths, red_test_paths


class Dataset(data.Dataset):
    """
    Custom dataset class for loading and transforming images for the 'normal'
    and 'red' eye classification task.

    This class extends the PyTorch Dataset class and is used to handle the loading and
    preprocessing of images. It supports data augmentation through transformations such
    as resizing, random horizontal flips, random rotations, and random resized crops.

    Args:
        red_paths (list of str): List of file paths for 'red' eye images.
        normal_paths (list of str): List of file paths for 'normal' eye images.
        type (str): Specifies whether the dataset is for training or evaluation. 
                Should be either 'train' or 'val'.
    """

    def __init__(self, red_paths: list[str],
                 normal_paths: list[str],
                 type: str["train", "val"]):

        if type == "train":
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((32,32)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(25),
                transforms.RandomResizedCrop((32,32), scale=(0.8, 1.0)),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        elif type == "val":
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((32,32)),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        self.normal_paths = normal_paths
        self.red_paths = red_paths
        self.labels = [0] * len(normal_paths) + [1] * len(red_paths)
        self.paths = normal_paths + red_paths

    def __len__(self):
        """
        Returns the total number of images in the dataset.

        Returns:
            int: Total number of images.
        """
        return self.paths

    def __getitem__(self, idx):
        """
        Retrieves the image and label at the specified index.

        This method loads an image from the file system, applies the defined
        transformations, and returns the transformed image along with its label.

        Args:
            idx (int): Index of the image to retrieve.

        Returns:
            tuple: A tuple containing:
                - image (torch.Tensor): The transformed image tensor.
                - label (int): The label corresponding to the image.
        """
        img_path = self.paths[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)

        return image, label

def denormalize(image: torch.Tensor):
    """
    Denormalizes an image tensor and converts it to a numpy array for visualization.

    This function reverses the normalization applied during the transformation to
    convert the image tensor back to a format suitable for visualization with matplotlib.

    Args:
        image (torch.Tensor): Normalized image tensor.

    Returns:
        numpy.ndarray: Denormalized image array.
    """
    inv_normalize = transforms.Normalize(
        mean=[-0.5 / 0.5, -0.5 / 0.5, -0.5 / 0.5],
        std=[1 / 0.5, 1 / 0.5, 1 / 0.5]
    )
    return (inv_normalize(image) * 255.).type(torch.uint8).permute(1, 2, 0).numpy()

def get_data_loader(dataset: Dataset, batch_size: int):
    """
    Creates a DataLoader with a WeightedRandomSampler to handle class imbalance.

    This function calculates the weights for balancing the classes and creates a DataLoader
    that uses a WeightedRandomSampler to ensure that each class is represented proportionally
    in each batch.

    Args:
        dataset (Dataset): Custom dataset object containing images and labels.
        batch_size (int): Number of samples per batch.

    Returns:
        DataLoader: DataLoader object with balanced sampling.
    """
    # Calculate weights for balancing the classes
    class_counts = [dataset.labels.count(0), dataset.labels.count(1)]
    class_weights = [1.0 / class_count for class_count in class_counts]
    sample_weights = [class_weights[label] for label in dataset.labels]

    # This sampler ensures that each class is represented proportionally
    # in each batch by assigning higher sampling weights to the minority class.
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=3,
        pin_memory=True,
        shuffle=False  # Sampler already handles shuffling
    )

if __name__ == '__main__':
    """
    Main function to load datasets, create DataLoader, and save batches of images for visualization.

    This script initializes the setup, retrieves the dataset paths, creates the dataset and
    DataLoader objects, and then iterates through the DataLoader to save batches of images
    for visualization.
    """
    config = Config()

    normal_train_paths, red_train_paths, normal_test_paths, red_test_paths = get_paths()

    print('Size of normal training set: ', len(normal_train_paths))
    print('Size of red training set: ', len(red_train_paths))
    print('Size of normal test set: ', len(normal_test_paths))
    print('Size of red test set: ', len(red_test_paths))

    train_dataset = Dataset(red_train_paths, normal_train_paths, type="train")
    train_loader = get_data_loader(train_dataset, batch_size=config.BATCH)

    imgs, labels = next(iter(train_loader))

    print(f'Size of reds after concatenation: {imgs.size()}')

    output_dir = "dataset_output_images"
    os.makedirs(output_dir, exist_ok=True)

    for i in range(10):  # Display 5 batches
        img, label = next(iter(train_loader))
        print(f'Normal: {sum(label==0).item()}, Red: {sum(label==1).item()}')
        for idx, (img, label) in enumerate(zip(imgs, labels)):
            img = denormalize(img)
            plt.imshow(img)
            plt.title(f'label: {label}')
            plt.savefig(os.path.join(output_dir, f"batch_{i}_image_{idx}.png"))
            plt.close()