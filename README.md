# Red eye effect classification

The goal of the assignment is to create a classifier of human eye images which will distinguish images with the red-eye effect.

## Development environment

To build a Docker image:

```bash
docker build -t red-eye-effect-classification/python:v1 -f Dockerfile .
```

To run a Docker container:

```bash
docker run --gpus device=0 -it --entrypoint bash -p 8888:8888 -v "$(pwd)":/red-eye-effect-classification red-eye-effect-classification/python:v1
```

## Dataset

The training and test images are saved in folder `data`.

The dataset contains:

- 80 normal training images (label = 0)
- 20 red training images (label = 1)
- 50 normal test images (label = 0)
- 50 red test images (label = 1)

The `dataset.py` script handles the loading, transforming, and batching of images. Running it sets up the dataset paths, creates a custom dataset and data loader with transformations and visualizes some batches of images by saving them to the `dataset_output_images` directory.
