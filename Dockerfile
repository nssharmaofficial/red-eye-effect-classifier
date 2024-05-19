FROM ubuntu:22.04

RUN apt-get update && apt-get install -y \
    software-properties-common
RUN add-apt-repository universe
RUN apt-get update && apt-get install -y \
    git \
    python3.10 \
    python3-pip

RUN pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118

WORKDIR /red-eye-effect-classification
COPY requirements.txt /red-eye-effect-classification/requirements.txt
RUN pip3 install -r requirements.txt
