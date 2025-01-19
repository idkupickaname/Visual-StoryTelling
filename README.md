# Visual Story Telling

This repository contains code for a model that generates creative stories from sequences of images. Leveraging advanced deep learning techniques, this system utilizes PyTorch and various neural network architectures to produce high-quality narratives. The model has been evaluated using the VIST dataset and demonstrates notable performance metrics.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Technologies](#technologies)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Models](#models)
- [Results](#results)

## Overview

This project focuses on generating coherent and engaging stories from sequences of images. The system utilizes a combination of Bi-Directional GRU and reinforcement learning-based NLP models developed in PyTorch. The narrative generation is enhanced by leveraging the EfficientNet B5 and ResNet 152 architectures for image feature extraction.

## Features

- Creative story generation from image sequences
- Utilizes advanced neural network architectures: Bi-Directional GRU, EfficientNet B5, ResNet 152
- Incorporates reinforcement learning for improved narrative relevance
- High performance in narrative generation with detailed evaluation metrics

## Technologies

The project employs the following technologies and algorithms:

- PyTorch
- Convolutional Neural Networks (CNNs)
- Bi-Directional GRU
- EfficientNet B5
- ResNet 152
- Reinforcement Learning

## Installation

To get a local copy up and running, follow these simple steps:

1. Clone the repository:

    ```sh
    git clone https://github.com/idkupickaname/Visual-StoryTelling.git
    ```

2. Navigate to the project directory:

    ```sh
    cd Visual-StoryTelling
    ```

## Usage

1. Prepare the dataset and place it in the/Visual-StoryTelling/DATA directory. The dataset should be downloaded from the link provided in the [Dataset](#dataset) section.

2. Run the following command to start training of the base model (to warmstart the base model).
   
    ```sh
    python train.py --id XE --data_dir DATADIR --start_rl -1
    ```

3. Run the following command to start training of the base model + reinforcement learning model so that they get trained to work together.

    ```sh
    python train_AREL.py --id AREL --start_from_model PRETRAINED_MODEL
    ```
    
4. Run the following command to test the base model's performance.
   
    ```sh
    python train.py --option test --beam_size 3 --start_from_model data/save/XE/model.pth
    ```
    
5. Run the following command to test the base model + reinforcement learning model's performance:
   
    ```sh
    python train_AREL.py --option test --beam_size 3 --start_from_model data/save/AREL/model.pth
    ```
    
## Dataset

The project uses the VIST dataset, which includes 210,000 images and 50,000 unique stories. Download the dataset from [VIST Dataset](https://visionandlanguage.net/VIST/index.html) and preprocess it using EfficientNet B5 or the Resnet 152 pretrained CNN models.

## Models

The project employs the following models:

- **Bi-Directional GRU**: For sequential storytelling and capturing dependencies in narrative structure.
- **EfficientNet B5 and ResNet 152**: For feature extraction from images.
- **Reinforcement Learning**: To enhance narrative relevance and coherence in story generation.

## Results

The system achieves notable performance with the following evaluation metrics:

- **ROUGE Score**: 29.6
- **METEOR Score**: 35.1
- **CIDEr Score**: 9.1

These metrics indicate the effectiveness of the model in generating accurate and engaging narratives based on image sequences.
