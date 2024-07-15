# Potato Disease Classification using Deep Learning

This project focuses on developing a deep-learning model to classify potato diseases using images. The model is built using TensorFlow and Keras, leveraging the [plantVillage dataset](https://www.kaggle.com/datasets/arjuntejaswi/plant-village).

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)

## Overview

The aim of this project is to accurately classify potato diseases (Early or Late blight) from images using a convolutional neural network (CNN). The project involves data preprocessing, model building, training, evaluation and visualisation.

## Dataset

The dataset used in this project is the [plantVillage dataset](https://www.kaggle.com/datasets/arjuntejaswi/plant-village), which contains labelled images of various potato diseases. The dataset is divided into training, validation, and test sets.

## Preprocessing the data

Preprocessing steps include:
- Resizing images to a standard size.
- Normalizing pixel values.
- Applying data augmentation techniques like random flipping and rotation to increase the diversity of the training data.

## Model Architecture

The model is a convolutional neural network (CNN) built using TensorFlow and Keras. It consists of:
- Multiple convolutional layers for feature extraction.
- MaxPooling layers for downsampling.
- Fully connected layers for classification.

## Training

The model is trained on the preprocessed dataset with the following settings:
- Image size: 256x256 pixels
- Batch size: 32
- Epochs: 50
- Optimizer: Adam
- Loss function: Sparse Categorical Crossentropy

## Evaluation

The model's performance is evaluated using the validation and test sets. Key metrics include accuracy and loss, visualized through training and validation curves.


## Results

The model achieves significant accuracy in classifying potato diseases, demonstrating the effectiveness of data augmentation and model tuning. Visualisations of predictions and confidence levels are provided.
