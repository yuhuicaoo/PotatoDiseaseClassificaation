# Plant Disease Classification using Deep Learning

This project focuses on developing a deep learning model to classify plant diseases using images. The model is built using TensorFlow and Keras, leveraging the PlantVillage dataset.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Usage](#usage)
- [Results](#results)

## Overview

The aim of this project is to accurately classify plant diseases from images using a convolutional neural network (CNN). The project involves data preprocessing, model building, training, and evaluation.

## Dataset

The dataset used in this project is the <a href="https://www.kaggle.com/datasets/arjuntejaswi/plant-village" target="_blank">PlantVillage dataset</a>, which contains labelled images of various plant diseases. The dataset is divided into training, validation, and test sets.

## Preprocessing

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

## Usage

To use the model for predictions:
1. Load the saved model.
2. Preprocess the input image.
3. Use the model to predict the class of the image.

## Results

The model achieves significant accuracy in classifying plant diseases, demonstrating the effectiveness of data augmentation and model tuning. Visualizations of predictions and confidence levels are provided.
