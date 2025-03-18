# Image-Classifier-Application

#### Context

AI algorithms are being incorporated more and more into everyday applications. In this project, a deep learning model is trained to recognize and classify different species of flowers, then exported for use in an application.

This project implements an image classifier capable of recognizing different species of flowers using deep learning techniques. The classifier is built with PyTorch and can be used as a command-line application.

#### Project Overview

The **Image Classifier Application** utilizes a deep learning model to classify images of flowers into various species. The project involves:

- Loading and preprocessing the flower dataset.
- Training a Convolutional Neural Network (CNN) using PyTorch.
- Evaluating the model's performance.
- Deploying the trained model for image classification through a command-line interface.

#### Data Sources

We will be using a dataset of 102 flower categories provided by Udacity, as put together by the Department of Engineering Science, University of Oxford.

#### Project Structure

In this project, I first implemented an image classifier using PyTorch in a Jupyter notebook. After building and training a deep neural network on the flower dataset, I transformed it into a command-line application for broader usability. The application consists of two main Python scripts:
- `train.py`: Trains a new network on a dataset and saves the model as a checkpoint.
- `predict.py`: Uses a trained network to predict the class of an input image.
