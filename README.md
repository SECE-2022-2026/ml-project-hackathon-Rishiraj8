[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/KPIpT6T5)
# Crop Classification Web App using Flask and PyTorch

This project is a web application for classifying agricultural crops using a deep learning model built with PyTorch. The web app is powered by Flask and allows users to upload an image of a crop, which is then classified by a pre-trained model. The model uses a convolutional neural network (CNN) based on the ResNet-18 architecture.

## Features

- Upload an image of a crop.
- Classify the crop based on a pre-trained model.
- Display the predicted crop label and the confidence score.
- Allows uploading and viewing the image with the prediction.

## Table of Contents

- [Project Overview](#project-overview)
- [Setup Instructions](#setup-instructions)
- [Dependencies](#dependencies)
- [Running the Application](#running-the-application)
- [Model](#model)
- [License](#license)

## Project Overview

This Flask web application uses a trained PyTorch model to classify images of agricultural crops. The model is based on ResNet-18, a popular convolutional neural network architecture, and has been fine-tuned for crop classification tasks.

### Workflow

1. Users upload an image via the web interface.
2. The server processes the image and applies necessary transformations.
3. The image is passed through the trained ResNet-18 model.
4. The model predicts the class of the crop, and the prediction with confidence score is returned to the user.

## Setup Instructions

Follow these steps to set up and run the project on your local machine.

### 1. Clone the Repository

```bash
git clone <repository-url>
