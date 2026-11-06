# Explainable ANN for Industrial Anomaly Detection and Defect Localisation

## 1. Project Overview

This project develops a Convolutional Autoencoder-based anomaly detection system for identifying and localising defects in hazelnut images.

The project uses the MVTec Anomaly Detection (MVTec AD) dataset, specifically the Hazelnut category.

The model is trained using normal hazelnut images only. During inference, it reconstructs an input image and calculates pixel-level reconstruction errors. Larger reconstruction errors indicate potentially anomalous regions.

A Streamlit application provides an interface for uploading images and viewing anomaly detection results.

## 2. Objectives

- Develop a CNN Autoencoder for unsupervised industrial anomaly detection.
- Identify defective hazelnut images using reconstruction errors.
- Localise potential defects using anomaly heatmaps.
- Evaluate the model using appropriate classification and localisation metrics.
- Investigate explainability through reconstruction-error visualisations.
- Compare the CNN Autoencoder with an Isolation Forest baseline and a pretrained CLIP foundation model.

## 3. Dataset

**Dataset:** MVTec Anomaly Detection (MVTec AD)  
**Category:** Hazelnut

The dataset contains normal and defective hazelnut images. Defect categories include:

- Crack
- Cut
- Hole
- Print

The dataset was divided into training, validation and testing subsets. Only normal images were used to train the CNN Autoencoder.

Refer to the official MVTec AD dataset documentation for dataset information and licensing.

## 4. Model Architecture

The primary model is a Convolutional Autoencoder consisting of:

- An encoder that extracts spatial image features.
- A compressed latent representation.
- A decoder that reconstructs the input image.
- ReLU activations and a final Sigmoid activation.

The input images are resized to 128 × 128 pixels and normalised to the range [0, 1].

The model uses Mean Squared Error (MSE) reconstruction loss.

For image-level classification, the anomaly score is calculated as the average reconstruction error of the highest-error 1% of pixels.

## 5. Explainability

Reconstruction-error heatmaps are used to visualise regions where the model has difficulty reconstructing an image.

The project also investigates:

- Reconstruction-error overlays.
- Predicted anomaly masks.
- Occlusion-based sensitivity analysis.
- Incorrect prediction analysis.

## 6. Foundation Model Comparison

The project evaluates the pretrained CLIP model (`openai/clip-vit-base-patch32`) using different text prompt configurations.

Its performance is compared with the CNN Autoencoder using the same validation and test datasets.

## 7. Application

The Streamlit application allows users to:

1. Upload a hazelnut image.
2. Generate a reconstructed image.
3. Calculate an anomaly score.
4. Classify the image as potentially defective or normal.
5. Visualise anomaly heatmaps.
6. View predicted defect regions.

## 8. Installation and Setup

### Requirements

- Python 3.12
- PyTorch
- Torchvision
- Streamlit
- NumPy
- Matplotlib
- Pillow

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the application

From the main project directory:

```bash
python -m streamlit run app/app.py
```

## 9. Project Structure

```text
Hazelnut_Anomaly_Detection_Coursework/
│
├── app/
│   └── app.py
│
├── src/
│   └── model.py
│
├── models/
│   └── cnn_autoencoder_experiment_1.pth
│
├── notebooks/
├── results/
├── documentation/
│
├── requirements.txt
└── README.md
```

Additional notebooks, evaluation scripts, experiment logs and results are organised in their respective folders.

## 10. Limitations

- The dataset is relatively small.
- The model may produce false positives due to natural variations in normal hazelnut images.
- Small or subtle defects may not always be detected.
- The system is an academic prototype and requires further validation before industrial deployment.
- Predictions on images outside the MVTec Hazelnut dataset may vary.

## 11. Academic Use

This project was developed as part of MSc Data Science coursework.

The MVTec AD dataset is subject to its published licence. Refer to the official dataset documentation before any redistribution or commercial use.