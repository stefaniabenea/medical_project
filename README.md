# Lung Disease Image Classification

This project implements a complete pipeline for classifying medical lung images (X-rays) into multiple categories (e.g., **Normal**, **COVID-19**, **Bacterial Pneumonia**, **Viral Pneumonia**).  
It includes both a custom CNN model and a pretrained ResNet18, an API for inference, and a Gradio web interface for easy testing.

## Project Structure

- **`train.py`** – training script with Early Stopping, learning rate scheduler, and best model saving.
- **`dataset.py`** – dataset loading with `Albumentations` for augmentation and preprocessing.
- **`model.py`** – defines the custom CNN model and loads pretrained ResNet18.
- **`utils.py`** – helper functions (plots, seeding, transformations, etc.).
- **`predict.py`** – script for predictions on a single image or a folder of images.
- **`api.py`** – FastAPI server for inference via HTTP requests.
- **`client.py`** – Python client for calling the API.
- **`gradio_demo.py`** – Gradio web interface for uploading images and getting predictions.
- **`plots/`** – training and accuracy/loss plots.
- **`models/`** – saved model files (`.pth`).


## Installation and Setup

1. **Clone the repository**  
   ```
   git clone https://github.com/user/lung-disease-classification.git
   cd lung-disease-classification
   ```

2. **Create a virtual environment and install dependencies**

```
conda create -n lung_env python=3.9
conda activate lung_env
pip install -r requirements.txt
```

3. **Dataset folder structure**

data/
    Normal/
        img1.jpg
        img2.jpg
        ...
    COVID/
        ...
    Pneumonia_Bacterial/
        ...
    Pneumonia_Viral/
        ...

##  Training

### Custom CNN
```
python train.py --model_name CNN
```
### ResNet18 (train only the final layer)
```
python train.py --model_name resnet18
```
The trained models are saved in the models/ directory.

## Predictions

### On a single image
```
python predict.py --input path/to/image.jpg --model_name CNN
```

### On a folder of images
```
python predict.py --input path/to/folder --model_name resnet18
```
## FastAPI Inference
This project includes a FastAPI-based REST API to serve the trained models (CNN or ResNet18) for inference.

1. Start the API server

Make sure you have the trained model file saved in the `models/` directory, e.g.:
models/CNN.pth
models/resnet18.pth

Run the server with:
```
# Start API with the custom CNN model
python api.py --model_name CNN

# Or with the ResNet18 model
python api.py --model_name resnet18
```
The server will start on:
http://0.0.0.0:8000

2. Test using the Python client
Use the provided client.py script to send a prediction request to the API:

```
python client.py --input path/to/image.jpg --model_name CNN
```
Example output:
Prediction for image 'path/to/image.jpg': COVID-19

## Gradio Demo
You can also launch an interactive web interface using the gradio_demo.py script:
```
python gradio_demo.py
```
This will start a local Gradio interface at: http://127.0.0.1:7860

You can upload images directly and get instant predictions.

## Observations

- Albumentations augmentations improved the model’s generalization.

- Early stopping helped prevent overfitting.

- The FastAPI server enables easy integration with external apps.

- The Gradio interface makes testing quick and user-friendly.

## Results

### Custom CNN
- Test Accuracy: 91.71%
- Train Accuracy: 84.75%

### ResNet18 (only final layer trained)
- Test Accuracy: 84.88%
- Train Accuracy: 86.94%