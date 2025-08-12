import torch
from PIL import Image
import os
from model import CNN
from utils import get_albumentations_transforms, get_model
import numpy as np
import pandas as pd
import argparse


parser = argparse.ArgumentParser(description="Script for CNN model inference")
parser.add_argument("--input", type=str, required=True, help="The path to an image or a folder of images")
parser.add_argument("--model_name", required=True, type=str, choices= ["CNN", "resnet18"], help="Choose between 'CNN' (custom model) and pretrained 'resnet18' model")

args = parser.parse_args()
model_name = args.model_name
input_path = args.input

transforms = get_albumentations_transforms(train=False, model_name=model_name)
class_names = sorted([d for d in os.listdir("data") if os.path.isdir(os.path.join("data", d))])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = get_model(model_name,len(class_names))
a = torch.load(f"models/{model_name}.pth", map_location=device)
model.load_state_dict(a)
model = model.to(device)

def predict_image(image_path, model, device, transforms):
    image = Image.open(image_path).convert("RGB")
    image = np.array(image)
    augmented_image = transforms(image=image)
    image_tensor = augmented_image['image'].unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
        prediction = torch.argmax(output,dim=1)
        return prediction.item()

def predict_folder(folder_path, model, device, transforms, class_names, csv_path="results.csv"):
    
    results = []
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    images = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(valid_exts)]
    for image_path in images:
        try:
            predicted_class = predict_image(image_path, model, device, transforms)
            results.append({"image": image_path, "prediction": class_names[predicted_class]})
        except Exception as e:
            print(f"Failed to process {image_path}: {e}")
            
    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False)
    print(f"Results saved to csv")
    return results



if os.path.isdir(input_path):
    print(f"{input_path} is processed as a folder")
    results = predict_folder(input_path, model, device, transforms, class_names)
    for r in results:
        print(f"{r['image']} is classified as {r['prediction']}")
    
else:
    print(f"{input_path} is processed as image")
    prediction = predict_image(input_path, model, device, transforms)
    print(f"The image {input_path} is classified as {class_names[prediction]}")


        





