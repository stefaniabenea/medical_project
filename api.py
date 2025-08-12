from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import io
import torch
import numpy as np
from utils import get_albumentations_transforms, get_model
import uvicorn
from fastapi.encoders import jsonable_encoder
import os
import argparse

app = FastAPI()
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, required=True)
args = parser.parse_args()
model_name = args.model_name

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_names = sorted([d for d in os.listdir("data") if os.path.isdir(os.path.join("data", d))])

model = get_model(model_name, num_classes=len(class_names))
model.load_state_dict(torch.load(f"models/{model_name}.pth", map_location=device))
model.to(device)
model.eval()

transform = get_albumentations_transforms(train=False, model_name=model_name)

@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_np = np.array(image)
    augmented = transform(image=image_np)
    tensor = augmented['image'].unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(tensor)
        pred_idx = torch.argmax(output, dim=1).item()

    return JSONResponse({"prediction": class_names[pred_idx]})
    

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
