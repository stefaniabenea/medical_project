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


app = FastAPI()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_names = sorted([d for d in os.listdir("data") if os.path.isdir(os.path.join("data", d))])

cnn_model = get_model("CNN", num_classes=len(class_names))
cnn_model.load_state_dict(torch.load("models/CNN.pth", map_location=device))
cnn_model.to(device)
cnn_model.eval()

resnet_model = get_model("resnet18", num_classes=len(class_names))
resnet_model.load_state_dict(torch.load("models/resnet18.pth", map_location=device))
resnet_model.to(device)
resnet_model.eval()



@app.post("/")
@app.post("/predict/")
async def predict_image(file: UploadFile = File(...), model_name: str = "CNN"):
    if model_name == "CNN":
        selected_model = cnn_model
    elif model_name == "resnet18":
        selected_model = resnet_model
    else:
        return JSONResponse({"error": "Invalid model name. Choose 'CNN' or 'resnet18'."}, status_code=400)   
    transform = get_albumentations_transforms(train=False, model_name=selected_model)
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_np = np.array(image)
    augmented = transform(image=image_np)
    tensor = augmented['image'].unsqueeze(0).to(device)
    with torch.no_grad():
        output = selected_model(tensor)
        pred_idx = torch.argmax(output, dim=1).item()

    return JSONResponse({"prediction": class_names[pred_idx]})
    

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
