import gradio as gr
import requests
import io
from PIL import Image

API_URL = "http://localhost:8000/predict/"  

def predict_api(image, model_name):
    
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    buffered.seek(0)

    
    files = {'file': ('image.jpg', buffered, 'image/jpeg')}
    data = {'model_name': model_name}
    response = requests.post(API_URL, files=files, data=data)

    if response.status_code == 200:
        prediction = response.json().get("prediction", "No prediction")
        return prediction
    else:
        return f"Eroare API: {response.status_code}"

iface = gr.Interface(
    fn=predict_api,
    inputs=[gr.Image(type="pil", label="Input Image"), gr.Dropdown(choices=["CNN", "resnet18"], label="Model")],
    outputs="text",
    title="Lung disease image classification using API",
    description="Upload an image. The FastAPI server runs the inference."
)

iface.launch()
