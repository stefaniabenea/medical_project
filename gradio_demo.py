import gradio as gr
import requests
import io
from PIL import Image

API_URL = "http://localhost:8000/predict/"  

def predict_api(image):
    
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    buffered.seek(0)

    
    files = {'file': ('image.jpg', buffered, 'image/jpeg')}
    response = requests.post(API_URL, files=files)

    if response.status_code == 200:
        prediction = response.json().get("prediction", "No prediction")
        return prediction
    else:
        return f"Eroare API: {response.status_code}"

iface = gr.Interface(
    fn=predict_api,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Lung disease image classification using API",
    description="Upload an image. The FastAPI server runs the inference."
)

iface.launch()
