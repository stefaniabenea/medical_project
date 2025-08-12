import requests
import argparse

def predict_image(api_url, image_path, model_name):
    with open(image_path, 'rb') as f:
        files = {
            "file": ("image.jpg", f, "image/jpeg"),
            "model_name": (None, model_name)
        }
        response = requests.post(api_url, files=files)
    if response.status_code == 200:
        result = response.json()
        print(f"Prediction for image '{image_path}': {result['prediction']}")
        return result['prediction']
    else:
        print(f"Error at request: {response.status_code}")
        print(response.text)
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Image path")
    parser.add_argument("--model_name", default="CNN", help="Model to use")

    args = parser.parse_args()
    API_URL = "http://localhost:8000/predict/"

    predict_image(API_URL, args.input, args.model_name)