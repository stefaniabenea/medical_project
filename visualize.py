from model import CNN
import torch
from utils import visualize_filters, visualize_augmentation, visualize_activations, get_albumentations_transforms, get_model
import argparse
from dataset import prepare_data

# visualize filters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description="Script for visualisations of augmentations, filters and activations")
parser.add_argument("--mode", nargs='+', type=str, choices=["augmentation", "filters", "activations"], required=True, help="Choose between these modes: augmentation, filters and activations")
parser.add_argument("--img_path", type=str, required=False, help="The image path to see the applied augmentation or activations")
parser.add_argument("--layer", type=str, required=False, help="The layer for which the filters/activations are displayed")
parser.add_argument("--model_name", required=True, type=str, choices= ["CNN", "resnet18"], help="Choose between 'CNN' (custom model) and pretrained 'resnet18' model")

args = parser.parse_args()
model_name = args.model_name
_,_, class_names = prepare_data(data_dir="data",model_name=model_name)
model = get_model(model_name,len(class_names))
a = torch.load(f"models/{model_name}.pth", map_location=device)
model.load_state_dict(a)
model = model.to(device)
model.eval()



if 'filters' in args.mode and not args.layer:
    parser.error("--layer is required for 'filters' mode")

if 'activations' in args.mode and (not args.img_path or not args.layer):
    parser.error("--img_path and --layer are required for 'activations' mode")

if 'augmentation' in args.mode and not args.img_path:
    parser.error("--img_path is required for 'augmentation' mode")

for mode in args.mode:
    if mode =='augmentation':
        img_path = args.img_path
        visualize_augmentation(img_path)
        print(f"Running mode: {mode}")
    elif mode == 'filters':
        layer = args.layer
        visualize_filters(model, layer, cols=6)
        print(f"Running mode: {mode}")
    elif mode == 'activations':
        layer=args.layer
        img_path=args.img_path
        visualize_activations(model, layer, img_path, get_albumentations_transforms(train=False,model_name=model_name), device)
        print(f"Running mode: {mode}")

#path ='D:/learning/NeuSurfaceDefect_project/test_images/scratches_268.jpg'



