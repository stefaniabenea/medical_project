import torch
from torchvision.datasets import ImageFolder
from utils import get_transforms, get_albumentations_transforms, AlbumentationsImageFolder
from torch.utils.data import DataLoader, random_split
import os


def prepare_data(data_dir, model_name, batch_size=32, val_ratio=0.2):
    full_dataset = AlbumentationsImageFolder(root=data_dir,transform=get_albumentations_transforms(train=True, model_name=model_name))
    total_size = len(full_dataset)
    val_size = int(total_size * val_ratio)
    train_size = total_size - val_size

    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    val_dataset.dataset.alb_transformation = get_albumentations_transforms(train=False, model_name=model_name)

    train_loader = DataLoader(dataset = train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset = val_dataset, batch_size=batch_size, shuffle=False)

    class_names = full_dataset.classes
    return train_loader, val_loader, class_names

