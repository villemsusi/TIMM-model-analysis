from pathlib import Path
import random
import os

import numpy as np

import pandas as pd

from PIL import Image

import timm
from timm.models import resnet

# Import PyTorch dependencies
import torch
from torch import nn
from torchvision import transforms

class InferenceWrapper(nn.Module):
    def __init__(self, model, normalize_mean, normalize_std, scale_inp=False, channels_first=False):
        super().__init__()
        self.model = model
        self.register_buffer("normalize_mean", normalize_mean)
        self.register_buffer("normalize_std", normalize_std)
        self.scale_inp = scale_inp
        self.channels_first = channels_first
        self.softmax = nn.Softmax(dim=1)

    def preprocess_input(self, x):
        if self.scale_inp:
            x = x / 255.0

        if self.channels_first:
            x = x.permute(0, 3, 1, 2)

        x = (x - self.normalize_mean) / self.normalize_std
        return x

    def forward(self, x):
        x = self.preprocess_input(x)
        x = self.model(x)
        x = self.softmax(x)
        return x

def predict(target, input):
    test_img = Image.open(input)
    #test_img.show()
    
    target_cls = target

    infer_sz = 288
    inp_img = test_img.resize((infer_sz, infer_sz))
    #inp_img.show()

    img_tensor = transforms.ToTensor()(inp_img)[None].to(device)

    with torch.no_grad():
        pred_scores = wrapped_model(img_tensor)

    confidence_score = pred_scores.max()

    pred_class = class_names[torch.argmax(pred_scores)]

    pred_data = pd.Series({
        "Input Size:": inp_img.size,
        "Target Class:": target_cls,
        "Predicted Class:": pred_class,
        "Confidence Score:": f"{confidence_score*100:.2f}%"
    })
    return pred_data
    

if __name__ == "__main__":
    device = 'cpu'
    dtype = torch.float32
    
    checkpoint_dir = Path("PetClassifier/2023-12-06_03-19-36/")
    checkpoint_path = checkpoint_dir/"resnet18d.pth"

    class_names = ['Abyssinian', 'american_bulldog', 'american_pit_bull_terrier', 'basset_hound', 'beagle', 'Bengal', 'Birman', 'Bombay', 'boxer', 'British_Shorthair', 'chihuahua', 'Egyptian_Mau', 'english_cocker_spaniel', 'english_setter', 'german_shorthaired', 'great_pyrenees', 'havanese', 'japanese_chin', 'keeshond', 'leonberger', 'Maine_Coon', 'miniature_pinscher', 'newfoundland', 'Persian', 'pomeranian', 'pug', 'Ragdoll', 'Russian_Blue', 'saint_bernard', 'samoyed', 'scottish_terrier', 'shiba_inu', 'Siamese', 'Sphynx', 'staffordshire_bull_terrier', 'wheaten_terrier', 'yorkshire_terrier']

    resnet_model = 'resnet18d'
    model_cfg = resnet.default_cfgs[resnet_model].default.to_dict()

    resnet18 = timm.create_model(resnet_model, num_classes=len(class_names))
    resnet18 = resnet18.to(device=device, dtype=dtype).eval()
    resnet18.device = device

    model = resnet18
    model.load_state_dict(torch.load(checkpoint_path));

    mean, std = model_cfg['mean'], model_cfg['std']
    norm_stats = (mean, std)

    normalize_mean = torch.tensor(norm_stats[0]).view(1, 3, 1, 1)
    normalize_std = torch.tensor(norm_stats[1]).view(1, 3, 1, 1)

    wrapped_model = InferenceWrapper(model, normalize_mean, normalize_std).to(device=device)
    wrapped_model.eval()

    res = []
    direc = "oxford-iiit-pet/images/test/"
    for label in os.listdir(direc):
        for img in os.listdir(f"{direc}{label}/"):
            res.append(predict(label, f"{direc}{label}/{img}"))

    res = pd.DataFrame(res)
    res.to_csv("predicts.csv")

    print(res)
    

    