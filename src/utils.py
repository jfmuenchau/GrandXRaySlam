import random
from typing import List
import pandas as pd
import PIL, PIL.ImageOps, PIL.ImageEnhance, PIL.ImageDraw
import torch.nn as nn
import torch
from torchmetrics.functional import accuracy, recall, precision, auroc

def shear_x(img, magnitude):  # magnitude ∈ [-0.3, 0.3]
    return img.transform(img.size, PIL.Image.AFFINE, (1, magnitude, 0, 0, 1, 0))

def shear_y(img, magnitude):  # magnitude ∈ [-0.3, 0.3]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, magnitude, 1, 0))

def translate_x(img, magnitude):  # magnitude ∈ [-150, 150] px
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, magnitude, 0, 1, 0))

def translate_y(img, magnitude):  # magnitude ∈ [-150, 150] px
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, magnitude))

def rotate(img, magnitude):  # degrees, [-30, 30]
    return img.rotate(magnitude)

def autocontrast(img, _):  
    return PIL.ImageOps.autocontrast(img)

def invert(img, _):  
    return PIL.ImageOps.invert(img)

def equalize(img, _):  
    return PIL.ImageOps.equalize(img)

def solarize(img, magnitude):  # threshold in [0, 256]
    return PIL.ImageOps.solarize(img, magnitude)

def posterize(img, magnitude):  # bits [4, 8]
    magnitude = int(magnitude)
    return PIL.ImageOps.posterize(img, magnitude)

def contrast(img, magnitude):  # factor ∈ [0.1, 1.9]
    return PIL.ImageEnhance.Contrast(img).enhance(magnitude)

def color(img, magnitude):  # factor ∈ [0.1, 1.9]
    return PIL.ImageEnhance.Color(img).enhance(magnitude)

def brightness(img, magnitude):  # factor ∈ [0.1, 1.9]
    return PIL.ImageEnhance.Brightness(img).enhance(magnitude)

def sharpness(img, magnitude):  # factor ∈ [0.1, 1.9]
    return PIL.ImageEnhance.Sharpness(img).enhance(magnitude)


class AdaAugment:

    def __init__(self):
        self.key_transform = {}
        self.key_magnitude = {}
        self.transforms = [
            shear_x, shear_y, 
            translate_x, translate_y,
            rotate, autocontrast, 
            invert, equalize, 
            solarize, posterize, 
            contrast, color, 
            brightness, sharpness
        ]

    def set_magnitude(self, key, m):
        self.key_magnitude[key] = m
    
    def set_transform(self, key, transform_idx):
        self.key_transform[key] = transform_idx

    def __call__(self, key, img):
        return img


def calculate_metrics(prediction, ground_truth, stage):
    precision_ = precision(
        prediction, ground_truth, task="multilabel", num_labels=14
    )
    recall_ = recall(
        prediction, ground_truth, task="multilabel", num_labels=14
    )
    accuracy_ = accuracy(
        prediction, ground_truth, task="multilabel", num_labels=14
    )
    auroc_ = auroc(
        prediction, ground_truth, task="multilabel", num_labels=14
    )
    
    return {
        stage + "/precision":precision_,
        stage + "/recall":recall_,
        stage + "/accuracy":accuracy_,
        stage + "_auroc":auroc_
    }


def get_class_weights(path):
    df = pd.read_csv(path)
    weights = []
    label_columns = [
        'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Enlarged Cardiomediastinum',
        'Fracture', 'Lung Lesion', 'Lung Opacity', 'No Finding', 'Pleural Effusion',
        'Pleural Other', 'Pneumonia', 'Pneumothorax', 'Support Devices'
    ]

    for label in label_columns:
        percent = df[label].sum() / len(df)
        weights.append(percent)
    return weights

class FocalLoss(nn.Module):

    def __init__(self, weights:List=None, gamma=2.0, reduction="mean"):
        super().__init__()

        self.weights = weights
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logit, target):
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            logit, target,
            pos_weight=self.weights,
            reduction="none"
        )
        probs = torch.exp(-bce_loss)
        F_loss = (1-probs) ** self.gamma * bce_loss

        if self.reduction == "mean":
            return F_loss.mean()
        else:
            return F_loss.sum()
