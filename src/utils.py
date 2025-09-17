from typing import List
import pandas as pd
import torch.nn as nn
import torch
from torchmetrics.functional import accuracy, recall, precision, auroc
import random
from PIL import Image, ImageEnhance, ImageOps, ImageFilter


def shear_x(img, magnitude):
    level = magnitude * 0.3 * random.choice([-1, 1])
    return img.transform(img.size, Image.AFFINE, (1, level, 0, 0, 1, 0))

def shear_y(img, magnitude):
    level = magnitude * 0.3 * random.choice([-1, 1])
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, level, 1, 0))

def translate_x(img, magnitude):
    max_shift = 0.3 * img.size[0]
    level = magnitude * max_shift * random.choice([-1, 1])
    return img.transform(img.size, Image.AFFINE, (1, 0, level, 0, 1, 0))

def translate_y(img, magnitude):
    max_shift = 0.3 * img.size[1]
    level = magnitude * max_shift * random.choice([-1, 1])
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, level))

def rotate(img, magnitude):
    degrees = magnitude * 30 * random.choice([-1, 1])
    return img.rotate(degrees)

def contrast(img, magnitude):
    enhancer = ImageEnhance.Contrast(img)
    factor = 1.0 + (magnitude * random.choice([-0.9, 0.9]))
    return enhancer.enhance(factor)

def brightness(img, magnitude):
    enhancer = ImageEnhance.Brightness(img)
    factor = 1.0 + (magnitude * random.choice([-0.9, 0.9]))
    return enhancer.enhance(factor)

def sharpness(img, magnitude):
    enhancer = ImageEnhance.Sharpness(img)
    factor = 1.0 + (magnitude * random.choice([-0.9, 0.9]))
    return enhancer.enhance(factor)

def equalize(img, magnitude=None):
    return ImageOps.equalize(img)

def gaussian_blur(img, magnitude):
    radius = magnitude * 2
    return img.filter(ImageFilter.GaussianBlur(radius))

def identity(img, magnitude=None):
    return img

augmentation_space = [
    shear_x, shear_y, 
    translate_x, translate_y,
    rotate, equalize, 
    contrast, brightness, 
    sharpness, identity
]

class AdaAugment:

    def __init__(self):
        self.key_transform = {}
        self.key_magnitude = {}
        self.transforms = [
            shear_x, shear_y, 
            translate_x, translate_y,
            rotate, equalize, 
            contrast, brightness, 
            sharpness, identity
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
            reduction="none"
        )
        probs = torch.exp(-bce_loss)
        F_loss = self.weights.to(target.device) * (1-probs) ** self.gamma * bce_loss

        if self.reduction == "mean":
            return F_loss.mean()
        else:
            return F_loss.sum()


class ValueMemoryEMA:
    def __init__(self, alpha):
        """
        alpha: EMA factor (0 < alpha < 1)
        """
        self.alpha = alpha
        self.values = {}  # stores EMA per key

    def __call__(self, keys, vals):
        """
        keys: list of sample identifiers
        vals: torch.Tensor of shape (len(keys), D)
        Returns: new EMA values, old stored values
        """
        stored_list = []
        new_list = []

        for i, key in enumerate(keys):
            val = vals[i]
            if key not in self.values:
                # initialize first EMA as the current value
                self.values[key] = val.clone()
                old = torch.tensor([0])
            else:
                old = self.values[key]
                # EMA update
                self.values[key] = self.alpha * old + (1 - self.alpha) * val

            stored_list.append(old.unsqueeze(0))
            new_list.append(self.values[key].unsqueeze(0))

        stored = torch.cat(stored_list, dim=0)
        new = torch.cat(new_list, dim=0)
        return new, stored

    def get(self, key):
        """
        Access the current EMA for a single key
        """
        return self.values.get(key, None)

    def get_multi(self, keys):
        """
        Access current EMA for multiple keys
        """
        return torch.stack([self.values[k] for k in keys])