from torchvision import models
import torch.nn as nn
import timm

def get_resnet18(num_classes=14):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(512, num_classes)
    return model


def get_resnet50(num_classes=14):
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    model.fc = nn.Linear(2048, out_features=num_classes)

    return model


def get_vit_tiny(num_classes=14):
    model = timm.create_model("vit_tiny_patch16_224", pretrained=True)
    model.head = nn.Linear(192, num_classes)
    return model


def get_vit_base(num_classes=14):
    model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
    model.head = nn.Linear(768, num_classes)
    return model