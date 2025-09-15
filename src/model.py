from torchvision import models
import torch.nn as nn
import timm


def freeze_all(model):
    for param in model.parameters():
        param.requires_grad=False

def get_resnet18(num_classes=14, fine_tune=True):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    if fine_tune:
        freeze_all(model)                # freeze all layers
        for param in model.layer4.parameters():
            param.requires_grad = True   # unfreeze last block

    model.fc = nn.Linear(512, num_classes)  # replace classifier
    return model


def get_resnet50(num_classes=14, fine_tune=True):
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    if fine_tune:
        freeze_all(model)
        for param in model.layer4.parameters():
            param.requires_grad = True   # unfreeze last block

    model.fc = nn.Linear(2048, num_classes)
    return model


def get_vit_tiny(num_classes=14, fine_tune=True):
    model = timm.create_model("vit_tiny_patch16_224", pretrained=True)

    if fine_tune:
        freeze_all(model)
        for param in model.blocks[-2:].parameters():
            param.requires_grad = True   # unfreeze last 2 transformer blocks

    model.head = nn.Linear(192, num_classes)
    return model


def get_vit_base(num_classes=14, fine_tune=True):
    model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)

    if fine_tune:
        freeze_all(model)
        for param in model.encoder.layers[-2:].parameters():
            param.requires_grad = True   # unfreeze last 2 transformer blocks

    model.head = nn.Linear(768, num_classes)
    return model
