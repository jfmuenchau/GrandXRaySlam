from torchvision import models
import torch.nn as nn
import timm


def freeze_all(model):
    for param in model.parameters():
        param.requires_grad=False

def get_resnet18(num_classes=14, fine_tune=True):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    if fine_tune:
        freeze_all(model)
        for param in model.layer4.parameters():
            param.requires_grad = True

    model.fc = nn.Linear(512, num_classes)
    return model

def get_resnet34(num_classes=14, fine_tune=True):
    model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)

    if fine_tune:
        freeze_all(model)
        for param in model.layer4.parameters():
            param.requires_grad = True

    model.fc = nn.Linear(512, num_classes)
    return model

def get_effnetb0(num_classes=14, fine_tune=True):
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

    if fine_tune:
        freeze_all(model)
        for idx in range(6, 8):
            for param in model.features[idx].parameters():
                param.requires_grad = True

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features=1280, out_features=num_classes)
    )
    
    return model

def get_effnetb2(num_classes=14, fine_tune=True):
    model = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.IMAGENET1K_V1)

    if fine_tune:
        freeze_all(model)
        for param in model.features[7].parameters():
            param.requires_grad = True

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),  # B2 uses higher dropout than B0
        nn.Linear(in_features=1408, out_features=num_classes)
    )
    
    return model

def get_vit_tiny(num_classes=14, fine_tune=True):
    model = timm.create_model("vit_tiny_patch16_224", pretrained=True)

    if fine_tune:
        freeze_all(model)
        for param in model.blocks[-2:].parameters():
            param.requires_grad = True

    model.head = nn.Linear(192, num_classes)
    return model

def get_convnext_tiny(num_classes=14, fine_tune=True):
    model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)

    if fine_tune:
        freeze_all(model)
        for param in model.features[6].parameters():
            param.requires_grad = True

    model.classifier[2] = nn.Linear(in_features=768, out_features=num_classes)
    return model

def get_swin(num_classes=14, fine_tune=True):
    model = timm.create_model("swin_s3_tiny_224", pretrained=True, num_classes=num_classes)

    if fine_tune:
        freeze_all(model)
        for param in model.layers[-1].blocks[-2:].parameters():
            param.requires_grad = True

    return model

def get_coatnet(num_classes=14, fine_tune=True):
    model = timm.create_model("coatnet_3_rw_224.sw_in12k", pretrained=True, num_classes=num_classes)

    if fine_tune:
        freeze_all(model)
        for param in model.stages[-1].parameters():
            param.requires_grad = True

    return model
