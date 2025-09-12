import torch.nn as nn
import torch
from torchvision.models import resnet50, ResNet50_Weights
from lightning import LightningModule
import torchmetrics


class ModelApp(LightningModule):
    def __init__(self, epochs, batch_size, lr, ada=False, weights:str=None):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs

        self.ada_augment = ada

        