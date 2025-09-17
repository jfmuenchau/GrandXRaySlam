import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from lightning import LightningModule
from torch.optim.lr_scheduler import ReduceLROnPlateau
from .dataset import *
from .utils import *
from .model import *
from .agent import *
from glob import glob
from sklearn.model_selection import KFold

models_ = {
    "res18": (get_resnet18, 512),
    "effb0": (get_effnetb0, 1280),
    "swin" : (get_swin, 768),
    "convnext" : (get_convnext_tiny, 768),
    "coat": (get_coatnet, 1536)
}

class ModelApp(LightningModule):
    def __init__(self, batch_size:int, lr:float, model:str, weights:List, focal:bool, ada=False, fold=0, num_workers=2):
        super().__init__()
        get_model, in_features = models_[model]
        self.model = get_model(fine_tune=True)
        self.batch_size = batch_size
        self.lr = lr

        self.ada = ada
        self.num_workers = num_workers
        self.train_split, self.val_split = self._set_up_datasets(fold)
        self.weights = weights
        self._set_up_loss_fn(focal)
        if ada:
            self.agent = Agent(in_features, out_features=1)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            params=self.model.parameters(), 
            lr=self.lr, 
            weight_decay=5e-4
        )
        return {
            "optimizer":optimizer,
            "lr_scheduler": {
                "scheduler":ReduceLROnPlateau(optimizer, mode="max", factor=0.1, patience=2),
                "monitor":"val_auroc"
            }
        }
    
    def _set_up_datasets(self, fold):
        tar_files = glob("./data/train/*.tar")
        kf = KFold(shuffle=True, random_state=42)
        train_idx, val_idx = list(kf.split(tar_files))[fold] 
        train_split = [tar_files[i] for i in train_idx]
        val_split   = [tar_files[i] for i in val_idx]
        print(train_split, val_split)
        return train_split, val_split
    
    def _set_up_loss_fn(self, focal:bool=False):
        if focal:
            self.loss_fn = FocalLoss(weights=self.weights)
        else:
            self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=self.weights)

    def train_dataloader(self):
        dataset, self.adaaugment = get_dataset(
            tar_files=self.train_split, ada_augment=self.ada, train=True
        )
        return DataLoader(
            dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )
    
    def val_dataloader(self):
        dataset, _ = get_dataset(
            tar_files=self.val_split, train=False
        )
        return DataLoader(
            dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )
    
    def test_dataloader(self):
        dataset, _ = get_dataset(
            tar_files=glob("../data/test/*.tar"), train=False
        )
        return DataLoader(
            dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )
    
    def training_step(self, batch, batch_idx):
        if self.ada:
            pass

        else:
            loss = self.train_step(batch)
        return loss
    
    def ada_train_step(self, batch):
        keys, img, label = batch
        output = self(img)

        loss = self.loss_fn(output, label.float())
        probs = torch.sigmoid(output)
        # action
        # transfer action to AdaAugment
        # difference between magnitude and transform modeling
        # send loss to Agent to update Critic and Actor

        self.log(name="train_loss", value=loss, on_epoch=True, on_step=False)
        metrics = calculate_metrics(probs, label, stage="train")
        self.log_dict(metrics, on_step=False, on_epoch=True)
        return loss

    
    def train_step(self, batch):
        keys, img, label = batch
        output = self(img)

        loss = self.loss_fn(output, label.float())
        probs = torch.sigmoid(output)
        self.log(name="train_loss", value=loss, on_epoch=True, on_step=False)
        metrics = calculate_metrics(probs, label, stage="train")
        self.log_dict(metrics, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        _, img, label = batch
        output = self(img)
        loss = self.loss_fn(output, label.float())
        probs = torch.sigmoid(output)

        self.log(name="val_loss", value=loss, on_epoch=True, on_step=False, prog_bar=True)
        metrics = calculate_metrics(probs, label, stage="val")
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)