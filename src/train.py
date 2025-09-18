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
    "res34": (get_resnet34, 512),
    "vit_t":(get_vit_tiny, 192),
    "effb0": (get_effnetb0, 1280),
    "convnext" : (get_convnext_tiny, 768),
}

class ModelApp(LightningModule):
    def __init__(self, batch_size:int, lr:float, model:str, weights:List, focal:bool, fine_tune, ada=False, fold=0, num_workers=2, rand_m_t=(1,0)):
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
            self.r_weight = 0.7
            self.rand_m , self.rand_t = rand_m_t
            actor_flag = not bool(self.rand_m)
            control_flag = not bool(self.rand_t)

            self.agent = Agent(in_features, actor=actor_flag, control=control_flag)

    def forward(self, x):
        return self.model(x)

    def _register_head_hook(self):
        self.state = {}
        self._hook_handle = None  # Store the hook handle

        def hook(module, input, output):
            self.state['head_input'] = input[0].detach()

        if isinstance(self.model, models.ResNet):
            self._hook_handle = self.model.fc.register_forward_hook(hook)
        elif isinstance(self.model, models.EfficientNet):
            self._hook_handle = self.model.classifier[-1].register_forward_hook(hook)
        elif isinstance(self.model, models.ConvNeXt):
            self._hook_handle = self.model.classifier[2].register_forward_hook(hook)
        elif isinstance(self.model, timm.models.vision_transformer.VisionTransformer):
            self._hook_handle = self.model.head.register_forward_hook(hook)

    def _remove_head_hook(self):
        if hasattr(self, '_hook_handle') and self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            params=self.model.parameters(), 
            lr=self.lr, 
            #weight_decay=1e-5
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
            self.loss_fn = FocalLoss(weights=self.weights, reduction="none")
        else:
            self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=self.weights, reduction="none")

    def train_dataloader(self):
        dataset, self.adaaugment = get_dataset(
            tar_files=self.train_split, ada_augment=self.ada, train=True, rand_m_t=(self.rand_m, self.rand_t)
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
    
    def on_train_epoch_start(self):
        if self.ada:
            self._register_head_hook()
    
    def on_train_epoch_end(self):
        if self.ada:
            self._remove_head_hook()

    def training_step(self, batch, batch_idx):
        if self.ada:
            loss = self.ada_train_step(batch)

        else:
            loss = self.train_step(batch)
        self.log(name="train_loss", value=loss, on_epoch=True, on_step=False)

        return loss
    
    def ada_train_step(self, batch):
        keys, img, label = batch
        output = self(img)
        state = self.state['head_input']

        loss = self.loss_fn(output, label.float())
        prev_loss = self.agent.loss_memory(keys, loss.detach())
        probs = torch.sigmoid(output)
        metrics = calculate_metrics(probs, label, stage="train")
        self.log_dict(metrics, on_step=False, on_epoch=True)

        action, transform_idx = self.agent.action(state)
        self.adaaugment.set(keys, action, transform_idx)

        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1).unsqueeze(1)
        reward = self.r_weight * (loss.detach().mean() - prev_loss.mean()) + (1- self.r_weight) * entropy

        self.log(name="reward", value=reward.mean())
        actor_loss, critic_loss = self.agent.update(keys, state, reward)

        self.log_dict({
            "actor_loss":actor_loss,
            "critic_loss":critic_loss,
            "magnitude":action.mean()
        }, on_epoch=True, on_step=False)
        return loss.mean()
    
    def train_step(self, batch):
        keys, img, label = batch
        output = self(img)

        loss = self.loss_fn(output, label.float())
        probs = torch.sigmoid(output)
        
        metrics = calculate_metrics(probs, label, stage="train")
        self.log_dict(metrics, on_step=False, on_epoch=True)
        return loss.mean()
    
    def validation_step(self, batch, batch_idx):
        _, img, label = batch
        output = self(img)
        loss = self.loss_fn(output, label.float())
        probs = torch.sigmoid(output)

        self.log(name="val_loss", value=loss.mean(), on_epoch=True, on_step=False, prog_bar=True)
        metrics = calculate_metrics(probs, label, stage="val")
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)