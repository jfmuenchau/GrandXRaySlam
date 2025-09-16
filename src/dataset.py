import webdataset as wds
from torchvision import transforms
import torch
from typing import List
from .utils import *
import numpy as np
import pandas as pd
import random

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def parse_labels(label_bytes):
    return torch.tensor([int(x) for x in label_bytes.decode("utf-8").split()])

wds.autodecode.decoders["cls"] = lambda data: data  # keep bytes

def get_dataset(tar_files:List, train:bool, ada_augment=False):
    """
    tar_files: List of tar files to include in the dataset
    ada_augment: True if AdaAugment should be applied
    """
    if train:
        if ada_augment:
            augment = AdaAugment()

            def augment_sample(sample):
                key = sample["__key__"]
                img = sample["jpg"]
                img = augment(key, img)
                sample["jpg"] = img
                return sample

            dataset = (
                wds.WebDataset(tar_files, shardshuffle=len(tar_files))
                .decode("pil")
                .map(augment_sample)
                .to_tuple("__key__", "jpg", "cls")
                .map_tuple(lambda k: k,
                        lambda img : transform(img),
                        parse_labels)
            )
            return dataset, augment

        else:
            def augment_sample(sample):
                img = sample["jpg"]
                method = random.choice(augmentation_space)
                magntiude = random.random()
                img = method(img, magntiude)
                sample["jpg"] = img
                return sample
                
            dataset = (
                wds.WebDataset(tar_files, shardshuffle=len(tar_files))
                .decode("pil")
                .map(augment_sample)
                .to_tuple("__key__", "jpg", "cls")
                .map_tuple(lambda k: k,
                        lambda img: transform(img),
                        parse_labels)
            )
            return dataset, False
    else:
        dataset = (
            wds.WebDataset(tar_files, shardshuffle=len(tar_files))
            .decode("pil")
            .to_tuple("__key__", "jpg", "cls")
            .map_tuple(lambda k: k,
                    lambda img: transform(img),
                    parse_labels)
        )
        return dataset, False

def get_metadata():
    df = pd.read_csv("../data/train1.csv")

    def fill_age(df):
        mean = df["Age"].mean()
        std = df["Age"].std()
        n_missing = df["Age"].isna().sum()

        df.loc[df["Age"].isna(), "Age"] = np.random.normal(
            loc=mean,
            scale=std,
            size=n_missing
        )
        return df

    def fill_sex(df):
        n_missing = df["Sex"].isna().sum()
        choices = ["Male", "Female"]

        df.loc[df["Sex"].isna(), "Sex"] = np.random.choice(
            choices, 
            size = n_missing,
            replace=True
        )
        return df

    df = fill_age(df)
    df = fill_sex(df)