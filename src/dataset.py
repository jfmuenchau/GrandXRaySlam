import webdataset as wds
from torchvision import transforms
import torch
from typing import List
from .utils import AdaAugment

transform = transforms.ToTensor()

def parse_labels(label_bytes):
    return torch.tensor([int(x) for x in label_bytes.decode("utf-8").split()])

wds.autodecode.decoders["cls"] = lambda data: data  # keep bytes

def get_dataset(tar_files:List, ada_augment=False):
    """
    tar_files: List of tar files to include in the dataset
    ada_augment: True if AdaAugment should be applied
    """
    if ada_augment:
        augment = AdaAugment()

        def augment_sample(sample):
            key = sample["__key__"]
            img = sample["jpg"]
            img = augment(key, img)
            sample["jpg"] = img
            return sample

        dataset = (
            wds.WebDataset(tar_files)
            .decode("pil")
            .map(augment_sample)
            .to_tuple("__key__", "jpg", "cls")
            .map_tuple(lambda k: k,
                       lambda img : transform(img),
                       parse_labels)
        )
        return dataset, augment

    else:
        dataset = (
            wds.WebDataset(tar_files)
            .decode("pil")
            .to_tuple("__key__", "jpg", "cls")
            .map_tuple(lambda k: k,
                    lambda img: transform(img),
                    parse_labels)
        )
        return dataset, None