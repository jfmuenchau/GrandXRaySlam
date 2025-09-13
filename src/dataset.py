import webdataset as wds
from torchvision import transforms
import torch
from typing import List

transform = transforms.ToTensor()

def parse_labels(label_bytes):
    return torch.tensor([int(x) for x in label_bytes.decode("utf-8").split()])

wds.autodecode.decoders["cls"] = lambda data: data  # keep bytes

def get_dataset(tar_files:List):
    """
    tar_files: List of tar files to include in the dataset
    """
    dataset = (
        wds.WebDataset(tar_files)
        .decode("pil")
        .to_tuple("__key__", "jpg", "cls")
        .map_tuple(lambda k: k,
                lambda img: transform(img),
                parse_labels)
    )
    return dataset