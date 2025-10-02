import torch
import torch.nn as nn
from torchvision.transforms import transforms
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

import pandas as pd

import os
from typing import Optional, Union



# Preprocessing data
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# add Gaussian noise
class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std=0.05, p=0.5):
        """
        mean: Gaussian noise mean
        std: Gaussian noise standard deviation
        p:   Probability of applying noise
        """
        self.mean = mean
        self.std = std
        self.p = p

    def __call__(self, tensor):
        if torch.rand(1).item() < self.p:  # Decide whether to add noise based on probability
            noise = torch.randn_like(tensor) * self.std + self.mean
            tensor = tensor + noise
        return tensor.clamp(0., 1.)  # Clamp to [0,1]

    def __repr__(self):
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std}, p={self.p})"


# preprocessing + augmentation for training set
train_transform = transforms.Compose([
    transforms.Resize((64, 64)),                         # resize
    transforms.RandomHorizontalFlip(p=0.5),                # horizontal flip
    transforms.RandomRotation(degrees=15),                 # rotation ±15°
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # brightness and contrast adjustment
    transforms.ToTensor(),
    AddGaussianNoise(mean=0.0, std=0.05, p=0.5),           # add Gaussian noise
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])])

# preprocessing for test set
test_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])])


# dataset definition
class ImageDataset(Dataset):
    """
    General-purpose and CelebA-friendly image dataset:
    - Unified RGB, resize, and normalization
    - Supports both supervised/unsupervised learning
    - Supports mapping CelebA labels {-1, 1} to {0, 1} (can be disabled/customized)
    - labels can be pandas.Series / dict / None
    """
    def __init__(
        self,
        image_file_list,
        image_dir: str = "data/img_align_celeba/img_align_celeba",
        labels: Optional[Union["pd.Series", dict]] = None,
        transform: Optional[nn.Module] = None,
        label_mapping: Optional[dict] = {-1: 0, 1: 1},
        return_filename: bool = False,
    ):
        assert os.path.exists(image_dir), f"Image dir not found: {image_dir}"
        self.image_dir = image_dir
        self.image_files = list(image_file_list)
        self.transform = transform
        self.labels = labels
        self.label_mapping = label_mapping
        self.return_filename = return_filename

        # Try to detect pandas.Series
        self._is_pandas_series = False
        try:
            import pandas as pd  # Only used for type checking
            self._is_pandas_series = isinstance(labels, pd.Series)
        except Exception:
            pass

    def __len__(self):
        return len(self.image_files)

    def _get_label(self, img_name):
        if self.labels is None:
            return 0  # unsupervised: placeholder label

        # pandas.Series
        if self._is_pandas_series:
            y = self.labels.loc[img_name]
        else:  # dict-like
            y = self.labels[img_name]

        # Optional mapping (for CelebA: -1/1 -> 0/1)
        if self.label_mapping is not None:
            y = self.label_mapping.get(int(y), y)
        return int(y)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)

        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        label = self._get_label(img_name)

        if self.return_filename:
            return image, label, img_name
        return image, label