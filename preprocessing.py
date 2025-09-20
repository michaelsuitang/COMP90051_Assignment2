import torch
from torchvision.transforms import transforms

import os

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms



# Preprocessing data
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

class ImageDataset(Dataset):
    def __init__(self, 
                 image_file_list,
                 image_dir="data/img_align_celeba/img_align_celeba", 
                 labels=None,  # This should be a pandas series, indexed by filename, values are labels of a certain attribute. Only needed for classification
                 transform=transform):
        """
        Args:
            image_dir (str): Path to the folder containing images
            class_mapping (dict): Maps substrings in filenames to integer labels.
                                  Example: {'cat': 0, 'dog': 1}
            transform (callable, optional): Optional transform to apply to images
        """
        assert(os.path.exists(image_dir))
        self.image_dir = image_dir
        self.transform = transform
        self.image_files = image_file_list
        self.class_mapping = {-1: 0, 1: 1}
        self.labels = labels

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        # Apply transform
        if self.transform:
            image = self.transform(image)

        # Find label
        if self.labels:
            return image, self.labels[img_name]

        # For unsupervised learning, label is always 0.
        return image, 0
