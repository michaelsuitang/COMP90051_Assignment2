import numpy as np
import time, os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.models as models


# Fully connected layer encoder.
class FcEncoder(nn.Module):
    def __init__(self, input_dim=224*224*3, output_dim=256):
        super(FcEncoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # flatten
        return self.fc1(x)

# 5-layer CNN encoder, can change # of channels in the middle layer. (2-layer downsampling would be too aggressive)
# Each layer halves size of image by max pooling
# TODO: Too few parameters now. Add more layers/MaxPool/FC layer at the end?
class CnnEncoder(nn.Module):
    def __init__(self, hidden_c1=8, hidden_c2=16, hidden_c3=32, hidden_c4=64, latent_dim=256, kernel_size=17, padding=8):
        super(CnnEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=hidden_c1, kernel_size=kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(in_channels=hidden_c1, out_channels=hidden_c2, kernel_size=kernel_size, padding=padding)
        self.conv3 = nn.Conv2d(in_channels=hidden_c2, out_channels=hidden_c3, kernel_size=kernel_size, padding=padding)
        self.conv4 = nn.Conv2d(in_channels=hidden_c3, out_channels=hidden_c4, kernel_size=kernel_size, padding=padding)
        self.conv5 = nn.Conv2d(in_channels=hidden_c4, out_channels=latent_dim, kernel_size=kernel_size, padding=padding)

    def forward(self, x):
        x = self.max_pool(F.relu(self.conv1(x)))
        x = self.max_pool(F.relu(self.conv2(x)))
        x = self.max_pool(F.relu(self.conv3(x)))
        x = self.max_pool(F.relu(self.conv4(x)))
        x = self.max_pool(F.relu(self.conv5(x)))
        x = x.mean([2, 3])
        return x

# DenseNet121 Encoder. 
class DensenetEncoder(nn.Module):
    def __init__(self, latent_dim=256):
        super(DensenetEncoder, self).__init__()
        densenet = models.densenet121()
        self.encoder = densenet.features
        # Output of self.encoder has dimension of 1024, we want to reduce to 256.
        self.fc = nn.Linear(1024, latent_dim)

    def forward(self, x):
        z = self.encoder(x)              # shape: (B, 1024, 7, 7)
        z = z.mean([2, 3])               # global average pooling → (B, 1024), taking mean over dimension 2 and 3.
        z = self.fc(z)               # latent vector (B, latent_dim)
        return z


# For Fairness, we should use the same Decoder for all three autoencoders
class Decoder(nn.Module):
    def __init__(self, latent_dim=256, in_channels=16, hidden_c1=8, hidden_c2=8, kernel_size=8):
        super().__init__()
        self.fc = nn.Linear(latent_dim, in_channels * 4 * 4)

        self.deconv1 = nn.ConvTranspose2d(in_channels, hidden_c1, kernel_size=kernel_size, stride=4, padding=3)  # 4 -> 14
        self.deconv2 = nn.ConvTranspose2d(hidden_c1, hidden_c2, kernel_size=kernel_size, stride=4, padding=2)    # 14 -> 56
        self.deconv3 = nn.ConvTranspose2d(hidden_c2, 3,         kernel_size=kernel_size, stride=4, padding=2)    # 56 -> 224

    def forward(self, z):                 # z: (B, 256)
        z = self.fc(z)                    # (B, 16*4*4)
        z = z.view(z.size(0), 16, 4, 4)   # (B, 16, 4, 4)
        z = F.relu(self.deconv1(z))
        z = F.relu(self.deconv2(z))
        out = torch.tanh(self.deconv3(z))   
        return out

class Classifier(nn.Module):
    def __init__(self, hidden_width=64):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(in_features=256, out_features=hidden_width)
        self.fc2 = nn.Linear(in_features=hidden_width, out_features=2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x   # Softmax is performed inside CrossEntropyLoss, no softmax here