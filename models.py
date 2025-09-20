import numpy as np
import time, os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.models as models


if torch.cuda.is_available():
    device = torch.device("cuda:0")
elif torch.backends.mps.is_available(): 
    device = torch.device("mps")
else:
    device = torch.device("cpu")


# Fully connected 2-layer encoder, dimensions of input and output should be fixed, but width of middle layer can be tuned.
class FcEncoder(nn.Module):
    def __init__(self, hidden_dim=4096, input_dim=224*224, output_dim=256):
        super(FcEncoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 3-layer CNN encoder, can change # of channels in the middle layer. (2-layer downsampling would be too aggressive)
# first layer reduce 224*224 to 56 * 56, second layer reduce to 14 * 14
class CnnEncoder(nn.Module):
    def __init__(self, out_channels=16, hidden_c1=8, hidden_c2=8, kernel_size=16):
        super(CnnEncoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=hidden_c1, kernel_size=kernel_size, stride=4, padding=2)
        self.conv2 = nn.Conv2d(in_channels=hidden_c1, out_channels=hidden_c2, kernel_size=kernel_size, stride=4, padding=2)
        self.conv3 = nn.Conv2d(in_channels=hidden_c2, out_channels=hidden_c2, kernel_size=kernel_size, stride=4, padding=3)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        out = self.conv3(x)
        return out

# DenseNet121 Encoder. 
class DensenetEncoder(nn.Module):
    def __init__(self, latent_dim=256):
        super(DensenetEncoder, self).__init__()
        densenet = models.densenet121(pretrained=True)
        self.encoder = densenet.features
        # Output of self.encoder has dimension of 1024, we want to reduce to 256.
        self.enc_fc = nn.Linear(1024, latent_dim)

    def forward(self, x):
        z = self.encoder(x)              # shape: (B, 1024, 7, 7)
        z = z.mean([2, 3])               # global average pooling → (B, 1024), taking mean over dimension 2 and 3.
        z = self.enc_fc(z)               # latent vector (B, latent_dim)
        return z


# For Fairness, we should use the same Decoder for all three autoencoders
class Decoder(nn.Module):
    def __init__(self, in_channels=16, hidden_c1=8, hidden_c2=8, kernel_size=8):
        super(Decoder, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(in_channels=in_channels, out_channels=hidden_c1, kernel_size=kernel_size, stride=4, padding=3)
        self.deconv2 = nn.ConvTranspose2d(in_channels=hidden_c1, out_channels=hidden_c2, kernel_size=kernel_size, stride=4, padding=2)
        self.deconv3 = nn.ConvTranspose2d(in_channels=hidden_c2, out_channels=3, kernel_size=kernel_size, stride=4, padding=2)

    def forward(self, x):
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        out = torch.sigmoid(self.deconv3(x)) # Expect input to be in [-1, 1] hence the sigmoid function
        return out


def train_autoencoder(encoder, decoder, train_loader, test_loader, optimizer, criterion, device, n_epochs=10):
    """
    Generic training loop for supervised multiclass learning
    """
    LOG_INTERVAL = 500
    running_loss = list()
    start_time = time.time()
    encoder.to(device)
    decoder.to(device)

    for epoch in range(n_epochs):
        epoch_loss = 0.

        for i, data in enumerate(train_loader):  # Loop over elements in training set
            x, _ = data
            batch_size = x.shape[0]
            
            x = x.to(device)

            optimizer.zero_grad()         # Reset gradients
            code = encoder(x)
            reconstructed_x = decoder(code)

            loss = criterion(input=reconstructed_x, target=x) / batch_size

            loss.backward()               # Backward pass (compute parameter gradients)
            optimizer.step()              # Update weight parameter u

            running_loss.append(loss.item())
            epoch_loss += loss.item()

            if i % LOG_INTERVAL == 0:
                deltaT = time.time() - start_time
                mean_loss = epoch_loss / (i+1)
                print('[TRAIN] Epoch {} [{}/{}]| Mean loss {:.4f} | Time {:.2f} s'.format(epoch,
                    i, len(train_loader), mean_loss, deltaT))

        print('Epoch complete! Mean training loss: {:.4f}'.format(epoch_loss/len(train_loader)))

        test_loss = 0.

        for i, data in enumerate(test_loader):
            x, _ = data
            x = x.to(device)

            with torch.no_grad():
                code = encoder(x)
                reconstructed_x = decoder(code)

                test_loss += criterion(input=reconstructed_x, target=x).item() / batch_size

        print('[TEST] Mean loss {:.4f}'.format(test_loss/len(test_loader)))
