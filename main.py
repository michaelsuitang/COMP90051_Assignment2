import os
import random

import torch
from torch.utils.data import DataLoader

from models import FcEncoder, CnnEncoder, DensenetEncoder, Decoder, Classifier
from preprocessing import transform, ImageDataset
from training import train_autoencoder

if torch.cuda.is_available():
    device = torch.device("cuda:0")
elif torch.backends.mps.is_available(): 
    device = torch.device("mps")
else:
    device = torch.device("cpu")

random.seed(90051)

# Prepare train and test datasets
filenames = os.listdir("data/img_align_celeba/img_align_celeba")
random.shuffle(filenames)

split_idx = int(0.8 * len(filenames))
train_filenames = filenames[:split_idx]
test_filenames  = filenames[split_idx:]

BATCH_SIZE = 16

train_dataset = ImageDataset(image_file_list=train_filenames, image_dir="data/img_align_celeba/img_align_celeba", transform=transform)
test_dataset = ImageDataset(image_file_list=test_filenames, image_dir="data/img_align_celeba/img_align_celeba", transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

fc_encoder = FcEncoder()
cnn_encoder = CnnEncoder()
densenet_encoder = DensenetEncoder()

fc_decoder = Decoder()
cnn_decoder = Decoder()
densenet_decoder = Decoder()

criterion = torch.nn.MSELoss(reduction="sum")

# Training Autoencoders
# TODO: k-fold cross validation on learning rate and betas.
fc_optim = torch.optim.Adam(list(fc_encoder.parameters()) + list(fc_decoder.parameters()), lr=1e-2, betas=(0.9, 0.999))
cnn_optim = torch.optim.Adam(list(cnn_encoder.parameters()) + list(cnn_decoder.parameters()), lr=1e-2, betas=(0.9, 0.999))
densenet_optim = torch.optim.Adam(list(densenet_encoder.parameters()) + list(densenet_decoder.parameters()), lr=1e-2, betas=(0.9, 0.999))

train_autoencoder(fc_encoder, fc_decoder, train_dataloader, test_dataloader, fc_optim, criterion, device)