import os
import random

import torch
from torch.utils.data import DataLoader

from models import FcEncoder, CnnEncoder, EfficientNetEncoder, Decoder, Classifier
from preprocessing import train_transform, test_transform, ImageDataset
from training import train_autoencoder

if __name__ == "__main__":

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif torch.backends.mps.is_available(): 
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    random.seed(90051)

    BATCH_SIZE = 64
    IMAGE_DIR = "data/img_align_celeba/img_align_celeba"

    # Prepare train and test datasets
    filenames = os.listdir(IMAGE_DIR)
    random.shuffle(filenames)

    split_idx = int(0.8 * len(filenames))
    train_filenames = filenames[:split_idx]
    test_filenames  = filenames[split_idx:]

    train_ds = ImageDataset(
        image_file_list=train_filenames,   # list of training image filenames
        image_dir=str(IMAGE_DIR),      # directory where the images are stored
        labels=None,                   # no labels for now (unsupervised / placeholder)
        transform=train_transform      # preprocessing + augmentation for training set
    )

    # Create the testing dataset
    test_ds = ImageDataset(
        image_file_list=test_filenames,    # list of testing image filenames
        image_dir=str(IMAGE_DIR),      # directory where the images are stored
        labels=None,                   # no labels for now
        transform=test_transform       # preprocessing only for test set
    )

    # Create DataLoaders to efficiently load data in batches
    train_loader = DataLoader(
        train_ds,
        batch_size=64,     # number of images per batch
        shuffle=True,      # shuffle the training data at every epoch
        num_workers=4,     # number of subprocesses to use for data loading
        pin_memory=True    # speeds up transfer to GPU
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=64,     # number of images per batch
        shuffle=False,     # do not shuffle test data
        num_workers=4,
        pin_memory=True
    )

    fc_encoder = FcEncoder()
    cnn_encoder = CnnEncoder()
    efficientnet_encoder = EfficientNetEncoder()

    fc_decoder = Decoder()
    cnn_decoder = Decoder()
    efficientnet_decoder = Decoder()

    criterion = torch.nn.MSELoss(reduction="sum")

    # Training Autoencoders
    # TODO: k-fold cross validation on learning rate and betas.
    fc_optim = torch.optim.Adam(list(fc_encoder.parameters()) + list(fc_decoder.parameters()), lr=1e-2, betas=(0.9, 0.999))
    cnn_optim = torch.optim.Adam(list(cnn_encoder.parameters()) + list(cnn_decoder.parameters()), lr=1e-2, betas=(0.9, 0.999))
    efficientnet_optim = torch.optim.Adam(list(efficientnet_encoder.parameters()) + list(efficientnet_decoder.parameters()), lr=1e-2, betas=(0.9, 0.999))


    # train_autoencoder(fc_encoder, fc_decoder, train_loader, test_loader, fc_optim, criterion, device)
    # train_autoencoder(cnn_encoder, cnn_decoder, train_loader, test_loader, cnn_optim, criterion, device)
    train_autoencoder(efficientnet_encoder, efficientnet_decoder, train_loader, test_loader, efficientnet_optim, criterion, device)