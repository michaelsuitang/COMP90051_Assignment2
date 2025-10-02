import torch
from torch import nn
from torch.utils.data import DataLoader
import random
from models import FcEncoder, Decoder
from preprocessing import transform, ImageDataset
from training import train_autoencoder

def bootstrap_split(filenames, B=3, seed=42):
    random.seed(seed)
    n = len(filenames)
    splits = []
    for _ in range(B):
        train_idx = [random.randint(0, n - 1) for _ in range(n)] 
        val_idx = list(set(range(n)) - set(train_idx))  
        splits.append((train_idx, val_idx))
    return splits


def tune_fc_with_bootstrap(filenames, device, B=3):

    # Step 1: lr
    lr_candidates = [1e-2, 1e-3, 5e-4]
    betas_fixed = (0.9, 0.999)
    best_lr, best_loss = None, float("inf")
    print(ImageDataset.__init__.__code__.co_varnames)
    for lr in lr_candidates:
        print(f"\n[Step 1] Trying lr={lr}, betas={betas_fixed}")
        losses = []
        for b, (train_ids, val_ids) in enumerate(bootstrap_split(filenames, B)):
            train_files = [filenames[i] for i in train_ids]
            val_files   = [filenames[i] for i in val_ids]
           

            train_dataset = ImageDataset(image_file_list = train_files, image_dir= "data/img_align_celeba/img_align_celeba", transform = transform)
            val_dataset   = ImageDataset(image_file_list = val_files, image_dir = "data/img_align_celeba/img_align_celeba", transform = transform)

            train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
            val_loader   = DataLoader(val_dataset, batch_size=16, shuffle=False)

            encoder = FcEncoder().to(device)
            decoder = Decoder().to(device)
            optimizer = torch.optim.Adam(
                list(encoder.parameters()) + list(decoder.parameters()),
                lr=lr, betas=betas_fixed
            )
            criterion = nn.MSELoss()

            val_loss = train_autoencoder(encoder, decoder, train_loader, val_loader, optimizer, criterion, device)
            losses.append(val_loss)

        avg_loss = sum(losses) / len(losses)
        print(f"lr={lr} -> avg val loss={avg_loss:.4f}")
        if avg_loss < best_loss:
            best_lr, best_loss = lr, avg_loss

    print(f"\nBest lr={best_lr} with val loss={best_loss:.4f}")

    # Step 2: betas
    betas_candidates = [(0.9, 0.999), (0.9, 0.99), (0.5, 0.999)]
    best_betas, best_loss = None, float("inf")

    for betas in betas_candidates:
        print(f"\n[Step 2] Trying betas={betas}, lr={best_lr}")
        losses = []
        for b, (train_ids, val_ids) in enumerate(bootstrap_split(filenames, B)):
            train_files = [filenames[i] for i in train_ids]
            val_files   = [filenames[i] for i in val_ids]

            train_dataset = ImageDataset(image_file_list=train_files, image_dir="data/img_align_celeba/img_align_celeba", transform=transform)
            val_dataset   = ImageDataset(image_file_list=val_files, image_dir="data/img_align_celeba/img_align_celeba", transform=transform)

            train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
            val_loader   = DataLoader(val_dataset, batch_size=16, shuffle=False)

            encoder = FcEncoder().to(device)
            decoder = Decoder().to(device)
            optimizer = torch.optim.Adam(
                list(encoder.parameters()) + list(decoder.parameters()),
                lr=best_lr, betas=betas
            )
            criterion = nn.MSELoss()

            val_loss = train_autoencoder(encoder, decoder, train_loader, val_loader, optimizer, criterion, device)
            losses.append(val_loss)

        avg_loss = sum(losses) / len(losses)
        print(f"betas={betas} -> avg val loss={avg_loss:.4f}")
        if avg_loss < best_loss:
            best_betas, best_loss = betas, avg_loss

    print(f"\n Final best params: lr={best_lr}, betas={best_betas}, val loss={best_loss:.4f}")
    return best_lr, best_betas
