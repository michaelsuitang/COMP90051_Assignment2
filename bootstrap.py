import torch
from torch import nn
from torch.utils.data import DataLoader
import random
from models import FcEncoder, CnnEncoder, EfficientNetEncoder, Decoder
from preprocessing import train_transform, test_transform, ImageDataset
from training import train_autoencoder

def bootstrap_split(filenames, B=10, seed=90051):
    random.seed(seed)
    n = len(filenames)
    splits = []
    for _ in range(B):
        train_filenames = [filenames[random.randint(0, n - 1)] for _ in range(n)] 
        val_filenames = list(set(filenames) - set(train_filenames))  
        splits.append((train_filenames, val_filenames))
    return splits

def train_and_val(train_filenames, val_filenames, model, device, lr, betas, train_transform=train_transform, test_transform=test_transform, batch_size=64, img_dir="data/img_align_celeba/img_align_celeba"):

    train_dataset = ImageDataset(image_file_list = train_filenames, image_dir= img_dir, labels=None, transform = train_transform)
    val_dataset   = ImageDataset(image_file_list = val_filenames, image_dir = img_dir, labels=None, transform = test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    if model == "FC":
        encoder = FcEncoder().to(device)
    elif model == "CNN":
        encoder = CnnEncoder().to(device)
    elif model == "EfficientNet":
        encoder = EfficientNetEncoder().to(device)
    else:
        raise Exception("Model type must be FC, CNN or EfficientNet.")

    decoder = Decoder().to(device)
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=lr, betas=betas
    )
    criterion = nn.MSELoss(reduction="sum")
    running_loss, val_losses = train_autoencoder(encoder, decoder, train_loader, val_loader, optimizer, criterion, device)
    return running_loss, val_losses, encoder, decoder

# Tune for each train-test split. Here filenames should be train files generated using bootstrap_split
def tune_with_bootstrap(filenames, val_filenames, device, model, lr_candidates=[1e-3, 1e-4, 1e-5], B=3, train_transform=train_transform, test_transform=test_transform, batch_size=64, img_dir="data/img_align_celeba/img_align_celeba"):

    # Step 1: Find the best lr
    betas_fixed = (0.9, 0.999)
    best_lr, best_loss = None, float("inf")
    for lr in lr_candidates:
        print(f"\n[Step 1] Trying lr={lr}, betas={betas_fixed}")
        losses = []
        for b, (train_files, val_files) in enumerate(bootstrap_split(filenames, B)):
            
            _, val_losses, _, _ = train_and_val(train_files, val_files, model, device, lr, betas_fixed, train_transform, test_transform, batch_size, img_dir)
            losses.append(val_losses[-1])

        avg_loss = sum(losses) / len(losses)
        print(f"lr={lr} -> avg val loss={avg_loss:.4f}\n")
        if avg_loss < best_loss:
            best_lr, best_loss = lr, avg_loss

    print(f"\nBest lr={best_lr} with val loss={best_loss:.4f}\n")
    
    # Step 2: Train the model using best lr
    print(f"\nTraining model {model} using best lr: {best_lr} on the whole training data.\n")
    running_loss, val_losses, encoder, _ = train_and_val(filenames, val_filenames, model, device, best_lr, betas_fixed, train_transform, test_transform, batch_size, img_dir)
    return running_loss, val_losses, encoder
    












