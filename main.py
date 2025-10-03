import os
import random
import time

import torch
from torch.utils.data import DataLoader

from preprocessing import train_transform, test_transform, ImageDataset
from bootstrap import tune_with_bootstrap, bootstrap_split

def store_output(filename_list, encoder, output_file, device, transform=test_transform, batch_size=64, img_dir="data/img_align_celeba/img_align_celeba"):
    start_time = time.time()
    ds = ImageDataset(
        image_file_list=filename_list,   # list of all image filenames 
        image_dir=img_dir,      # directory where the images are stored
        labels=None,                   # no labels for now (unsupervised / placeholder)
        transform=transform      # preprocessing + augmentation for training set
    )

    loader = DataLoader(
        ds,
        batch_size=batch_size,     # number of images per batch
        shuffle=False,     # do not shuffle test data
        # num_workers=4,
        pin_memory=True
    )
    encoder.eval()
    print("Extracting Features.")
    output_list = []
    for x, _ in loader:
        x = x.to(device)
        out = encoder(x)
        output_list.append(out.cpu())

    print("Features extracted. Time: ", time.time()-start_time)
    all_outputs = torch.cat(output_list, dim=0)
    torch.save(all_outputs, output_file)


if __name__ == "__main__":

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif torch.backends.mps.is_available(): 
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("Device: ", device)

    random.seed(90051)

    BATCH_SIZE = 64
    IMAGE_DIR = "data/img_align_celeba/img_align_celeba"
    OUTPUT_DIR = "data/output"
    MODEL = "FC"
    LEARNING_RATES = [1e-2, 1e-3, 1e-4]

    # Prepare 10 train/test splits
    filenames = sorted(os.listdir(IMAGE_DIR))[:50000]
    splits = bootstrap_split(filenames, B=10)

    # Outer bootstrap loop
    for i, (train_files, test_files) in enumerate(splits):
        output_dir = os.path.join(OUTPUT_DIR, os.path.join(MODEL, str(i)))
        print(output_dir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Dump train and test splits into files. 
        with open(os.path.join(output_dir, "train.txt"), "w") as f:
            for file in train_files:
                f.write(str(file) + "\n")
        with open(os.path.join(output_dir, "test.txt"), "w") as f:
            for file in test_files:
                f.write(str(file) + "\n")

        # Inner bootstraps
        running_loss, val_loss, encoder = tune_with_bootstrap(train_files, 
                                                              test_files, 
                                                              device, 
                                                              model=MODEL, 
                                                              lr_candidates=LEARNING_RATES,
                                                              B=3,
                                                              train_transform=train_transform,
                                                              test_transform=test_transform,
                                                              batch_size=BATCH_SIZE,
                                                              img_dir=IMAGE_DIR)
        
        # Record losses
        with open(os.path.join(output_dir, "running_loss.txt"), "w") as f:
            for loss in running_loss:
                f.write(str(loss) + "\n")
        with open(os.path.join(output_dir, "val_loss.txt"), "w") as f:
            for loss in val_loss:
                f.write(str(loss) + "\n")

        # Record features
        tensor_output_file = os.path.join(output_dir, "features.pt")
        store_output(filenames, encoder, tensor_output_file, device, transform=test_transform, batch_size=BATCH_SIZE, img_dir=IMAGE_DIR)


        