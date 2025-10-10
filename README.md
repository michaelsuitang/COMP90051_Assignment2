# COMP90051_Assignment2
Investigate performance of different feature extractors on binary classification tasks

## Methodology:
1. Convert images under data directory into pyTorch Tensors
2. Data preprocessing: Sampling, Resizing, etc.
3. Construct three different encoders (Fully connected network, CNN, EfficientNet-b0) and one Decoder model (3-layer Inverse CNN). The encoders should have the same output dimension (256)
4. Train three autoencoders, perform cross validation on learning rate/betas
5. Detach decoder, attach 2-layer FC network on top of encoders
6. Select a few (5~10) binary labels from list_attr_celeba.csv, train the newly attached FC networks on each of the labels. Do not fine tune the backbone encoders.
7. Record and compare results (accuracy, precision etc)
## Data:
Data from unsupervised learning can be found under the directory data/output. For each of three models, we train on the same 10 bootstrap samples, and extract all features of the first 50000 images (sorted by file name) in the image directory. train.txt and test.txt are the filenames of train/test split of this bootstrap sample, and should be the same (up to permutation) across different models. E.g. data/output/CNN/0/train.txt should be identical to data/output/FC/0/train.txt. Running_loss.txt contains the training loss of each batch when the model is trained. Batchsize is 64, each epoch should have 781 batches, and it's trained for 10 epochs. Val_loss.txt is loss on test set of each epoch. 
