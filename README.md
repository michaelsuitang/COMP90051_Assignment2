# COMP90051_Assignment2
Investigate performance of different feature extractors on binary classification tasks

## Methodology:
1. Convert images under data directory into pyTorch Tensors
2. Data preprocessing: Sampling, Resizing, etc.
3. Construct three different encoders (Fully connected network, CNN, Densenet121) and one Decoder model (3-layer Inverse CNN). The encoders should have the same output dimension (256)
4. Train three autoencoders, perform cross validation on learning rate/betas
5. Detach decoder, attach 2-layer FC network on top of encoders
6. Select a few (5~10) binary labels from list_attr_celeba.csv, train the newly attached FC networks on each of the labels. Do not fine tune the backbone encoders.
7. Record and compare results (accuracy, precision etc)
