import time
import torch



def train_autoencoder(encoder, decoder, train_loader, test_loader, optimizer, criterion, device, n_epochs=10):
    """
    Generic training loop for supervised multiclass learning
    """
    LOG_INTERVAL = 100
    running_loss = list()
    test_losses = list()
    start_time = time.time()
    encoder.to(device)
    decoder.to(device)

    for epoch in range(n_epochs):
        epoch_loss = 0.
        print(time.time() - start_time)
        for i, data in enumerate(train_loader):  # Loop over elements in training set
            if not i:
                print(time.time() - start_time)
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

        print('Epoch complete! Mean training loss: {:.4f} | Time {:.2f} s'.format(epoch_loss/len(train_loader), time.time()-start_time))

        test_loss = 0.

        for i, data in enumerate(test_loader):
            x, _ = data
            x = x.to(device)

            with torch.no_grad():
                code = encoder(x)
                reconstructed_x = decoder(code)

                test_loss += criterion(input=reconstructed_x, target=x).item() / batch_size

        print('[TEST] Mean loss {:.4f} | Time {:.2f} s'.format(test_loss/len(test_loader), time.time()-start_time))
        test_losses.append(test_loss)
    return running_loss, test_losses