import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt


def weights_init(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
        nn.init.xavier_uniform_(m.weight.data.double())
        if m.bias is not None:
            nn.init.constant_(m.bias.data.double(), 0.01)


def train_model(model, dataloader, criterion, optimizer, scheduler, epochs = 100, n_batches = 100):
    mse_values = []
    for epoch in range(epochs):
        epoch_mse = 0.0

        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f'Epoch {epoch + 1}/{epochs}')):
            if batch_idx >= n_batches:
                break
            optimizer.zero_grad()
            mixture = batch['mixture'].reshape([2, int(dataloader.dataset.audio_length*44100)]).double()
            target = batch['training_target'].reshape([2, int(dataloader.dataset.audio_length*44100)]).double()

            output = model(mixture)
            output = output.type_as(target)
            loss = criterion(output, target)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

            optimizer.step()

            epoch_mse += loss.item()

        epoch_mse /= len(dataloader)
        mse_values.append(epoch_mse)
        # print(f'\n Epoch {epoch + 1}/{epochs} - Mse: {epoch_mse}')

        scheduler.step()

    print("Lowest MSE value:", min(mse_values))

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, epochs + 1), mse_values, marker='o')
    plt.title('Change in MSE over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Squared Error')
    plt.grid(True)
    plt.show()
    plt.figure(figsize=(12, 6))

    return mse_values
