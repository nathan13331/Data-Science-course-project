import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

from train import train_model, weights_init
from dataset import MusdbDataset
from unet import UNet
from visualization import plot_sample_output


# Constants
learning_rate = 0.0001
epochs = 10
max_channels = 5.0
n_batches = 10

# Initialize model, dataset, dataloader, criterion, and optimizer
model = UNet(in_channels=2, out_channel=2).double()
model.apply(weights_init)

dataset = MusdbDataset(
    root_dir="./musdb18",
    max_len=max_channels
)
dataloader = DataLoader(dataset)
criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)

# Train the model
mse_values = train_model(model, dataloader, criterion, optimizer, scheduler, epochs, n_batches)

# Plot sample output
plot_sample_output(model, dataset, max_channels)

# Save the model
# torch.save(model.state_dict(), 'audio_separation_model.pth')
