from dataset import MusdbDataset
from unet import UNet
from visualization import plot_sample_output


# Constants
max_channels = 5.0

# Initialize model, dataset, dataloader, criterion, and optimizer
model = UNet(in_channels=2, out_channel=2).double()

dataset = MusdbDataset(
    root_dir="./musdb18",
    max_len=max_channels
)

# Plot sample output
plot_sample_output(model, dataset, max_channels)
