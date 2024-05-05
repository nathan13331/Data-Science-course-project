import torch
import matplotlib.pyplot as plt
import numpy as np
import random


def plot_sample_output(model, dataset, max_channels=5.0):
    # Take a random data point from the dataset
    sample = dataset.mus.tracks[0]

    chunk_duration = max_channels
    chunk_start = random.uniform(0, sample.duration - chunk_duration)

    sample.chunk_duration = chunk_duration
    sample.chunk_start = chunk_start

    x = sample.audio.T.astype(np.float64)
    y = sample.targets['vocals'].audio.T.astype(np.float64)

    max_value = max(np.max(np.abs(y)), np.max(np.abs(x)))
    y = y / max_value
    x = x / max_value

    mixture_test = torch.from_numpy(x).float().reshape([2, int(dataset.audio_length*44100)]).double()
    target_test = torch.from_numpy(y).float().reshape([2, int(dataset.audio_length*44100)]).double()

    model_output_test = model(mixture_test).detach().numpy()

    # Plotting
    max_abs_value = max(
        abs(x).max(),
        abs(y).max(),
        abs(model_output_test).max()
    )
    plt.subplot(3, 1, 1)
    plt.plot(x[0, :1000], label='Mixture (Channel 1)')
    plt.plot(x[1, :1000], label='Mixture (Channel 2)')
    plt.title('Input Mixture, Target, and Model Output')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.ylim(-max_abs_value, max_abs_value)

    plt.subplot(3, 1, 2)
    plt.plot(y[0, :1000], label='Target (Channel 1)')
    plt.plot(y[1, :1000], label='Target (Channel 2)')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.ylim(-max_abs_value, max_abs_value)

    plt.subplot(3, 1, 3)
    plt.plot(model_output_test[0, :1000], label='Model Output (Channel 1)')
    plt.plot(model_output_test[1, :1000], label='Model Output (Channel 2)')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.ylim(-max_abs_value, max_abs_value)

    plt.tight_layout()
    plt.show()

    return x, y
