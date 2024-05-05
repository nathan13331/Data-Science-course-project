import musdb
import random
import numpy as np
import torch
from torch.utils.data import Dataset


class MusdbDataset(Dataset):
    def __init__(self, root_dir, max_len, subset='train'):
        self.mus = musdb.DB(root=root_dir, subsets=subset)
        self.audio_length = max_len

    def __len__(self):
        return len(self.mus)

    def __getitem__(self, idx):
        track = self.mus.tracks[idx]
        chunk_duration = self.audio_length
        chunk_start = random.uniform(0, track.duration - chunk_duration)

        track.chunk_duration = chunk_duration
        track.chunk_start = chunk_start

        x = track.audio.T.astype(np.float64)
        y = track.targets['vocals'].audio.T.astype(np.float64)

        max_value = max(np.max(np.abs(y)), np.max(np.abs(x)))
        target_audio_normalized = np.zeros_like(y)
        mixture_audio_normalized = np.zeros_like(x)
        if max_value != 0:
            target_audio_normalized = y / max_value
            mixture_audio_normalized = x / max_value

        return {'mixture': torch.from_numpy(target_audio_normalized).float(),
                'training_target': torch.from_numpy(mixture_audio_normalized).float()}
