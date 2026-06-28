import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvDQN(nn.Module):
    def __init__(self, in_channels: int, num_actions: int, grid_size: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # Dynamically compute the flattened size based on grid_size
        flat_size = 64 * grid_size * grid_size

        self.fc = nn.Sequential(
            nn.Linear(flat_size, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        q = self.fc(x)
        return q