import torch
import torch.nn as nn


class UNet(nn.Module):
    def __init__(self, inputChannelCount: int, outputChannelCount: int):
        super(UNet, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(inputChannelCount, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
        )

        # Final output layer
        self.out = nn.Conv2d(64, outputChannelCount, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.encoder(x)
        # Decoder
        x = self.decoder(x1)
        # Concatenate skip connection from encoder
        x = torch.cat([x, x1], dim=1)
        # Output layer
        x = self.out(x)
        return x
