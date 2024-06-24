import torch
from torch import nn


class Autoencoder(nn.Module):
    """
    Autoencoder base class for joining 
    encoder and decoder models.
    """
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        latent = self.encoder(x)
        x = self.decoder(latent)

        return latent, x
    
    
class MLPEncoder(nn.Module):
    """
    MLP Encoder Class
    """
    def __init__(self, input_size, widths):
        super().__init__()
        self.widths = widths
        self.blocks = nn.Sequential(*[
            nn.Sequential(
                nn.Linear(input_size if i == 0 else widths[i-1], width),
                nn.ReLU()
            )
            for i, width in enumerate(self.widths)
        ])

    def forward(self, x):
        x = self.blocks(x)
        return x


class MLPDecoder(nn.Module):
    """
    MLP Decoder Class
    """
    def __init__(self, output_size, widths):
        super().__init__()
        self.widths = widths
        self.blocks = nn.Sequential(*[
            nn.Sequential(
                nn.Linear(width, widths[i+1] if (i+1) < len(widths) else output_size),
                nn.ReLU() if (i+1) < len(widths) else nn.Sigmoid(),
            )
            for i, width in enumerate(self.widths)
        ])

    def forward(self, x):
        x = self.blocks(x)
        return x


class ConvolutionalEncoder(nn.Module):
    """
    Simple VGG-style convolutional encoder
    After two conv blocks, there's a max pool that reduces the spatial dimensions by half.
    """
    def __init__(self, in_channels=1, channels=[]):
        super().__init__()
        self.channels = channels

        self.blocks = nn.Sequential(*[
            nn.Sequential(
                nn.Conv2d(in_channels=in_channels if i == 0 else self.channels[i-1], out_channels=channels, kernel_size=3, padding=1),
                nn.ReLU(),

                nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1),
                nn.ReLU(),

                nn.MaxPool2d(kernel_size=2, stride=2)
            )

            for i, channels in enumerate(self.channels)
        ])

    def forward(self, x):
        x = self.blocks(x)
        return x


class ConvolutionalDecoder(nn.Module):
    """
    VGG-style convolutional decoder with transpose convolution
    After after upsampling the previous kernel maps, we apply two conv blocks
    """
    def __init__(self, in_channels, out_channels, channels=[]):
        super().__init__()
        self.channels = channels

        self.blocks = nn.Sequential(*[
            nn.Sequential(
                nn.ConvTranspose2d(in_channels=in_channels if i == 0 else self.channels[i-1], out_channels=channels if (i+1) < len(self.channels) else self.channels[i-1], kernel_size=2, stride=2),
                nn.ReLU(),

                nn.Conv2d(in_channels=channels if (i+1) < len(self.channels) else self.channels[i-1], out_channels=channels if (i+1) < len(self.channels) else out_channels, kernel_size=3, stride=1, padding=1),
                nn.ReLU() if (i+1) < len(self.channels) else nn.Sigmoid(),

            )
            
            for i, channels in enumerate(self.channels)
        ])

    def forward(self, x):
        x = self.blocks(x)
        return x