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
        x = self.decoder(latent[0] if self.encoder.vae else latent)

        return latent, x
    
    
class MLPEncoder(nn.Module):
    """
    MLP Encoder Class
    """
    def __init__(self, input_size, widths, vae=False):
        super().__init__()
        self.widths = widths
        self.vae = vae
        self.blocks = nn.Sequential(*[
            nn.Sequential(
                nn.Linear(input_size if i == 0 else widths[i-1], width if (i < (len(self.widths) - 1) or not vae) else 2 * width),
                nn.ReLU() if i < (len(self.widths) - 1) else nn.Identity()
            )
            for i, width in enumerate(self.widths)
        ])

    
    def forward(self, x):
        x = self.blocks(x)
        if not self.vae:
            return x
        
        # x of shape (N, 2 * E)
        x = x.view(x.shape[0], -1, 2)

        mu = x[:, :, 0]
        logvar = x[:, :, 1]

        z = self._reparameterize(mu, logvar)

        return (z, mu, logvar)


    def _reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.rand_like(std)

        return mu + eps * std
    


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
        if type(x) != torch.Tensor:
            x = x[0]

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