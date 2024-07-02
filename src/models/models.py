import torch
from torch import nn


class DenseAutoencoder(nn.Module):
    def __init__(self, widths):
        super().__init__()

        self.widths = widths

        encoder_blocks = []
        decoder_blocks = []

        for i in range(len(widths)-1):
            encoder_blocks.append(nn.Sequential(
                nn.Linear(widths[i], widths[i+1]),
                nn.ReLU()
            ))

        for i in range(len(widths)-1)[::-1]:
            decoder_blocks.append(nn.Sequential(
                nn.Linear(widths[i+1], widths[i]),
                nn.ReLU() if i != 0 else nn.Sigmoid()
            ))

        self.encoder = nn.Sequential(*encoder_blocks)
        self.decoder = nn.Sequential(*decoder_blocks)

    def forward(self, x):
        z = self.encoder(x)
        x = self.decoder(z)

        return z, x


class VariationalAutoencoder(nn.Module):
    def __init__(self, widths):
        super().__init__()

        self.widths = widths

        encoder_blocks = []
        decoder_blocks = []

        for i in range(len(widths)-1):
            encoder_blocks.append(nn.Sequential(
                nn.Linear(widths[i], widths[i+1] * 2 if i == len(widths) - 2 else widths[i+1]),
                nn.Identity() if i == len(widths) - 2 else nn.ReLU()
            ))

        # only difference with vanilla autoencoder is the encoder of the vae predicts the mu and variance of the 
        # distribution of the embeddings.

        for i in range(len(widths)-1)[::-1]:
            decoder_blocks.append(nn.Sequential(
                nn.Linear(widths[i+1], widths[i]),
                nn.ReLU() if i != 0 else nn.Sigmoid()
            ))


        self.encoder = nn.Sequential(*encoder_blocks)
        self.decoder = nn.Sequential(*decoder_blocks)

    def forward(self, x):
        z = self.encoder(x)
        z = z.view(z.shape[0], -1, 2)

        mu = z[:, :, 0]
        logvar = z[:, :, 1]

        z = self._reparameterise(mu, logvar)

        x = self.decoder(z)

        return z, mu, logvar, x

    def _reparameterise(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        z = mu + eps * std

        return z
    

class ConvolutionalAutoencoder(nn.Module):
    def __init__(self, widths):
        super().__init__()
        
        self.widths = widths

        encoder_blocks = []
        decoder_blocks = []

        for i in range(len(widths)-1):
            encoder_blocks.append(nn.Sequential(
                nn.Conv2d(in_channels=widths[i], out_channels=widths[i+1], kernel_size=3, stride=1, padding=1),
                nn.ReLU(),

                nn.Conv2d(in_channels=widths[i+1], out_channels=widths[i+1], kernel_size=3, stride=1, padding=1),
                nn.ReLU(),

                nn.MaxPool2d(kernel_size=2, stride=2)
            ))

        # For the decoder blocks, we use a Transpose convolution -> Same Conv layer

        for i in range(len(widths)-1)[::-1]:

            decoder_blocks.append(nn.Sequential(
                nn.ConvTranspose2d(in_channels=widths[i+1], out_channels=widths[i+1], kernel_size=2, stride=2, padding=0, output_padding=0), # exactly double image
                nn.ReLU(),

                nn.Conv2d(in_channels=widths[i+1], out_channels=widths[i], kernel_size=3, stride=1, padding=1),
                nn.ReLU() if i != 0 else nn.Sigmoid()
                
            ))


        self.encoder = nn.Sequential(*encoder_blocks)
        self.decoder = nn.Sequential(*decoder_blocks)
    
    def forward(self, x):
        z = self.encoder(x)
        x = self.decoder(z)

        return z.view(z.shape[0], -1), x
    
    