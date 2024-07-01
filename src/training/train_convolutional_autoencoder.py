import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from models import ConvolutionalEncoder, ConvolutionalDecoder, Autoencoder

from data import TrendImageDataset

from trainer import Trainer

from loss import BCE

train = TrendImageDataset(root="../data/processed/images/staggered/train")
test = TrendImageDataset(root="../data/processed/images/staggered/test")

train = DataLoader(train, batch_size=16, shuffle=True)
test = DataLoader(test, batch_size=16, shuffle=True)

model = Autoencoder(
    encoder=ConvolutionalEncoder(chanels=[16, 32, 64, 128]),
    decoder=ConvolutionalDecoder(in_channels=128, channels=[64, 32, 16, 1])
)

optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)


writer = SummaryWriter()

class Loss(nn.Module):
    def __init__(self):
        super().__init__()

        self.bce = BCE()
        self.mse = nn.MSELoss()

    def forward(self, p, y):
        bce_loss = self.bce(p, y)
        mse_loss = self.mse(p, y)
        loss = bce_loss + mse_loss
        return loss


lossfn = Loss()

trainer = Trainer(
    model=model, 
    optimiser=optimiser, 
    lossfn=lossfn, 
    train=train, 
    test=test, 
    writer=writer, 
    save_dir="../checkpoints/conv_autoencoder/checkpoint.pth"
)

trainer.run(epochs=100)