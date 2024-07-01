import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from models.models import DenseAutoencoder
from data .data import TrendImageDataset
from .trainer import Trainer


train = TrendImageDataset(root="../data/processed/images/staggered/train", flatten=True)
test = TrendImageDataset(root="../data/processed/images/staggered/test", flatten=True)

train = DataLoader(train, batch_size=128, shuffle=True)
test = DataLoader(test, batch_size=128, shuffle=True)

model = DenseAutoencoder(widths=[64*128, 512, 128])

optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)

writer = SummaryWriter()

class Loss(nn.Module):
    def __init__(self):
        super().__init__()

        self.bce = nn.BCELoss()
    def forward(self, z, p, y):
        bce_loss = self.bce(p, y)
        return bce_loss


lossfn = Loss()

trainer = Trainer(
    model=model, 
    optimiser=optimiser, 
    lossfn=lossfn, 
    train=train, 
    test=test, 
    writer=writer, 
    save_dir="../checkpoints/mlp_autoencoder"
)

trainer.run(epochs=100)