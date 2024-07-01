import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from models.models import DenseAutoencoder
from models.loss import BinaryContrastiveLoss
from data.data import TrendImageDatasetContrastive
from .trainer import ContrastiveTrainer


train = TrendImageDatasetContrastive(root="../data/processed/images/staggered/train", flatten=True)
test = TrendImageDatasetContrastive(root="../data/processed/images/staggered/test", flatten=True)

train = DataLoader(train, batch_size=128, shuffle=True)
test = DataLoader(test, batch_size=128, shuffle=True)

model = DenseAutoencoder(widths=[64*128, 512, 128])

optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)

writer = SummaryWriter()


class Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss()
        self.contrastive_loss = BinaryContrastiveLoss()

    def forward(self, z1, z2, p1, p2, x1, x2, label):
        bce_loss = self.bce(p1, x1) + self.bce(p2, x2)
        contrastive_loss = self.contrastive_loss(z1, z2, label)

        loss = bce_loss + contrastive_loss

        return loss
    

lossfn = Loss()

trainer = ContrastiveTrainer(
    model=model, 
    optimiser=optimiser, 
    lossfn=lossfn, 
    train=train, 
    test=test, 
    writer=writer, 
    save_dir="../checkpoints/mlp_autoencoder"
)

trainer.run(epochs=100)