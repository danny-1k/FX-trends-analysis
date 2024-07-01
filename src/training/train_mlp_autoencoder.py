import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from models import MLPEncoder, MLPDecoder, Autoencoder
from data import TrendArrayDataset, TrendImageDataset
from loss import BCE
from trainer import Trainer


train = TrendImageDataset(root="../data/processed/images/staggered/train", flatten=True)
test = TrendImageDataset(root="../data/processed/images/staggered/test", flatten=True)

train = DataLoader(train, batch_size=128, shuffle=True)
test = DataLoader(test, batch_size=128, shuffle=True)

input_dim = 128*64


model = Autoencoder(
    encoder=MLPEncoder(input_dim, [512, 128]),
    decoder=MLPDecoder(input_dim, widths=[128, 512])
)
 
optimiser = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=.99)


writer = SummaryWriter()

class Loss(nn.Module):
    def __init__(self):
        super().__init__()

        # self.bce = BCE()
        # self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()

    def forward(self, z, p, y):
        # bce_loss = self.bce(p, y)
        # mse_loss = self.mse(p, y)
        mae_loss = self.mae(p, y)
        return mae_loss


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