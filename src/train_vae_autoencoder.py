import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from models import MLPEncoder, MLPDecoder, Autoencoder
from data import TrendImageDataset
from trainer import Trainer
from loss import BCE, KLReg


train = TrendImageDataset(root="../data/processed/images/staggered/train", flatten=True)
test = TrendImageDataset(root="../data/processed/images/staggered/test", flatten=True)

train = DataLoader(train, batch_size=1, shuffle=True)
test = DataLoader(test, batch_size=1, shuffle=True)


input_dim = 128*64

model = Autoencoder(
    encoder=MLPEncoder(input_dim, [2048, 512, 128], vae=True),
    decoder=MLPDecoder(input_dim, widths=[128, 512, 2048])
)


optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)


writer = SummaryWriter()

class Loss(nn.Module):
    def __init__(self):
        super().__init__()

        self.bce = BCE()
        self.mse = nn.MSELoss()
        self.kl = KLReg()

    def forward(self, latent, p, y):
        z, mu, logvar = latent
        
        bce_loss = self.bce(p, y)
        mse_loss = self.mse(p, y)
        kl_term = self.kl(mu, logvar)

        loss = bce_loss + mse_loss + kl_term

        return loss


lossfn = Loss()

trainer = Trainer(
    model=model, 
    optimiser=optimiser, 
    lossfn=lossfn, 
    train=train, 
    test=test, 
    writer=writer, 
    save_dir="../checkpoints/vae_autoencoder"
)

trainer.run(epochs=100)