import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
from models import ConvolutionalEncoder, ConvolutionalDecoder, Autoencoder
from models import MLPEncoder, MLPDecoder, Autoencoder

from tqdm import tqdm
from data import TrendImageDataset
from utils import RunningAverager



train = TrendImageDataset(root="../data/processed/images/staggered/train")
test = TrendImageDataset(root="../data/processed/images/staggered/test")

train = DataLoader(train, batch_size=16, shuffle=True)
test = DataLoader(test, batch_size=16, shuffle=True)


channels = [8, 16, 32, 64, 128]
input_dim = 128*64

# model = Autoencoder(
#     encoder=ConvolutionalEncoder(in_channels=1, channels=channels),
#     decoder=ConvolutionalDecoder(in_channels=128, out_channels=1, channels=channels[::-1])
# )


model = Autoencoder(
    encoder=MLPEncoder(input_dim, [2048, 512, 128]),
    decoder=MLPDecoder(input_dim, widths=[128, 512, 2048])
)

optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)


class BCE(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, p, y):
        p = p.clamp(min=1e-8, max=9.9e-1)
        loss = - (y * torch.log(p + self.eps) + (1-y) * torch.log(1-p)) 
        loss = loss.mean()

        return loss


class VarianceReg(nn.Module):
    def __init__(self, gamma=1):
        super().__init__()
        self.gamma = gamma

    def forward(self, z):
        z = z.view(z.shape[0], -1)

        return (self.gamma - z.std(axis=1)).clamp(max=0).mean()


class CovarianceReg(nn.Module):
    def __init__(self, nu=1):
        super().__init__()
        self.nu = nu

    def forward(self, z):
        z = z.view(z.shape[0], -1)

        embed_dims  = z.shape[-1]

        z_centered = z - z.mean(axis=1)
        covariance_matrix = (1/embed_dims) * z_centered @ z_centered.T
        covariance_matrix = covariance_matrix * (1-torch.eye(embed_dims).to(covariance_matrix.device))
        
        return covariance_matrix.mean()


class MSEBCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = BCE()

    def forward(self, p, y):
        loss = self.mse(p, y) + self.bce(p, y)
        return loss

EPOCHS = 100
lossfn = MSEBCELoss()#nn.MSELoss()
var_reg = VarianceReg()
covariance_reg = CovarianceReg()

writer = SummaryWriter()

best_test_loss = float("inf")


for epoch in range(EPOCHS):
    train_loss = RunningAverager()
    test_loss = RunningAverager()

    for i, x in enumerate(tqdm(train)):
        optimiser.zero_grad()
        x = x.view(x.shape[0], -1)

        # print(x.max())

        z, p = model(x)
        # p = model(x)
        loss = lossfn(p, x)
        
        loss.backward()
        optimiser.step()

        train_loss.update(loss.item())

    writer.add_scalar(tag="Train/Loss", scalar_value=train_loss.value, global_step=epoch)
    writer.add_image(tag="Train/Predictions", img_tensor=make_grid(p.view(p.shape[0], 1, 64, 128)), global_step=epoch)
    writer.add_image(tag="Train/GroudnTruth", img_tensor=make_grid(x.view(x.shape[0], 1, 64, 128)), global_step=epoch)


    with torch.no_grad():
        for i, x in enumerate(tqdm(test)):
            x = x.view(x.shape[0], -1)
            _, p = model(x)
            # p = model(x)
            loss = lossfn(p, x)

            test_loss.update(loss.item())

    writer.add_scalar(tag="Test/Loss", scalar_value=test_loss.value, global_step=epoch)
    writer.add_image(tag="Test/Predictions", img_tensor=make_grid(p.view(p.shape[0], 1, 64, 128)), global_step=epoch)
    writer.add_image(tag="Test/GroudnTruth", img_tensor=make_grid(x.view(x.shape[0], 1, 64, 128)), global_step=epoch)


    if test_loss.value < best_test_loss:
        best_test_loss = test_loss.value

        torch.save({
            "epoch": epoch + 1,
            "train_loss": train_loss.value, 
            "test_loss": test_loss.value,
            "state_dict": model.state_dict(),
            "model": model, 
        }, f"../checkpoints/conv_autoencoder_reg/checkpoint.pth")