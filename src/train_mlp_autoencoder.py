import torch
from torch import nn
from models import MLPEncoder, MLPDecoder, Autoencoder


input_dim = 128*64


model = Autoencoder(
    encoder=MLPEncoder(input_dim, [2048, 1024, 512]),
    decoder=MLPDecoder(input_dim, widths=[512, 1024, 2048])
)



print(model)



x = torch.zeros((1, input_dim))

latent, p = model(x)

print(p.shape)