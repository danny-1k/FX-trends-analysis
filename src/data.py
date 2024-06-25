import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

from models import Autoencoder, MLPEncoder, MLPDecoder




class TrendImageDataset(Dataset):
    """
    Loads Images and converts them to Torch Tensors.
    """
    
    def __init__(self, root):
        self.images = [os.path.join(root, f) for f in os.listdir(root)]
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        f = self.images[idx]
        x = np.asarray(Image.open(f))
        x = x / 255
        x = torch.from_numpy(x)
        x = x.unsqueeze(0).float()

        return x


class TrendDataset(Dataset):
    def __init__(self, df, converter, window=5):
        self.df = df
        self.window = window
        self.converter = converter

    def __len__(self):
        return int(len(self.df) // self.window) * 17 # 17 currencies

    def __getitem__(self, idx):
        window_index = idx // 17
        currency_index = idx % 17

        chunk = self.df.iloc[window_index * self.window: window_index * self.window + self.window]
        image_labels = list(filter(lambda x: x.startswith("GBP"), list(chunk.columns)))#list(chunk.columns())
        currency = image_labels[currency_index]

        image = self.converter(chunk[currency])

        # images = {label: self.converter(chunk[label]) for label in image_labels}

        return image
    

class MLPTrendEmbeddingDataset(TrendDataset):
    def __init__(self, df, converter, window=5):
        super().__init__(df=df, converter=converter, window=window)

        self.model = Autoencoder(
            encoder=MLPEncoder(128*64, [2048, 512, 128]),
            decoder=MLPDecoder(128*64, widths=[128, 512, 2048])
        )

        checkpoint = torch.load("../checkpoints/conv_autoencoder/checkpoint.pth")
        state_dict = checkpoint["state_dict"]

        self.model.load_state_dict(state_dict)
        self.model.requires_grad_(False)


    @torch.no_grad()
    def __getitem__(self, idx):
        image = super().__getitem__(idx)
        image = torch.from_numpy(image)
        image = image.view(1, -1).float()
        embedding, _ = self.model(image)
        embedding = embedding.squeeze()
        
        return embedding, image.view(1, 64, 128)
