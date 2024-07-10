import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
# import tensorflow
# import tensorboard
# tensorflow.io.gfile = tensorboard.compat.tensorflow_stub.io.gfile
from data.data import TrendDataset
from utils.data import SeriesToImageConverter
import pandas as pd
from tqdm import tqdm
from models.models import DenseAutoencoder, ConvolutionalAutoencoder

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from sklearn.decomposition import PCA

import matplotlib.pyplot as plt


model = ConvolutionalAutoencoder(widths=[1, 4, 8, 8, 8])#DenseAutoencoder(widths=[64*128, 512, 64])
# model.load_state_dict(torch.load("../checkpoints/mlp_autoencoder_contrastive/checkpoint.pt")["model"])
model.load_state_dict(torch.load("../checkpoints/conv_autoencoder_contrastive/checkpoint.pt")["model"])


converter = SeriesToImageConverter(height=64, width=128)
train_csv = pd.read_csv("../data/processed/splits/train.csv")
train_dataset = TrendDataset(df=train_csv, converter=converter)#TrendDataset(df=train_csv, converter=converter)
train_dataset = DataLoader(train_dataset, batch_size=1, shuffle=False)

writer = SummaryWriter() 

embeddings = torch.zeros(len(train_dataset), 256)
# images = torch.zeros(len(train_dataset), 1, 64, 128)

with torch.no_grad():
    for idx, x in tqdm(enumerate(train_dataset), total=len(train_dataset)):
        x = x.float()
        embedding, _ = model(x)
        embedding = embedding.view(-1)

        # print(embedding.shape)

        embeddings[idx] = embedding.squeeze()
        # images[idx] = x

# writer.add_embedding(embeddings, label_img=images)
# writer.close()



# run pca on embeddings

embeddings = embeddings.detach().numpy()


pca = PCA(n_components=4)

x_pca = pca.fit_transform(embeddings)

print(x_pca.shape)
