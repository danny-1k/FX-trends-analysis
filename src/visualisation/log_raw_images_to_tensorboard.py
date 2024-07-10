import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from data.data import TrendDataset
from utils.data import SeriesToImageConverter
import pandas as pd
from tqdm import tqdm

converter = SeriesToImageConverter(height=64, width=128)
train_csv = pd.read_csv("../data/processed/splits/train.csv")
train_dataset = TrendDataset(df=train_csv, converter=converter)#TrendDataset(df=train_csv, converter=converter)
train_dataset = DataLoader(train_dataset, batch_size=1, shuffle=False)

writer = SummaryWriter()

embeddings = torch.zeros(len(train_dataset), 64*128)

with torch.no_grad():
    for idx, x in tqdm(enumerate(train_dataset), total=len(train_dataset)):
        embeddings[idx] = x.float().view(-1)

writer.add_embedding(embeddings)
writer.close()