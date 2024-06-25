import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
# import tensorflow
# import tensorboard
# tensorflow.io.gfile = tensorboard.compat.tensorflow_stub.io.gfile
from data import MLPTrendEmbeddingDataset
from utils import Converter
import pandas as pd
from tqdm import tqdm

converter = Converter(height=64, width=128)
train_csv = pd.read_csv("../data/processed/splits/train.csv")
train_dataset = MLPTrendEmbeddingDataset(df=train_csv, converter=converter)#TrendDataset(df=train_csv, converter=converter)
train_dataset = DataLoader(train_dataset, batch_size=1, shuffle=False)

writer = SummaryWriter()    

for idx, (embedding, image) in tqdm(enumerate(train_dataset), total=len(train_dataset)):
    writer.add_embedding(embedding, label_img=image, global_step=idx)
    writer.add_image("", torch.ones(3, 3, 3), global_step=0)
    # print(embedding.shape, image.shape)

print(len(train_dataset))


