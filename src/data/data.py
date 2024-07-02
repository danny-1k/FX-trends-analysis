import os
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from PIL import Image
import numpy as np



class TrendImageDataset(Dataset):
    """
    Loads Images and converts them to Torch Tensors.
    """
    
    def __init__(self, root, flatten=False):
        self.flatten = flatten
        self.images = [os.path.join(root, f) for f in os.listdir(root) if ".png" in f]
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        f = self.images[idx]
        x = np.asarray(Image.open(f))
        x = x / 255
        x = torch.from_numpy(x).float()

        if self.flatten:
            return x.view(-1)
        
        x = x.unsqueeze(0)

        return x
    

class TrendImageDatasetContrastive(Dataset):
    def __init__(self, root, flatten=False):
        self.flatten = flatten
        self.images = [os.path.join(root, f) for f in os.listdir(root) if ".png" in f]

        self.transform = transforms.RandomChoice([
                transforms.RandomRotation(degrees=(-10, 10)),
                transforms.RandomErasing(p=1),
                transforms.RandomAffine(degrees=(-10, 10), translate=(0.3, 0.3)),
            ])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # half the time, choose a random image from the dataset, 
        # other half, augment the current image.


        f = self.images[idx]
        x1 = self._get_img(idx)

        if np.random.random() > .5:
            x2 = self.transform(x1)
            y = 1  # y = 1 means they are the same

        else:
            x2 = self._get_img(np.random.randint(len(self)))
            y = 0 # y = 0 means they are not the same

        if self.flatten:
            x1 = x1.view(-1)
            x2 = x2.view(-1)

        
        return x1, x2, y


    def _get_img(self, idx):
        f = self.images[idx]
        x1 = np.asarray(Image.open(f)) / 255
        x1 = torch.from_numpy(x1).float()

        x1 = x1.unsqueeze(0)

        return x1

    
class TrendArrayDataset(Dataset):
    def __init__(self, root, flatten=False):
        self.flatten = flatten
        self.images = [os.path.join(root, f) for f in os.listdir(root) if ".npy" in f]

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        f = self.images[idx]
        x = np.load(f)
        x = torch.from_numpy(x).float()

        if self.flatten:
            return x.view(-1)
    
        x = x.unsqueeze(0)

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
    

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    d = TrendArrayDataset(root="../data/processed/images/staggered/train")

    print(d[0][0])

    plt.imshow(d[0][0])
    plt.show()