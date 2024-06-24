import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils import create_image_from_time_series


def read_df(f):
    df = pd.read_csv(f)
    df = df[[column for column in list(df.columns) if "Unnamed" not in column]]

    df.set_index(df["Date"], inplace=True)
    df.drop(["Date"], axis=1, inplace=True)
    df.sort_index(inplace=True)

    return df


TIME_FRAME = 5
IMAGE_HEIGHT = 64
IMAGE_WIDTH = 128

train_df = read_df("../data/processed/splits/train.csv")
test_df = read_df("../data/processed/splits/test.csv")


num_train_rounds = train_df.shape[0] - TIME_FRAME - 1
num_test_rounds = test_df.shape[0] - TIME_FRAME - 1


for i in tqdm(range(num_train_rounds)):
    window = train_df.iloc[i:i + TIME_FRAME]
    image_labels = list(window.columns)

    images_from_windows = [
        create_image_from_time_series(window[column].to_numpy(), height=IMAGE_HEIGHT, width=IMAGE_WIDTH) for column in image_labels
    ]

    images_from_windows = [Image.fromarray(image * 255).convert("L") for image in images_from_windows]

    for j in range(len(image_labels)):
        images_from_windows[j].save(f"../data/processed/images/staggered/train/{i}_{image_labels[j]}.jpg")


for i in tqdm(range(num_test_rounds)):
    window = test_df.iloc[i:i + TIME_FRAME]
    image_labels = list(window.columns)

    images_from_windows = [
        create_image_from_time_series(window[column].to_numpy(), height=IMAGE_HEIGHT, width=IMAGE_WIDTH) for column in image_labels
    ]

    images_from_windows = [Image.fromarray(image * 255).convert("L") for image in images_from_windows]

    for j in range(len(image_labels)):
        images_from_windows[j].save(f"../data/processed/images/staggered/test/{i}_{image_labels[j]}.jpg")

# plt.imshow(np.concatenate(images, axis=1))
# plt.show()
    # print(window)
    # break