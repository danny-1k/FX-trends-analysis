import os
import pandas as pd
from PIL import Image
from tqdm import tqdm
from utils.data import SeriesToImageConverter


def read_df(f):
    df = pd.read_csv(f)
    df = df[[column for column in list(df.columns) if "Unnamed" not in column]]

    df.set_index(df["Date"], inplace=True)
    df.drop(["Date"], axis=1, inplace=True)
    df.sort_index(inplace=True)

    return df


class TrendImageGenerator:
    def __init__(self, image_height, image_width, time_frame, root="../data/processed"):
        self.image_height = image_height
        self.image_width = image_width
        self.time_frame = time_frame
        self.root = root

        self.train, self.test = self._read_dfs()

        self.num_train_points = self.train.shape[0] // self.time_frame - 1
        self.num_test_points = self.test.shape[1] // self.time_frame - 1

        self.series_converter = SeriesToImageConverter(height=self.image_height, width=self.image_width)

    def _read_dfs(self):
        train_df = read_df(os.path.join(self.root, "splits/train.csv"))
        test_df = read_df(os.path.join(self.root, "splits/test.csv"))

        return train_df, test_df

    def _run_one_split(self, split):
        # After generating the images, we save them as png files (because it's lossless)

        for i in tqdm(range(self.num_train_points if split == "train" else self.num_test_points)):
            if split == "train":
                chunk = self.train.iloc[i:i + self.time_frame]
            else:
                chunk = self.test.iloc[i:i + self.time_frame]
            column_names = list(chunk.columns)

            images_from_windows = [self.series_converter(chunk[column_name].to_numpy()) for column_name in column_names]

            # convert image arrays to PIL Images

            images_from_windows = [Image.fromarray(image * 255).convert("L") for image in images_from_windows]

            for j in range(len(column_names)):
                image_label = f"{i}_{column_names[j]}"
                images_from_windows[j].save(os.path.join(self.root, f"images/staggered/{split}/{image_label}.png"))
            
    def run(self):
        self._run_one_split("train")
        self._run_one_split("test")


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()

    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--time_frame", type=int, default=5)
    parser.add_argument("--root", type=str, default="../data/processed")

    args = parser.parse_args()

    height = int(args.height)
    width = int(args.width)
    time_frame = int(args.time_frame)
    root = str(args.root)

    generator = TrendImageGenerator(
        image_height=height, 
        image_width=width, 
        time_frame=time_frame, 
        root=root
    )

    generator.run()
