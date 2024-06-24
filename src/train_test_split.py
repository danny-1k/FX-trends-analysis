import pandas as pd

TRAIN_PCT = 0.6 # because we need a sufficient testing period, we have to use a relatively small training pct


combined_df = pd.read_csv("../data/processed/combined/combined.csv")

num_samples = combined_df.shape[0]
num_train = int(TRAIN_PCT * num_samples)

train = combined_df.iloc[:num_train]
test = combined_df.iloc[num_train:]

train.to_csv("../data/processed/splits/train.csv", index=False)
test.to_csv("../data/processed/splits/test.csv", index=False)