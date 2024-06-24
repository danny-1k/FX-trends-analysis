import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce


if __name__ == "__main__":
    column = "Close"

    def read_prices(f):
        asset_name = f.split("/")[-1].split(".")[0]

        df = pd.read_csv(f)
        df["Date"] = pd.to_datetime(df["Date"])
        df.set_index("Date", inplace=True)

        df.sort_index(inplace=True)

        df = df[[column]]
        df.rename({column: asset_name}, axis=1, inplace=True)

        return df

    forex_prices = [read_prices(os.path.join("../data/historical/forex", f)) for f in os.listdir("../data/historical/forex")]
    commodity_prices = [read_prices(os.path.join("../data/historical/commodities", f)) for f in os.listdir("../data/historical/commodities")]

    merged_df = reduce(lambda left, right: pd.merge(left, right, left_index=True, right_index=True, how="outer"), [*forex_prices, *commodity_prices])

    merged_df.dropna(axis=0, inplace=True)


    if not os.path.exists("../data/processed/combined/combined.csv"):
        merged_df.to_csv("../data/processed/combined/combined.csv")



# plt.show()
# print(merged_df)

# print(commodity_prices[0])
# commodity_prices = [read_prices(f) for f in os.listdir("../data/historical/commodities")]


# plt.plot(df["Close"])
# plt.show()
# # print(df.columns)