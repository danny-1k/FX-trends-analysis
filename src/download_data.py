"""
Downloads data accross groups: Coins, Commodities (ETFs) and Stocks (SnP500 & Stoxx600 consituents) using the EOD api
"""


import os
import requests
from tqdm import tqdm
import pandas as pd


class DataDownloader:
    def __init__(self, root="../data/groups", to="../data/historical"):
        self.root = root
        self.to = to
        self.api_key = os.environ.get("ALPHAVANTAGE")
        if not self.api_key:
            raise ValueError("`ALPHAVANTAGE` env variable not set.")

    def pull_data(self, ticker):

        def convert_to_dataframe_json(json):
            """
            Input json of form:
            {
                "2000-01-01": {
                    "1. open": xxx,
                    "2. high": xxx,
                },
                ...
            }

            Pandas requires json dataframe to be of form:
            {
                "open": {
                    "2000-01-01": xxx,
                    "2000-02-01": xxx,
                    ...
                },
                ...
            }

            """

            dates = list(json.keys())
            columns = ["Open", "High", "Low", "Close"] # volume data is useless for forex
            new_json = {column:{} for column in columns }

            for date in dates:
                data = json[date]
                # print(data["1. Information"])
                # print(data[list(data.keys())[0]])
                new_json["Open"][date] = data["1. open"]
                new_json["High"][date] = data["2. high"]
                new_json["Low"][date] = data["3. low"]
                new_json["Close"][date] = data["4. close"]

            return new_json


        uri = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker}&outputsize=full&apikey={self.api_key}"
        res = requests.get(uri)
        data = res.json()
        data = data["Time Series (Daily)"]
        data = convert_to_dataframe_json(data) # convert to proper format
        data_frame = pd.DataFrame(data)
        return data_frame


    def start(self):
        for group in os.listdir(self.root):
            tickers = open(os.path.join(self.root, group), "r").read().split("\n")
            print(f"Processing: {group}")
            for ticker in tqdm(tickers):
                group_ = group.split(".")[0]
                
                if os.path.exists(os.path.join(self.to, f"{group_}/{ticker}.csv")):
                    continue
                if not os.path.exists(os.path.join(self.to, group_)):
                    os.makedirs(os.path.join(self.to, group_))

                df = self.pull_data(ticker)
                df.to_csv(os.path.join(self.to, f"{group_}/{ticker}.csv"))


if __name__ == "__main__":
    data_downloader = DataDownloader()
    data_downloader.start()