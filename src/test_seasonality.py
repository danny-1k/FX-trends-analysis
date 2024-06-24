import numpy as np

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("../data/historical/forex/GBPAUD.csv")


def format_df(df):
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)
    return df

def normalise_weekly(df):
    normalised_df = df.copy()
    normalised_df["week_num"] = normalised_df.index.isocalendar().week.apply(lambda x: str(x))
    normalised_df["week_num"] = normalised_df["week_num"] + normalised_df.index.isocalendar().year.apply(lambda x:str(x))

    grouped = normalised_df.groupby("week_num")["Close"]

    normalised_df = grouped.apply(lambda x: (x-x.min()) / (x.max() - x.min()))
    normalised_df = normalised_df.reset_index(level="week_num", drop=True)

    return normalised_df



gbp_aud = normalise_weekly(format_df(pd.read_csv("../data/historical/forex/GBPAUD.csv")))
gbp_cad = normalise_weekly(format_df(pd.read_csv("../data/historical/forex/GBPCAD.csv")))
gbp_chf = normalise_weekly(format_df(pd.read_csv("../data/historical/forex/GBPCHF.csv")))




gbp_cad_10_weeks = gbp_cad.to_list()[:250]
# interp_cad_10_weeks = np.interp(np.linspace(0, 49, 100), np.linspace(0, 49, 50), gbp_cad_10_weeks)

from statsmodels.tsa.seasonal import seasonal_decompose


result = seasonal_decompose(gbp_cad_10_weeks, model="additive", period=5)

seasonal = result.seasonal
trend = result.trend

trend = np.nan_to_num(trend, 0)

# plt.plot(trend)
# plt.show()

# There could be seasonality in the trend

trend_decomposition_results = seasonal_decompose(trend, model="additive", period=5)
trend_seasonal = trend_decomposition_results.seasonal
trend_trend = trend_decomposition_results.trend

# plt.plot(trend_trend)
# plt.plot(trend_seasonal)


season_comb = seasonal + trend_seasonal

# plt.plot(season_comb[:15])


plt.plot(gbp_cad_10_weeks[:25])

plt.show()
# plt.plot(gbp_cad.tolist()[:50], label="GBPxCAD")
# plt.plot(np.linspace(0, 49, 2*50), interp_cad_10_weeks)
# # plt.plot(gbp_chf.tolist(), label="GBPxCHF")

# plt.legend()
# plt.show()


# plt.plot(df[["Close"]])
# plt.show()
# print(df)