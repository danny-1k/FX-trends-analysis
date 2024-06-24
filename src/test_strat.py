import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt


df = pd.read_csv("../data/historical/forex/GBPCAD.csv")
df["Date"] = pd.to_datetime(df["Date"])
df.set_index("Date", inplace=True)

df["Day_of_week"] = df.index.isocalendar().day

df.sort_index(inplace=True)

print(df)

# strategy is buy on monday, sell on Tueday
# let's see how frequently we win.


bank = 50
bank_over_time = [bank]
position = 0
trades = []
win = []

prev = None

held = False

for i in range(len(df)):
    current = df.iloc[i]
    day = current["Day_of_week"]

    print(current["Close"])

    if day == 1: # monday

        if held == False:
            price = current["Close"]
            bank -= price * 40
            position = 1

            prev = price

    elif day == 2: # teusday
        if held == False:
            held = True
        else:
            price = current["Close"]
            if position == 1:
                bank += (40*price)
                position = 0

                if price > prev:
                    win.append(1)
                else:
                    win.append(0)

    bank_over_time.append(bank)



win = win[-100:]
avg_win = sum(win) / len(win)

print(avg_win)

plt.plot(bank_over_time[-100:])
plt.show()