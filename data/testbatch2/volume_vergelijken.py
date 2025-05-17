import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np

# === CONFIG ===
data_path = os.path.dirname(os.path.abspath(__file__))
stockdata = os.path.join(data_path, "stockdata")
start_date = None  # bv. "2020-01-01"
end_date = None    # bv. "2021-01-01"

# === Inladen van alle CSV's ===
all_data = []
for file in os.listdir(stockdata):
    if file.endswith(".csv"):
        df = pd.read_csv(os.path.join(stockdata, file))
        all_data.append(df)

df_all = pd.concat(all_data, ignore_index=True)
df_all["Date"] = pd.to_datetime(df_all["Date"])

# Filteren op datum als nodig
if start_date:
    df_all = df_all[df_all["Date"] >= pd.to_datetime(start_date)]
if end_date:
    df_all = df_all[df_all["Date"] <= pd.to_datetime(end_date)]

# Extra kolom: Volume × Close
df_all["VolumeClose"] = df_all["Volume"] * df_all["Close"]

# Zet 'Date' als index voor plotgemak
df_all.set_index("Date", inplace=True)

# === Plotfunctie per variabele ===
def plot_variable(variable_name, ylabel):
    plt.figure(figsize=(14, 7))
    for stock, group in df_all.groupby("Stock"):
        plt.plot(group.index, group[variable_name], label=stock, linewidth=1)
    plt.title(f"{variable_name} per aandeel over tijd")
    plt.xlabel("Datum")
    plt.ylabel(ylabel)
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize="small", ncol=1)
    plt.tight_layout()
    plt.grid(True)
    plt.show()


def plot_variable_log(variable_name, ylabel):
    plt.figure(figsize=(14, 7))
    
    # Voor de gemiddelden per dag
    daily_group = df_all.groupby("Date")[variable_name]

    # Voor elk aandeel: log10-waarden plotten
    for stock, group in df_all.groupby("Stock"):
        y = group[variable_name].copy()
        y[y <= 0] = np.nan  # log10 kan geen 0 of negatieve waarden aan
        y_log = np.log10(y)
        plt.plot(group.index, y_log, label=stock, linewidth=1, alpha=0.4)
    
    # Gemiddelde log10-waarde per dag (van alle aandelen)
    daily_mean = daily_group.mean()
    daily_mean[daily_mean <= 0] = np.nan
    log_daily_mean = np.log10(daily_mean)

    # Optioneel: rolling mean van 7 dagen
    rolling_log_mean = log_daily_mean.rolling(window=7, min_periods=1).mean()

    # Plot de gemiddelde lijn
    plt.plot(log_daily_mean.index, rolling_log_mean, label="Gemiddelde (7d)", color="black", linewidth=2.5, linestyle="--")

    # Labels en layout
    plt.title(f"Log10({variable_name}) per aandeel + gemiddelde trend")
    plt.xlabel("Datum")
    plt.ylabel(f"log10({ylabel})")
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize="small", ncol=1)
    plt.tight_layout()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.show()

# === 4 plots maken ===
plot_variable_log("Close", "Sluitprijs")
plot_variable_log("Volume", "Volume")
plot_variable_log("Turnover", "Turnover")
plot_variable_log("VolumeClose", "Volume × Close")

