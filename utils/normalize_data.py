import pandas as pd
import os
from tqdm import tqdm
import numpy as np

# Definieer de kolommen die we willen gebruiken uit de CSV-bestanden
# feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Turnover']
feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
prev_date_num = 20

# Basis pad naar de data-map
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Huidige scriptmap
print(base_path)
data_path = os.path.join(base_path, "data", "testbatch1")
print(data_path)
input_path = os.path.join(data_path, "stockdata")  # Map waar de CSV-bestanden staan
print(input_path)
daily_data_path = os.path.join(data_path, "normaliseddailydata")  # Map waar de CSV-bestanden moeten komen
print(daily_data_path)
stock_data_path = os.path.join(data_path, "normalisedstockdata")  # Map waar de CSV-bestanden moeten komen
print(stock_data_path)


def load_stock_data(stock_data_path):
    stock_files = [f for f in os.listdir(stock_data_path) if f.endswith('.csv')]
    stock_data = {}
    for stock_file in tqdm(stock_files, desc="Loading stock data"):
        stock_name = stock_file.split('.')[0]
        df = pd.read_csv(os.path.join(stock_data_path, stock_file))
        df['Date'] = pd.to_datetime(df['Date'])
        stock_data[stock_name] = df
    return stock_data


def normalise_stock_data(df):
    for stockname, df in tqdm(stock_data.items(), desc="Normalizing stock data"):
        df_normalized = df.copy()
        for col in feature_cols:
            # df_normalized[col] = df[col].rolling(window=prev_date_num, min_periods=prev_date_num).apply(lambda x: (x - x.median()) / (x.mad() + 1e-6))
            df_normalized[col] = (df[col] - df[col].rolling(window=prev_date_num).median()) / \
                     (df[col].rolling(window=prev_date_num).apply(lambda x: np.median(np.abs(x - x.median()))) + 1e-6)
        df_normalized = df_normalized.iloc[prev_date_num:]
        df_normalized.to_csv(os.path.join(stock_data_path, f"{stockname}.csv"), index=False)
        for date in all_dates:
            date_data = df_normalized[df_normalized['Date'] == date]
            if not date_data.empty:
                date_data.to_csv(os.path.join(daily_data_path, f"{date}.csv"), mode='a', 
                                header=not os.path.exists(os.path.join(daily_data_path, f"{date}.csv")), 
                                index=False)
        # for i in range(prev_date_num-1, len(all_dates)):
        #     iend = i
        #     enddt = all_dates[i]
        #     istart = iend - prev_date_num + 1
        #     startdt = all_dates[istart]    
        #     df_window = df[(df['Date'] >= startdt) & (df['Date'] <= enddt)]
        #     print(f"df window shape: {df_window.shape}")
        #     # Robuuste standaardisatie per feature
        #     for col in feature_cols:
        #         median = df_window[col].median()
        #         mad = (df_window[col] - median).abs().median()  # Median Absolute Deviation
        #         df_window[col] = (df_window[col] - median) / mad

stock_data = load_stock_data(input_path)
# print(stock_data)

all_dates = sorted({date.strftime('%Y-%m-%d') for df in stock_data.values() for date in df['Date'].tolist()})
print(f"Unique dates determined: {len(all_dates)}")

normalise_stock_data(stock_data)
