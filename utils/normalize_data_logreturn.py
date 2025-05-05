import pandas as pd
import os
from tqdm import tqdm
import numpy as np
import shutil
from collections import defaultdict

# Definieer de kolommen die we willen gebruiken uit de CSV-bestanden
feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
price_cols = ['Open', 'High', 'Low', 'Close']

# Basis pad naar de data-map
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(base_path)
data_path = os.path.join(base_path, "data", "testbatch2")
print(data_path)
input_path = os.path.join(data_path, "stockdata")
daily_data_path = os.path.join(data_path, "normaliseddailydata_log")
stock_data_path = os.path.join(data_path, "normalisedstockdata_log")

# Maak directories aan
if os.path.exists(daily_data_path):
    shutil.rmtree(daily_data_path)
os.makedirs(daily_data_path, exist_ok=True)
os.makedirs(stock_data_path, exist_ok=True)

def load_stock_data(stock_data_path):
    stock_files = [f for f in os.listdir(stock_data_path) if f.endswith('.csv')]
    stock_data = {}
    for stock_file in tqdm(stock_files, desc="Loading stock data"):
        stock_name = stock_file.split('.')[0]
        df = pd.read_csv(os.path.join(stock_data_path, stock_file))
        df['Date'] = pd.to_datetime(df['Date'])
        stock_data[stock_name] = df
    return stock_data

def calculate_log_returns(df):
    df = df.copy()
    
    # Prijs log returns
    for col in price_cols:
        df[col] = np.log(df[col] / df[col].shift(1))
    
    # Volume log transform (niet returns maar absolute waarden)
    if 'Volume' in df.columns:
        df['Volume'] = np.log1p(df['Volume'])  # log(1 + x) voor volume
    
    return df.iloc[1:]  # Verwijder eerste rij met NaN

def normalize_with_clipping(df, sigma=5):
    df = df.copy()
    for col in df.columns:
        if col in ['Date', 'Stock', 'Turnover']:
            continue
            
        # Bereken robuste statistieken
        median = df[col].median()
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1 + 1e-6
        
        # Clip extreme waarden (standaard 5 IQR)
        lower_bound = median - sigma * iqr
        upper_bound = median + sigma * iqr
        df[col] = df[col].clip(lower_bound, upper_bound)
    
    return df

def process_all_stocks(stock_data):
    daily_data_buffer = defaultdict(list)
    
    for stock_name, df in tqdm(stock_data.items(), desc="Processing stocks"):
        # Stap 1: Bereken log returns
        df_processed = calculate_log_returns(df)
        
        # Stap 2: Pas clipping toe
        # df_processed = normalize_with_clipping(df_processed)
        
        # Sla op per stock
        output_cols = ['Date', 'Stock'] + feature_cols + ['Turnover']
        df_processed[output_cols].to_csv(
            os.path.join(stock_data_path, f"{stock_name}.csv"), 
            index=False
        )
        
        # Groepeer per datum voor daily files
        for date, day_data in df_processed.groupby('Date'):
            daily_data_buffer[date.strftime('%Y-%m-%d')].append(day_data)
    
    # Schrijf daily files
    for date, chunks in tqdm(daily_data_buffer.items(), desc="Writing daily files"):
        pd.concat(chunks, ignore_index=True).to_csv(
            os.path.join(daily_data_path, f"{date}.csv"), 
            index=False
        )

# Hoofdproces
print("Starting normalization using log-returns...")
stock_data = load_stock_data(input_path)
all_dates = sorted({date.strftime('%Y-%m-%d') for df in stock_data.values() for date in df['Date'].tolist()})
print(f"Found {len(all_dates)} unique dates across {len(stock_data)} stocks")

process_all_stocks(stock_data)
print("Normalization completed successfully!")