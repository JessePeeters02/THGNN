import os
import pandas as pd
from tqdm import tqdm

# Configuratie
FEATURE_COLS = ['Open', 'High', 'Low', 'Close', 'Volume']
FEATURE_COLS = ['Open', 'High', 'Low', 'Close', 'Volume', 'Turnover']

# Pad configuratie
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(base_path, "data")
daily_data_path = os.path.join(data_path, "NASDAQ_per_dag_wTO")
stock_data_path = os.path.join(os.path.dirname(base_path), "portfolio_construction", "data", "NASDAQ_data_wTO")  # Map waar de CSV-bestanden staan
os.makedirs(daily_data_path, exist_ok=True)  # Zorg dat de output map bestaat

def transform_to_daily_structure(stock_data_path, filter_non_trading):
    stock_files = [f for f in os.listdir(stock_data_path) if f.endswith('.csv')]
    stock_data = []
    non_trading_stocks = []
    # testcounter = 0

    for stock_file in tqdm(stock_files, desc="Transforming data structure"):
        # testcounter += 1
        # if testcounter > 10:
        #     break
        stock_name = stock_file.split('.')[0]
        stock_df = pd.read_csv(os.path.join(stock_data_path, stock_file))
        stock_df['Date'] = pd.to_datetime(stock_df['Date'])
        stock_df['Stock'] = stock_name
    
        # Filter aandelen met niet-handelsdagen (indien gewenst)
        if filter_non_trading:
            if (stock_df['Volume'] == 0).any():  # Controleer of er minstens één dag is met volume = 0
                non_trading_stocks.append(stock_name)
                continue  # Sla dit aandeel over
        
        stock_data.append(stock_df[['Date', 'Stock'] + FEATURE_COLS])
        # print(stock_data)

    combined_df = pd.concat(stock_data, ignore_index=True)

    if filter_non_trading:
        print(f"Aantal aandelen totaal: {len(stock_files)}")
        print(f"Aantal aandelen met niet-handelsdagen: {len(non_trading_stocks)}")
        print(f"Aantal aandelen in de huidige dataset: {len(stock_data)}")

    # Sla op per dag
    for date, group in tqdm(combined_df.groupby('Date'), desc="Saving daily files"):
        date_str = date.strftime('%Y-%m-%d')
        group.to_csv(os.path.join(daily_data_path, f"{date_str}.csv"), index=False)

stock_data = transform_to_daily_structure(stock_data_path, filter_non_trading=True)