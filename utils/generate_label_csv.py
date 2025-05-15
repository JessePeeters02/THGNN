import os
import pandas as pd
from tqdm import tqdm

# Basis pad naar de data-map
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Huidige scriptmap
print(base_path)
data_path = os.path.join(base_path, "data", "NASDAQ_batches_5_200", "batch_2")
print(data_path)
raw_data_path = os.path.join(data_path, "stockdata")
print(raw_data_path)
labels_csv_path = os.path.join(data_path, "stock_labels.csv")

def load_stock_data(raw_stock_path):

    raw_files = [f for f in os.listdir(raw_stock_path) if f.endswith('.csv')]
    raw_data = {}
    for file in tqdm(raw_files, desc="Loading raw data"):
        stock_name = file.split('.')[0]
        df = pd.read_csv(os.path.join(raw_stock_path, file), parse_dates=['Date'])
        raw_data[stock_name] = df
    
    return raw_data

def calculate_label(raw_df, current_date):
    date_idx = raw_df[raw_df['Date'] == current_date].index[0]
    # print(date_idx)
    close_today = raw_df.iloc[date_idx]['Close']
    close_yesterday = raw_df.iloc[date_idx-1]['Close']
    return (close_today / close_yesterday) - 1

def create_labels_csv(raw_data, all_dates, output_path):
    # Maak een lijst van alle aandelen
    stock_names = sorted(raw_data.keys())
    
    # Maak een DataFrame met aandelen als index en datums als kolommen
    labels_df = pd.DataFrame(index=stock_names, columns=all_dates)
    
    # Vul de DataFrame met labels voor elke aandeel en elke datum
    for stock_name in tqdm(stock_names, desc="Processing stocks"):
        raw_df = raw_data[stock_name]
        for date_str in all_dates:
            try:
                current_date = pd.to_datetime(date_str)
                # Controleer of de datum bestaat in de raw data
                if current_date in raw_df['Date'].values:
                    label = round(calculate_label(raw_df, current_date),6)
                    labels_df.at[stock_name, date_str] = label
            except:
                # Als er een fout optreedt (bv. eerste dag heeft geen vorige dag), laat dan NaN staan
                pass
    
    # Sla de DataFrame op als CSV
    labels_df.to_csv(output_path)
    print(f"Labels saved to {output_path}")

# [Na het laden van de data...]
stock_data = load_stock_data(raw_data_path)
all_dates = sorted({date.strftime('%Y-%m-%d') for df in stock_data.values() for date in df['Date'].tolist()})

# Maak de labels CSV
create_labels_csv(stock_data, all_dates, labels_csv_path)