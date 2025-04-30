import os
import pandas as pd
from tqdm import tqdm

# Configuratie
FEATURE_COLS = ['Open', 'High', 'Low', 'Close', 'Volume', 'Turnover']

# Pad configuratie
base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(f"base_path: {base_path}")
data_path = os.path.join(base_path, "data", "testbatch2")
print(f"data_path: {data_path}")
stock_data_path = os.path.join(os.path.dirname(base_path), "portfolio_construction", "data", "NASDAQ", "NASDAQ_data_wTO")
print(f"stock_data_path: {stock_data_path}")
stock_data_output = os.path.join(data_path, "stockdata")
os.makedirs(stock_data_output, exist_ok=True)  # Maak de map aan als deze nog niet bestaat
print(f"stock_data_output: {stock_data_output}")
daily_data_output = os.path.join(data_path, "dailydata")
os.makedirs(daily_data_output, exist_ok=True)  # Maak de map aan als deze nog niet bestaat
print(f"daily_data_output: {daily_data_output}")
best_stocks_path = os.path.join(data_path, "top_200_stocks.csv")  # Waar de resultaten worden gehaald
print(f"best_stocks_path: {best_stocks_path}")


def collect_topN_stock(input_path, output_path):
    stock_files = [f for f in os.listdir(input_path) if f.endswith('.csv')]
    best_stocks_csv = pd.read_csv(best_stocks_path)
    best_stocks = [f for f in best_stocks_csv['Stock']]
    stock_counter = 0

    for stock_file in tqdm(stock_files, desc="Transforming data structure"):
        stock_name = stock_file.split('.')[0]
        if stock_name in best_stocks:
            stock_counter += 1
            stock_df = pd.read_csv(os.path.join(input_path, stock_file))
            stock_df["Stock"] = stock_name
            stock_df = stock_df[['Date', 'Stock'] + FEATURE_COLS]
            stock_df.to_csv(os.path.join(output_path, f"{stock_name}.csv"), index=False)
    print(f"Stock counter: {stock_counter}")


def transform_to_daily_structure(input_path, output_path):
    stock_files = [f for f in os.listdir(input_path) if f.endswith('.csv')]
    stock_data = []

    for stock_file in tqdm(stock_files, desc="Transforming data structure"):

        stock_name = stock_file.split('.')[0]
        stock_df = pd.read_csv(os.path.join(input_path, stock_file))
        stock_df['Date'] = pd.to_datetime(stock_df['Date'])
        stock_df['Stock'] = stock_name
    
        stock_data.append(stock_df[['Date', 'Stock'] + FEATURE_COLS])
        # print(stock_data)

    combined_df = pd.concat(stock_data, ignore_index=True)

    # Sla op per dag
    for date, group in tqdm(combined_df.groupby('Date'), desc="Saving daily files"):
        date_str = date.strftime('%Y-%m-%d')
        group.to_csv(os.path.join(output_path, f"{date_str}.csv"), index=False)
    print(f"data saved in {output_path}")


collect_topN_stock(stock_data_path, stock_data_output)
transform_to_daily_structure(stock_data_output, daily_data_output)