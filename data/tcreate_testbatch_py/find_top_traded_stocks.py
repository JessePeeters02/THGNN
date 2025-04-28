import os
import pandas as pd
from tqdm import tqdm

# Configuratie
base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(f"base_path: {base_path}")
stock_data_path = os.path.join(os.path.dirname(base_path), "portfolio_construction", "data", "NASDAQ", "NASDAQ_data_wTO")
print(f"stock_data_path: {stock_data_path}")
topN = 10  # Aantal aandelen om te selecteren
output_path = os.path.join(base_path, "data", "testbatch_mini", f"top_{topN}_stocks.csv")  # Waar de resultaten worden opgeslagen
os.makedirs(os.path.dirname(output_path), exist_ok=True)  # Maak de map aan als deze nog niet bestaat
print(f"output_path: {output_path}")

def get_top_traded_stocks(stock_data_path, top_n):
    stock_files = [f for f in os.listdir(stock_data_path) if f.endswith('.csv')]
    stock_volumes = []

    print("Analyseren van handelsvolumes...")
    no_volume_counter = 0
    for stock_file in tqdm(stock_files):
        stock_name = stock_file.split('.')[0]
        df = pd.read_csv(os.path.join(stock_data_path, stock_file))
        
        if 'Volume' not in df.columns:
            print(f"Waarschuwing: Geen 'Volume'-kolom in {stock_file}")
            continue
        if (df['Volume'] == 0).any():  # Controleer of er minstens één dag is met volume = 0
            # print(f"Waarschuwing: bevat ergens 0 volume {stock_file}")
            no_volume_counter += 1
            continue  # Sla dit aandeel over
        
        avg_volume = df['Volume'].mean()
        stock_volumes.append((stock_name, avg_volume))

    print(f"{no_volume_counter} stocks met ergens geen volume")
    # Sorteer op volume (hoog naar laag) en selecteer de top N
    stock_volumes.sort(key=lambda x: x[1], reverse=True)
    top_stocks = [stock[0] for stock in stock_volumes[:top_n]]

    # Sla resultaten op in een CSV
    pd.DataFrame({"Stock": top_stocks}).to_csv(output_path, index=False)
    print(f"Top {top_n} meest verhandelde aandelen opgeslagen in {output_path}")

if __name__ == "__main__":
    get_top_traded_stocks(stock_data_path, topN)