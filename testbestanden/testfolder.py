import pandas as pd
import os
from tqdm import tqdm
import numpy as np

# Definieer de kolommen die we willen gebruiken uit de CSV-bestanden
# feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Turnover']
feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
prev_date_num = 20
jump_threshold = 15
min_jump_duration = 5
cooldown_days = min_jump_duration
jump_cap = 50
outlier_cap = 30

# Basis pad naar de data-map
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Huidige scriptmap
print(base_path)
data_path = os.path.join(base_path, "data", "testbatch1")
print(data_path)
input_path = os.path.join(data_path, "stockdata")  # Map waar de CSV-bestanden staan
print(input_path)
# daily_data_path = os.path.join(data_path, "normaliseddailydatatest")  # Map waar de CSV-bestanden moeten komen
# os.makedirs(daily_data_path, exist_ok=True)  # Maak de map aan als deze nog niet bestaat
# print(daily_data_path)
stock_data_path = os.path.join(data_path, "test_normalisedstockdata")  # Map waar de CSV-bestanden moeten komen
os.makedirs(stock_data_path, exist_ok=True)  # Maak de map aan als deze nog niet bestaat
print(stock_data_path)


def load_stock_data(stock_data_path):
    stock_files = [f for f in os.listdir(stock_data_path) if f.endswith('.csv')]
    stock_data = {}
    for stock_file in tqdm(stock_files, desc="Loading stock data"):
        stock_name = stock_file.split('.')[0]
        # if stock_name == 'CHRD':
        df = pd.read_csv(os.path.join(stock_data_path, stock_file))
        df['Date'] = pd.to_datetime(df['Date'])
        stock_data[stock_name] = df
    return stock_data

# def normalize_with_jump_detection(
#     df, 
#     col, 
#     window, 
#     jump_threshold, 
#     min_jump_duration=5,
#     mode='reset'  # 'reset' of 'mask'
# ):
#     df = df.copy()
#     values = df[col].values.astype(np.float64)
#     normalized = np.zeros_like(values)
    
#     # 1. Bereken rolling statistics
#     rolling_median = pd.Series(values).rolling(window=window).median()
#     rolling_mad = pd.Series(values).rolling(window=window).apply(
#         lambda x: np.median(np.abs(x - np.median(x))) + 1e-6
#     )
    
#     # 2. Detecteer BLIJVENDE sprongen (min_jump_duration opeenvolgende dagen)
#     z_scores = (values - rolling_median) / rolling_mad
#     jump_candidates = (z_scores.abs() > jump_threshold)
#     jump_detected = jump_candidates.rolling(min_jump_duration).sum() >= min_jump_duration
#     jump_detected = jump_detected & ~jump_detected.shift(1).fillna(False)  # Alleen eerste dag van sprong
    
#     print(f"Aantal blijvende sprongen gedetecteerd: {jump_detected.sum()}")

#     # 3. Pas normalisatie toe per mode
#     current_region_start = 0
    
#     for i in range(len(values)):
#         if i < window:
#             normalized[i] = 0  # Eerste window niet normaliseren
#             continue
            
#         # Mode-specifieke logica
#         if mode == 'reset' and jump_detected.iloc[i]:
#             current_region_start = i  # Reset normalisatie-regio
#             print(f"Reset na sprong op {df['Date'].iloc[i]}")
            
#         # Bereken MAD voor huidige regio
#         region_values = values[max(current_region_start, i-window+1):i+1]
#         med = np.median(region_values)
#         mad = np.median(np.abs(region_values - med)) + 1e-6
        
#         # Mode-specifieke normalisatie
#         if mode == 'mask' and jump_detected.iloc[i]:
#             normalized[i] = np.nan  # Maskeren
#         else:
#             normalized[i] = (values[i] - med) / mad
            
#     return pd.Series(normalized, index=df.index), mode
def detect_jumps(df, master_col, window):
    values = df[master_col].values.astype(np.float64)

    # Bereken rolling median + mad
    rolling_median = pd.Series(values).rolling(window=window).median()
    rolling_mad = pd.Series(values).rolling(window=window).apply(
        lambda x: np.median(np.abs(x - np.median(x))) + 1e-6)

    z_scores = (values - rolling_median) / rolling_mad
    jump_candidates = (np.abs(z_scores) > jump_threshold)

    # Vooruitkijkend window (zoals je nu hebt)
    future_sum = pd.Series(jump_candidates.astype(int)).rolling(window=min_jump_duration).sum().shift(-(min_jump_duration - 1))
    jump_raw = (future_sum >= min_jump_duration).fillna(False)

    # ⏸ Cooldown toepassen: onderdruk detectie na een jump
    jump_detected = jump_raw.copy()
    for i in range(1, len(jump_raw)):
        if jump_detected.iloc[i - 1]:
            jump_detected.iloc[i:i + cooldown_days] = False  # suppress volgende dagen

    jump_dates = df['Date'][jump_detected].reset_index(drop=True)
    jump_z_scores = pd.Series(z_scores[jump_detected].values, index=jump_dates)

    print(f"Aantal jumps: {jump_detected.sum()}")
    for date, z in jump_z_scores.items():
        print(f"Jump op {date}: z-score = {z:.2f}")
    
    return jump_detected


def normalize_all_columns(df, jump_mask, window, mode='reset'):
    df = df.copy()
    cols = feature_cols  # Standaard kolommen
    
    for col in cols:
        values = df[col].values.astype(np.float64)
        normalized = np.zeros_like(values)
        current_region_start = 0
        
        for i in range(len(values)):
            if i < window:
                normalized[i] = 0
                continue
                
            if mode == 'reset' and jump_mask.iloc[i]:
                delta = values[i] - values[i - 1] if i > 0 else 1  # veiligheid
                normalized[i] = jump_cap if delta > 0 else -jump_cap
                current_region_start = i
                continue
                
            region_values = values[max(current_region_start, i-window+1):i+1]
            med = np.median(region_values)
            mad = np.median(np.abs(region_values - med)) + 1e-6
            
            norm_val = (values[i] - med) / mad
            if abs(norm_val) > outlier_cap:
                print(f'waarde overschrijdt cap: {norm_val} bij {df["Date"].iloc[i]} in kolom {col}')
            normalized[i] = np.clip(norm_val, -outlier_cap, outlier_cap)
        
        df[col] = normalized
    
    return df, mode

def normalise_stock_data(df):
    for stockname, df in tqdm(stock_data.items(), desc="Normalizing stock data"):
        df_normalized = df.copy()
        jump_mask = detect_jumps(df, 'Open', prev_date_num)
        # for col in feature_cols:
            # df_normalized[col] = df[col].rolling(window=prev_date_num, min_periods=prev_date_num).apply(lambda x: (x - x.median()) / (x.mad() + 1e-6))
            # Voorbeeldgebruik
        df_normalized, mode = normalize_all_columns(df, jump_mask, prev_date_num)
        df_normalized = df_normalized.iloc[prev_date_num:]
        df_normalized = df_normalized[['Date', 'Stock'] + feature_cols + ['Turnover']]
        df_normalized.to_csv(os.path.join(stock_data_path, f"{stockname}_{mode}.csv"), index=False)
        # for date in all_dates:
        #     date_data = df_normalized[df_normalized['Date'] == date]
        #     if not date_data.empty:
        #         date_data.to_csv(os.path.join(daily_data_path, f"{date}.csv"), mode='a', 
        #                         header=not os.path.exists(os.path.join(daily_data_path, f"{date}.csv")), 
        #                         index=False)

stock_data = load_stock_data(input_path)
# print(stock_data)

all_dates = sorted({date.strftime('%Y-%m-%d') for df in stock_data.values() for date in df['Date'].tolist()})
print(f"Unique dates determined: {len(all_dates)}")

normalise_stock_data(stock_data)