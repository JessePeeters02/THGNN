import os
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.spatial.distance import pdist, squareform

# Configuratie
FEATURE_COLS = ['Open', 'High', 'Low', 'Close', 'Volume']
WINDOW_SIZE = 20  # Aantal dagen voor correlatieberekening
MIN_TRADING_DAYS = 18  # Minimale dagen met data

# Pad configuratie
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(base_path, "data")
daily_data_path = os.path.join(data_path, "NASDAQ_per_dag")
relation_path = os.path.join(data_path, "relation_test")
os.makedirs(relation_path, exist_ok=True)

def load_daily_data(date):
    """Laad data voor een specifieke datum"""
    date_str = date.strftime('%Y-%m-%d')
    file_path = os.path.join(daily_data_path, f"{date_str}.csv")
    return pd.read_csv(file_path)

def compute_correlations_for_window(end_date, dates_in_window):
    """Bereken correlaties voor een tijdwindow"""
    window_data = []
    valid_stocks = None
    
    # Verzamel data voor het window
    for date in dates_in_window:
        try:
            daily_df = load_daily_data(date)
            if valid_stocks is None:
                valid_stocks = set(daily_df['Stock'])
            else:
                valid_stocks.intersection_update(set(daily_df['Stock']))
            window_data.append(daily_df)
        except FileNotFoundError:
            continue
    
    if not window_data or len(valid_stocks) < 2:
        return None
    
    # Bouw feature matrix
    valid_stocks = sorted(valid_stocks)
    features = {stock: [] for stock in valid_stocks}
    
    for daily_df in window_data:
        daily_df = daily_df[daily_df['Stock'].isin(valid_stocks)]
        for stock in valid_stocks:
            stock_data = daily_df[daily_df['Stock'] == stock][FEATURE_COLS].values
            if len(stock_data) > 0:
                features[stock].append(stock_data[0])
    
    # Controleer volledige windows
    features = {k: np.array(v) for k, v in features.items() if len(v) == WINDOW_SIZE}
    if len(features) < 2:
        return None
    
    # Bereken correlatiematrix (geoptimaliseerd)
    stock_names = sorted(features.keys())
    feature_matrix = np.array([features[name] for name in stock_names])
    
    # Vectorized correlatieberekening
    flat_features = feature_matrix.reshape(len(stock_names), -1)
    corr_matrix = np.corrcoef(flat_features)
    
    result_df = pd.DataFrame(corr_matrix, index=stock_names, columns=stock_names)
    result_df = result_df.fillna(0)
    np.fill_diagonal(result_df.values, 1)
    
    return result_df

def generate_all_correlations():
    """Genereer alle correlatiematrices"""
    # Verzamel alle beschikbare datums
    daily_files = [f for f in os.listdir(daily_data_path) if f.endswith('.csv')]
    all_dates = sorted([pd.to_datetime(f.split('.')[0]) for f in daily_files])
    
    # Bereken correlaties voor elk window
    for i in tqdm(range(WINDOW_SIZE, len(all_dates)), desc="Processing windows"):
        end_date = all_dates[i]
        start_date = all_dates[i - WINDOW_SIZE + 1]
        dates_in_window = all_dates[i - WINDOW_SIZE + 1:i + 1]
        
        t1 = time.time()
        result = compute_correlations_for_window(end_date, dates_in_window)
        t2 = time.time()
        
        if result is not None:
            result.to_csv(os.path.join(relation_path, f"{end_date.strftime('%Y-%m-%d')}.csv"))
            print(f"Processed {end_date.strftime('%Y-%m-%d')} in {t2-t1:.2f}s")

if __name__ == "__main__":
    generate_all_correlations()