import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
#test print statement
# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configuration
# feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
feature_cols = ['Open', 'High', 'Low', 'Close']
window_size = 20  # Window size for correlation calculation
n_jobs = -1  # Use all available cores (-1)

# Path setup
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(base_path)
# data_path = os.path.join(base_path, "data")
data_path = os.path.join(base_path, "data", "testbatch2")
print(data_path)
relation_path = os.path.join(data_path, "relations_test")
os.makedirs(relation_path, exist_ok=True)  # Create the directory if it doesn't exist
print(relation_path)
# stock_data_path = os.path.join(data_path, "NASDAQ_per_dag")
stock_data_path = os.path.join(data_path, "dailydata")
print(stock_data_path)

def load_all_stocks(stock_data_path):
    all_stock_data = []
    for file in tqdm(os.listdir(stock_data_path), desc="Loading data"):
        if file.endswith('.csv'):
            df = pd.read_csv(os.path.join(stock_data_path, file))
            all_stock_data.append(df[['Date', 'Stock'] + feature_cols])
    all_stock_data = pd.concat(all_stock_data, ignore_index=True)
    print(all_stock_data.head()) # kleine test om te zien of data deftig is ingeladen
    return all_stock_data

def calculate_correlation_matrix(combined_df, date_range):
    """Berekent correlaties per feature en middelt deze"""
    # print(f"\n=== Calculating correlations for date range: {date_range[0]} to {date_range[1]} ===")
    # Filter eerst op datum
    date_mask = (combined_df['Date'] >= date_range[0]) & (combined_df['Date'] <= date_range[1])
    filtered = combined_df.loc[date_mask]
    
    # Groepeer per stock en filter op window_size
    grouped = filtered.groupby('Stock')
    # Get all unique stocks
    unique_stocks = grouped.groups.keys()
    # print(f"Found {len(unique_stocks)} unique stocks in date range")
    
    # Directly create the 3D array without checking validity
    stock_arrays = np.stack([
        grouped.get_group(stock)[feature_cols].values.T 
        for stock in unique_stocks
    ])
    n_stocks = len(unique_stocks)
    # print(f"aantal unieke aandelen: {n_stocks}")
    corr_matrix = np.eye(n_stocks)
    # print("\nStarting correlation calculations...")
    # for i in tqdm(range(n_stocks), desc=f"Calculating {date_range[1]}"): # with or without tqdm?
    for i in range(n_stocks):
        for j in range(i+1, n_stocks):
            # stock_i = list(unique_stocks)[i]
            # stock_j = list(unique_stocks)[j]
            # print(f"\nCalculating correlations between {stock_i} and {stock_j}")
            # Bereken correlatie voor elke feature apart
            feature_correlations = []
            for f, feature in enumerate(feature_cols):
                # x = stock_arrays[i,f,:]
                # y = stock_arrays[j,f,:]
                
                # # Debug prints
                # print(f"  Feature {feature}:")
                # print(f"    {stock_i} values: {x}")
                # print(f"    {stock_j} values: {y}")
                corr = np.corrcoef(stock_arrays[i,f,:], stock_arrays[j,f,:])[0,1]
                feature_correlations.append(corr)
            
            # Gemiddelde correlatie over alle features
            avg_corr = np.round(np.nanmean(feature_correlations), 3)
            # print(f"  Average correlation: {avg_corr:.4f}")
            corr_matrix[i,j] = avg_corr
            corr_matrix[j,i] = avg_corr
    # print("\n=== Correlation matrix calculation complete ===")
    return pd.DataFrame(corr_matrix, index=unique_stocks, columns=unique_stocks)

def main(asc):
    # Load and filter data
    stock_data = load_all_stocks(stock_data_path)
    # Get all unique dates
    # print(type(stock_data))
    # print(stock_data['Date'])
    all_dates = stock_data['Date'].unique()
    all_dates.sort()
    # print(all_dates)
    print(f"Total unique dates: {len(all_dates)}")
    
    indices = range(window_size, len(all_dates))
    if not asc:
        indices = reversed(indices)

    # Process each time window
    for i in tqdm(indices, desc="Processing time windows"):
        end_date = all_dates[i]
        filename = os.path.join(relation_path, f"{end_date}.csv")
    
        # Skip if already exists
        if os.path.exists(filename):
            continue

        start_date = all_dates[i - window_size + 1]
        
        corr_matrix = calculate_correlation_matrix(stock_data, (start_date, end_date))
        
        if corr_matrix is not None:
            # Save results
            filename = os.path.join(relation_path, f"{end_date}.csv")
            corr_matrix.to_csv(filename)
        else:
            print(f"No data available for date range {start_date} to {end_date}. Skipping...")
            
if __name__ == "__main__":
    ascending = True #voor Jesse
    # ascending = False #voor Lawrence
    main(ascending)