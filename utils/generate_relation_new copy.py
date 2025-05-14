import os
import pandas as pd
from tqdm import tqdm
import warnings
import torch
import numpy as np

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configuration
feature_cols = ['Open', 'High', 'Low', 'Close']
window_size = 20  # Window size for correlation calculation
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Path setup
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(base_path)
data_path = os.path.join(base_path, "data", "testbatch_1000")
print(data_path)
relation_path = os.path.join(data_path, "correlations")
os.makedirs(relation_path, exist_ok=True)
print(relation_path)
stock_data_path = os.path.join(data_path, "dailydata")
print(stock_data_path)

def load_all_stocks(stock_data_path):
    """Load all stock data with progress tracking"""
    all_stock_data = []
    for file in tqdm(os.listdir(stock_data_path), desc="Loading data"):
        if file.endswith('.csv'):
            df = pd.read_csv(os.path.join(stock_data_path, file))
            all_stock_data.append(df[['Date', 'Stock'] + feature_cols])
    all_stock_data = pd.concat(all_stock_data, ignore_index=True)
    print(all_stock_data.head())
    return all_stock_data

def calculate_correlation_matrix_gpu(combined_df, date_range):
    """Calculate correlations per feature and average them, with diagonal set to 0"""
    date_mask = (combined_df['Date'] >= date_range[0]) & (combined_df['Date'] <= date_range[1])
    filtered = combined_df.loc[date_mask]
    
    grouped = filtered.groupby('Stock')
    unique_stocks = list(grouped.groups.keys())
    n_stocks = len(unique_stocks)
    
    # Create 3D tensor [n_stocks, n_features, window_size] on GPU
    stock_tensors = torch.stack([
        torch.tensor(grouped.get_group(stock)[feature_cols].values.T,
        dtype=torch.float32, device=device
    ) for stock in unique_stocks])  # Shape: [1000, 4, 20]
    
    # Initialize correlation matrix
    corr_matrix = torch.zeros((n_stocks, n_stocks), device=device)
    
    # Calculate correlation for each feature separately
    for feature_idx in range(len(feature_cols)):
        # Get data for this feature [1000, 20]
        feature_data = stock_tensors[:, feature_idx, :]
        
        # Normalize (z-score) the feature data
        means = torch.mean(feature_data, dim=1, keepdim=True)
        stds = torch.std(feature_data, dim=1, keepdim=True)
        normalized = (feature_data - means) / (stds + 1e-8)
        
        # Compute correlation matrix for this feature
        corr = torch.mm(normalized, normalized.T) / (window_size - 1)
        
        # Add to the accumulated correlations
        corr_matrix += corr
    
    # Average over all features
    corr_matrix /= len(feature_cols)
    
    # Set diagonal to 0 as requested
    corr_matrix.fill_diagonal_(0.0)
    
    return pd.DataFrame(
        corr_matrix.cpu().numpy(),
        index=unique_stocks,
        columns=unique_stocks
    ).round(3)

def main(asc):
    # Load and filter data
    stock_data = load_all_stocks(stock_data_path)
    all_dates = stock_data['Date'].unique()
    all_dates.sort()
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
        
        corr_matrix = calculate_correlation_matrix_gpu(stock_data, (start_date, end_date))
        
        if corr_matrix is not None:
            filename = os.path.join(relation_path, f"{end_date}.csv")
            corr_matrix.to_csv(filename)
        else:
            print(f"No data available for date range {start_date} to {end_date}. Skipping...")
            
if __name__ == "__main__":

    # Path setup
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print(base_path)
    data_path = os.path.join(base_path, "data", "NASDAQ_batches_5_200")
    print(data_path)

    for batchmap in os.listdir(data_path):
        print(batchmap)
        relation_path = os.path.join(data_path, batchmap, "correlations")
        os.makedirs(relation_path, exist_ok=True)
        print(relation_path)
        stock_data_path = os.path.join(data_path, batchmap, "dailydata")
        print(stock_data_path)

        ascending = True
        main(ascending)
