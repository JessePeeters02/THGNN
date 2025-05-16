import os
import torch
import pickle
import numpy as np
from tqdm import tqdm
import networkx as nx
import pandas as pd
from torch.autograd import Variable

# Definieer de kolommen die we willen gebruiken uit de CSV-bestanden
# feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Turnover']
prev_date_num = 20

# Basis pad naar de data-map
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Huidige scriptmap
print(base_path)
data_path = os.path.join(base_path, "data", "S&P500")
print(data_path)
raw_data_path = os.path.join(data_path, "stockdata")
print(raw_data_path)
stock_data_path = os.path.join(data_path, "normalisedstockdata")  # Map waar de CSV-bestanden staan
print(stock_data_path)


# Functie om de CSV-bestanden in te lezen en om te zetten naar een DataFrame
def load_stock_data(raw_stock_path, stock_data_path):

    raw_files = [f for f in os.listdir(raw_stock_path) if f.endswith('.csv')]
    raw_data = {}
    for file in tqdm(raw_files, desc="Loading raw data"):
        stock_name = file.split('.')[0]
        df = pd.read_csv(os.path.join(raw_stock_path, file), parse_dates=['Date'])
        raw_data[stock_name] = df

    stock_files = [f for f in os.listdir(stock_data_path) if f.endswith('.csv')]
    stock_data = {}
    for stock_file in tqdm(stock_files, desc="Loading stock data"):
        stock_name = stock_file.split('.')[0]
        df = pd.read_csv(os.path.join(stock_data_path, stock_file))
        df['Date'] = pd.to_datetime(df['Date'])
        stock_data[stock_name] = df
    
    return raw_data, stock_data

def calculate_label(raw_df, current_date):
    date_idx = raw_df[raw_df['Date'] == current_date].index[0]
    # print(date_idx)
    close_today = raw_df.iloc[date_idx]['Close']
    close_yesterday = raw_df.iloc[date_idx-1]['Close']
    return (close_today / close_yesterday) - 1

def random_graph_generator(lenght):

    threshold_pos = torch.empty(1).uniform_(0.60, 0.9).item()  # Alleen sterk positieve relaties
    threshold_neg = torch.empty(1).uniform_(0.09, 0.35).item()  # Alleen sterk negatieve relaties

    rand_mat = torch.rand(lenght, lenght)
    rand_mat = (rand_mat + rand_mat.T) / 2
    rand_mat.fill_diagonal_(0)

    pos_adj = (rand_mat > torch.empty(1).uniform_(threshold_pos-0.07, threshold_pos+0.07).item()).float()
    neg_adj = ((rand_mat < torch.empty(1).uniform_(threshold_neg-0.07, threshold_neg+0.07).item()) & (pos_adj == 0)).float()
    
    return pos_adj, neg_adj

# Functie om de relatiegrafieken te verwerken
def fun(iend, enddt, stock_data, pdn):
    istart = iend - pdn + 1
    startdt = all_dates[istart]
    # print(f"calculating window: {startdt} to {enddt}")
    
    pos_adj, neg_adj = random_graph_generator(len(stock_data))
        
    features = []
    labels = []
    day_last_code = []
        
    for stock_name, df in stock_data.items():
        df_window = df[(df['Date'] >= startdt) & (df['Date'] <= enddt)]
        # print(f"df window shape: {df_window.shape}")
        # print(f"df window: {df_window}")
        day_last_code.append([stock_name, enddt])
        raw_df = raw_data[stock_name]
        if len(df_window) == pdn:
            features.append(df_window[feature_cols].values)
            label = calculate_label(raw_df, end_date)
            labels.append(label)
        else:
            print(' huh, len df window pdn????')
            break
        
    output = {
        'pos_adj': torch.FloatTensor(pos_adj),
        'neg_adj': torch.FloatTensor(neg_adj),
        'features': torch.FloatTensor(np.array(features)),
        'labels': torch.FloatTensor(labels),
        'mask': [True] * len(labels)  # Alle samples zijn geldig
    }

    with open(os.path.join(data_train_predict_path, f"{enddt}.pkl"), 'wb') as f:
        pickle.dump(output, f)
    df = pd.DataFrame(columns=['code', 'dt'], data=day_last_code)
    df.to_csv(os.path.join(daily_stock_path, f"{enddt}.csv"), header=True, index=False, encoding='utf_8_sig')

# Laad de stock data
raw_data, stock_data = load_stock_data(raw_data_path, stock_data_path)
# print(stock_data)

# Bepaal de unieke datums uit de data
all_dates = sorted({date.strftime('%Y-%m-%d') for df in stock_data.values() for date in df['Date'].tolist()})
# print(f"Unique dates determined: {len(all_dates)}")
# print(all_dates)

for i in tqdm(range(prev_date_num-1, len(all_dates)), desc=f"Processing dates"):
    end_date = all_dates[i]
    data_train_predict_path = os.path.join(data_path, "data_train_predict_random1")
    os.makedirs(data_train_predict_path, exist_ok=True)
    daily_stock_path = os.path.join(data_path, "daily_stock_random1")
    os.makedirs(daily_stock_path, exist_ok=True)
    fun(i, end_date, stock_data, prev_date_num)