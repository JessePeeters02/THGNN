import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torchdiffeq import odeint
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
import os
import torch.nn.functional as F
import time


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu') 
print(f"Device: {device}")

# Hyperparameters
prev_date_num = 20
feature_cols1 = ['Open', 'High', 'Low', 'Close']
feature_cols2 = ['Open', 'High', 'Low', 'Close', 'Volume', 'Turnover']
restrict_last_n_days= None # None of bv 80 om da laatse 60 dagen te nemen (20-day time window geraak je in begin altijd kwijt)

min_neighbors = 3
sim_threshold_pos = 0.4
sim_threshold_neg = -0.4

edge_evaluation = True

def load_all_stocks(stock_data_path):
    all_stock_data = []
    for file in tqdm(os.listdir(stock_data_path), desc="Loading normalised data"):
        if file.endswith('.csv'):
            df = pd.read_csv(os.path.join(stock_data_path, file))
            all_stock_data.append(df[['Date', 'Stock'] + feature_cols2])
    all_stock_data = pd.concat(all_stock_data, ignore_index=True)
    print(all_stock_data.head()) # kleine test om te zien of data deftig is ingeladen

        # Enkel laatste X dagen
    if restrict_last_n_days is not None:
        all_dates = sorted(all_stock_data['Date'].unique())
        last_dates = all_dates[-restrict_last_n_days:]
        all_stock_data = all_stock_data[all_stock_data['Date'].isin(last_dates)]

    print(all_stock_data.head())  # test of data deftig is

    return all_stock_data

def load_raw_stocks(raw_stock_path, all_dates):
    raw_files = [f for f in os.listdir(raw_stock_path) if f.endswith('.csv')]
    raw_data = {}
    for file in tqdm(raw_files, desc="Loading raw data for label creation"):
        stock_name = file.split('.')[0]
        df = pd.read_csv(os.path.join(raw_stock_path, file), parse_dates=['Date'])
        # if restrict_last_n_days is not None:
            # all_dates = sorted(df['Date'].unique())
            # last_dates = all_dates[-restrict_last_n_days:]
        df = df[df['Date'].isin(all_dates)]
        df = df.reset_index(drop=True)
        raw_data[stock_name] = df[['Date', 'Stock'] + feature_cols1]
    return raw_data

def build_initial_edges_via_cosine_similarity(window_data):
    def gpu_featurewise_cosine(stock_tensor: torch.Tensor):
        """
        stock_tensor: (n_stocks, n_features, n_days)
        Retourneert: (n_stocks, n_stocks) gem. cosine similarity over features
        """
        n_stocks, n_feat, n_days = stock_tensor.shape
        sims = []
        for f in range(n_feat):
            feat_f = stock_tensor[:, f, :]  # (n_stocks, n_days)
            feat_f = F.normalize(feat_f, p=2, dim=1)
            sim_f = torch.mm(feat_f, feat_f.T)  # (n_stocks, n_stocks)
            sims.append(sim_f)
        mean_sim = sum(sims) / len(sims)  # gemiddelde over features
        return mean_sim

    # Data preparatie
    grouped = window_data.groupby('Stock')[feature_cols1]
    stock_arrays = np.array([group.values.T for _, group in grouped])  # (n_stocks, n_features, n_days)
    n_stocks = stock_arrays.shape[0]

    # Cosine similarity matrix op GPU, per feature gemiddeld
    stock_tensor = torch.tensor(stock_arrays, dtype=torch.float32, device=device)
    cos_matrix = gpu_featurewise_cosine(stock_tensor).cpu().numpy()

    # Bouw edges op basis van drempelwaarde
    pos_edges = []
    neg_edges = []

    # Garandeer minimum aantal buren
    for i in range(n_stocks):
        # Positieve edges
        strong_pos = np.where(cos_matrix[i] > sim_threshold_pos)[0]
        if len(strong_pos) < min_neighbors:
            # Voeg extra buren toe als er te weinig zijn
            cos_vals = cos_matrix[i].copy()
            top_pos = np.argsort(-cos_vals)[:min_neighbors]
            for j in top_pos:
                if cos_matrix[i,j] > 0:  # Alleen positieve correlaties toevoegen
                    pos_edges.append((i, j))
                    pos_edges.append((j, i))
        else:
            # Gebruik alleen de sterke correlaties
            for j in strong_pos:
                pos_edges.append((i, j))
                pos_edges.append((j, i))
        
        # Negatieve edges
        strong_neg = np.where(cos_matrix[i] < sim_threshold_neg)[0]
        if len(strong_neg) < min_neighbors:
            # Voeg extra buren toe als er te weinig zijn
            cos_vals = cos_matrix[i].copy()
            top_neg = np.argsort(cos_vals)[:min_neighbors]
            for j in top_neg:
                if cos_matrix[i,j] < 0:  # Alleen negatieve correlaties toevoegen
                    neg_edges.append((i, j))
                    neg_edges.append((j, i))
        else:
            # Gebruik alleen de sterke correlaties
            for j in strong_neg:
                neg_edges.append((i, j))
                neg_edges.append((j, i))
            
    # Converteer naar torch Tensors
    
    pos_edges = list(set(pos_edges))
    neg_edges = list(set(neg_edges))
    pos_edges = torch.LongTensor(list(zip(*pos_edges))) if pos_edges else torch.empty((2, 0), dtype=torch.long)
    neg_edges = torch.LongTensor(list(zip(*neg_edges))) if neg_edges else torch.empty((2, 0), dtype=torch.long)

    return pos_edges, neg_edges

def edges_to_adj_matrix(edges, num_nodes):
    """Converteer edges naar adjacency matrix"""
    adj = torch.zeros((num_nodes, num_nodes))
    if edges.size(1) > 0:
        adj[edges[0], edges[1]] = 1.0
    return adj

def calculate_label(raw_df, current_date):
    date_idx = raw_df[raw_df['Date'] == current_date].index[0]
    close_today = raw_df.iloc[date_idx]['Close']
    close_tomorrow = raw_df.iloc[date_idx+1]['Close']
    return (close_tomorrow / close_today) - 1

def prepare_dynamic_data(stock_data, window_size=20):

    for i in tqdm(range(window_size-1, len(date_to_idx)), desc="Preparing snapshots"):

        current_date = all_dates[i]

        window_dates = all_dates[i-window_size+1:i+1]
        window_data = stock_data[stock_data['Date'].isin(window_dates)]

        pos_pairs, neg_pairs = build_initial_edges_via_cosine_similarity(window_data)

        pos_adj = edges_to_adj_matrix(pos_pairs, len(unique_stocks))
        neg_adj = edges_to_adj_matrix(neg_pairs, len(unique_stocks))

        end_date = current_date
        end_idx = date_to_idx[end_date]
        start_idx = end_idx - prev_date_num + 1
        if start_idx < 0:
            print(f"Skipping {end_date} - not enough history")
            continue

        features, labels, stock_info = [], [], []
        grouped = window_data.groupby('Stock')
        stock_groups = {name: group for name, group in grouped}

        for stock_name in stock_groups.keys():
            stock_windowdata = stock_groups.get(stock_name)
            if len(stock_windowdata) == prev_date_num:
                features.append(stock_windowdata[feature_cols2].values)
                raw_df = raw_data[stock_name]
                labels.append(calculate_label(raw_df, current_date))
                stock_info.append([stock_name, end_date])
            else:
                print(f"Window data klopt niet voor {stock_name} op {end_date}")

        with open(os.path.join(data_train_predict_path, f"{end_date}.pkl"), 'wb') as f:
            pickle.dump({
                'pos_adj': pos_adj.cpu(),
                'neg_adj': neg_adj.cpu(),
                'features': torch.FloatTensor(np.array(features)).cpu(),
                'labels': torch.FloatTensor(labels).cpu(),
                'mask': [True] * len(labels)
            }, f)

        pd.DataFrame(stock_info, columns=['code', 'dt']).to_csv(
            os.path.join(daily_stock_path, f"{end_date}.csv"), index=False)


def CSI300():
    # alle paden relatief aanmaken
    global base_path, data_path, daily_data_path, raw_data_path, relation_path, snapshot_path, data_train_predict_path, daily_stock_path, log_path, stock_data, all_dates, date_to_idx, raw_data, unique_stocks
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_path, "data", "CSI300")
    daily_data_path = os.path.join(data_path, "normaliseddailydata")
    raw_data_path = os.path.join(data_path, "stockdata")
    # kies hieronder de map waarin je de resultaten wilt opslaan
    relation_path = os.path.join(data_path, "relation_onlycosine")
    os.makedirs(relation_path, exist_ok=True)
    snapshot_path= os.path.join(data_path, "intermediate_snapshots_onlycosine")
    os.makedirs(snapshot_path, exist_ok=True)
    data_train_predict_path = os.path.join(data_path, "data_train_predict_onlycosine")
    os.makedirs(data_train_predict_path, exist_ok=True)
    daily_stock_path = os.path.join(data_path, "daily_stock_onlycosine")
    os.makedirs(daily_stock_path, exist_ok=True)
    log_path = os.path.join(relation_path, f"snapshot_log.csv")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    # eenmalig inladen van alle data
    stock_data = load_all_stocks(daily_data_path)
    all_dates = sorted(stock_data['Date'].unique())
    date_to_idx = {date: idx for idx, date in enumerate(all_dates)}
    raw_data = load_raw_stocks(raw_data_path, all_dates)
    unique_stocks = sorted(stock_data['Stock'].unique())
    stock_data = stock_data.sort_values(['Stock', 'Date'])

    # start model
    prepare_dynamic_data(stock_data)

def SP500():
    # alle paden relatief aanmaken
    global base_path, data_path, daily_data_path, raw_data_path, relation_path, snapshot_path, data_train_predict_path, daily_stock_path, log_path, stock_data, all_dates, date_to_idx, raw_data, unique_stocks
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_path, "data", "S&P500")
    daily_data_path = os.path.join(data_path, "normaliseddailydata")
    raw_data_path = os.path.join(data_path, "stockdata")
    # kies hieronder de map waarin je de resultaten wilt opslaan
    relation_path = os.path.join(data_path, "relation_onlycosine")
    os.makedirs(relation_path, exist_ok=True)
    snapshot_path= os.path.join(data_path, "intermediate_snapshots_onlycosine")
    os.makedirs(snapshot_path, exist_ok=True)
    data_train_predict_path = os.path.join(data_path, "data_train_predict_onlycosine")
    os.makedirs(data_train_predict_path, exist_ok=True)
    daily_stock_path = os.path.join(data_path, "daily_stock_onlycosine")
    os.makedirs(daily_stock_path, exist_ok=True)
    log_path = os.path.join(relation_path, f"snapshot_log.csv")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    # eenmalig inladen van alle data
    stock_data = load_all_stocks(daily_data_path)
    all_dates = sorted(stock_data['Date'].unique())
    date_to_idx = {date: idx for idx, date in enumerate(all_dates)}
    raw_data = load_raw_stocks(raw_data_path, all_dates)
    unique_stocks = sorted(stock_data['Stock'].unique())
    stock_data = stock_data.sort_values(['Stock', 'Date'])

    # start model
    prepare_dynamic_data(stock_data)


def testbatch_mini():
    # alle paden relatief aanmaken
    global base_path, data_path, daily_data_path, raw_data_path, relation_path, snapshot_path, data_train_predict_path, daily_stock_path, log_path, stock_data, all_dates, date_to_idx, raw_data, unique_stocks
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_path, "data", "testbatch_mini")
    daily_data_path = os.path.join(data_path, "normaliseddailydata")
    raw_data_path = os.path.join(data_path, "stockdata")
    # kies hieronder de map waarin je de resultaten wilt opslaan
    relation_path = os.path.join(data_path, "relation_onlycosine")
    os.makedirs(relation_path, exist_ok=True)
    snapshot_path= os.path.join(data_path, "intermediate_snapshots_onlycosine")
    os.makedirs(snapshot_path, exist_ok=True)
    data_train_predict_path = os.path.join(data_path, "data_train_predict_onlycosine")
    os.makedirs(data_train_predict_path, exist_ok=True)
    daily_stock_path = os.path.join(data_path, "daily_stock_onlycosine")
    os.makedirs(daily_stock_path, exist_ok=True)
    log_path = os.path.join(relation_path, f"snapshot_log.csv")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    # eenmalig inladen van alle data
    stock_data = load_all_stocks(daily_data_path)
    all_dates = sorted(stock_data['Date'].unique())
    date_to_idx = {date: idx for idx, date in enumerate(all_dates)}
    raw_data = load_raw_stocks(raw_data_path, all_dates)
    unique_stocks = sorted(stock_data['Stock'].unique())
    stock_data = stock_data.sort_values(['Stock', 'Date'])

    # start model
    prepare_dynamic_data(stock_data)

def nasdaq5batches():
    # alle paden relatief aanmaken
    global base_path, data_path, daily_data_path, raw_data_path, relation_path, snapshot_path, data_train_predict_path, daily_stock_path, log_path, stock_data, all_dates, date_to_idx, raw_data, unique_stocks
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for i in range(5):
        data_path = os.path.join(base_path, "data", "NASDAQ_batches_5_200")
        data_path = os.path.join(data_path, f"batch_{i+1}")
        daily_data_path = os.path.join(data_path, "normaliseddailydata")
        raw_data_path = os.path.join(data_path, "stockdata")
        # kies hieronder de map waarin je de resultaten wilt opslaan
        relation_path = os.path.join(data_path, "relation_onlycosine")
        os.makedirs(relation_path, exist_ok=True)
        snapshot_path= os.path.join(data_path, "intermediate_snapshots_onlycosine")
        os.makedirs(snapshot_path, exist_ok=True)
        data_train_predict_path = os.path.join(data_path, "data_train_predict_onlycosine")
        os.makedirs(data_train_predict_path, exist_ok=True)
        daily_stock_path = os.path.join(data_path, "daily_stock_onlycosine")
        os.makedirs(daily_stock_path, exist_ok=True)
        log_path = os.path.join(relation_path, f"snapshot_log.csv")
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

        # eenmalig inladen van alle data
        stock_data = load_all_stocks(daily_data_path)
        all_dates = sorted(stock_data['Date'].unique())
        date_to_idx = {date: idx for idx, date in enumerate(all_dates)}
        raw_data = load_raw_stocks(raw_data_path, all_dates)
        unique_stocks = sorted(stock_data['Stock'].unique())
        stock_data = stock_data.sort_values(['Stock', 'Date'])

        # start model
        prepare_dynamic_data(stock_data)


# CSI300()
# SP500()
# testbatch_mini()
nasdaq5batches()