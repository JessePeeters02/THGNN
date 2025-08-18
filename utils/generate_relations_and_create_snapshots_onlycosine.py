import torch
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
import os
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Hyperparameters
prev_date_num = 20
feature_cols1 = ['Open', 'High', 'Low', 'Close']
feature_cols2 = ['Open', 'High', 'Low', 'Close', 'Volume', 'Turnover']
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
    print(all_stock_data.head())

    return all_stock_data

def load_raw_stocks(raw_stock_path, all_dates):
    raw_files = [f for f in os.listdir(raw_stock_path) if f.endswith('.csv')]
    raw_data = {}
    for file in tqdm(raw_files, desc="Loading raw data for label creation"):
        stock_name = file.split('.')[0]
        df = pd.read_csv(os.path.join(raw_stock_path, file), parse_dates=['Date'])
        df = df[df['Date'].isin(all_dates)]
        df = df.reset_index(drop=True)
        raw_data[stock_name] = df[['Date', 'Stock'] + feature_cols1]
    return raw_data

def build_initial_edges_via_cosine_similarity(window_data):
    def gpu_featurewise_cosine(stock_tensor: torch.Tensor):
        n_stocks, n_feat, n_days = stock_tensor.shape
        sims = []
        for f in range(n_feat):
            feat_f = stock_tensor[:, f, :]  # (n_stocks, n_days)
            feat_f = F.normalize(feat_f, p=2, dim=1)
            sim_f = torch.mm(feat_f, feat_f.T)  # (n_stocks, n_stocks)
            sims.append(sim_f)
        mean_sim = sum(sims) / len(sims)
        return mean_sim

    grouped = window_data.groupby('Stock')[feature_cols1]
    stock_arrays = np.array([group.values.T for _, group in grouped])  # (n_stocks, n_features, n_days)
    n_stocks = stock_arrays.shape[0]

    stock_tensor = torch.tensor(stock_arrays, dtype=torch.float32, device=device)
    cos_matrix = gpu_featurewise_cosine(stock_tensor).cpu().numpy()

    pos_edges = []
    neg_edges = []

    for i in range(n_stocks):
        strong_pos = np.where(cos_matrix[i] > sim_threshold_pos)[0]
        if len(strong_pos) < min_neighbors:
            cos_vals = cos_matrix[i].copy()
            top_pos = np.argsort(-cos_vals)[:min_neighbors]
            for j in top_pos:
                if cos_matrix[i,j] > 0:
                    pos_edges.append((i, j))
                    pos_edges.append((j, i))
        else:
            for j in strong_pos:
                pos_edges.append((i, j))
                pos_edges.append((j, i))
        
        strong_neg = np.where(cos_matrix[i] < sim_threshold_neg)[0]
        if len(strong_neg) < min_neighbors:
            cos_vals = cos_matrix[i].copy()
            top_neg = np.argsort(cos_vals)[:min_neighbors]
            for j in top_neg:
                if cos_matrix[i,j] < 0:
                    neg_edges.append((i, j))
                    neg_edges.append((j, i))
        else:
            for j in strong_neg:
                neg_edges.append((i, j))
                neg_edges.append((j, i))
                
    pos_edges = list(set(pos_edges))
    neg_edges = list(set(neg_edges))
    pos_edges = torch.LongTensor(list(zip(*pos_edges))) if pos_edges else torch.empty((2, 0), dtype=torch.long)
    neg_edges = torch.LongTensor(list(zip(*neg_edges))) if neg_edges else torch.empty((2, 0), dtype=torch.long)

    return pos_edges, neg_edges

def edges_to_adj_matrix(edges, num_nodes):
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

    for i in tqdm(range(window_size-1, len(date_to_idx)-1), desc="Preparing snapshots"):

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


if __name__ == "__main__":
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # data_path = os.path.join(base_path, "data", "CSI300")
    data_path = os.path.join(base_path, "data", "S&P500")
    daily_data_path = os.path.join(data_path, "normaliseddailydata")
    raw_data_path = os.path.join(data_path, "stockdata")
    relation_path = os.path.join(data_path, "relation_CS")
    os.makedirs(relation_path, exist_ok=True)
    snapshot_path= os.path.join(data_path, "intermediate_snapshots_CS")
    os.makedirs(snapshot_path, exist_ok=True)
    data_train_predict_path = os.path.join(data_path, "data_train_predict_CS")
    os.makedirs(data_train_predict_path, exist_ok=True)
    daily_stock_path = os.path.join(data_path, "daily_stock_CS")
    os.makedirs(daily_stock_path, exist_ok=True)
    log_path = os.path.join(relation_path, f"snapshot_log.csv")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    stock_data = load_all_stocks(daily_data_path)
    all_dates = sorted(stock_data['Date'].unique())
    date_to_idx = {date: idx for idx, date in enumerate(all_dates)}
    raw_data = load_raw_stocks(raw_data_path, all_dates)
    unique_stocks = sorted(stock_data['Stock'].unique())
    stock_data = stock_data.sort_values(['Stock', 'Date'])

    prepare_dynamic_data(stock_data)