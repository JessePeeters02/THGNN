import os
import pickle
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

# Hyperparameters
prev_date_num = 20
feature_cols1 = ['Open', 'High', 'Low', 'Close']
feature_cols2 = ['Open', 'High', 'Low', 'Close', 'Volume', 'Turnover']

min_neighbors = 3
sim_threshold_pos = 0.4
sim_threshold_neg = -0.4
margin = 2


def load_all_stocks(normalised_dir, restrict_last_n_days=None):
    """Laad alle genormaliseerde dagelijkse data (zelfde als DynamiSE)."""
    all_stock_data = []
    files = [f for f in os.listdir(normalised_dir) if f.endswith('.csv')]
    for file in tqdm(files, desc="Loading normalised data"):
        df = pd.read_csv(os.path.join(normalised_dir, file))
        all_stock_data.append(df[['Date', 'Stock'] + feature_cols2])
    all_stock_data = pd.concat(all_stock_data, ignore_index=True)
    return all_stock_data


def load_raw_stocks(raw_dir, all_dates):
    raw_files = [f for f in os.listdir(raw_dir) if f.endswith('.csv')]
    raw_data = {}
    for file in tqdm(raw_files, desc="Loading raw data for labels"):
        stock_name = file.split('.')[0]
        df = pd.read_csv(os.path.join(raw_dir, file), parse_dates=['Date'])
        df = df[df['Date'].astype(str).isin(all_dates)].reset_index(drop=True)
        raw_data[stock_name] = df[['Date', 'Stock'] + feature_cols1]
    return raw_data


def gpu_featurewise_cosine(stock_tensor: torch.Tensor):
    n_stocks, n_feat, _ = stock_tensor.shape
    sims = []
    for f in range(n_feat):
        feat_f = stock_tensor[:, f, :]           # (N, D)
        feat_f = F.normalize(feat_f, p=2, dim=1) # row-wise
        sim_f = feat_f @ feat_f.t()              # (N, N)
        sims.append(sim_f)
    mean_sim = sum(sims) / len(sims)
    return mean_sim


def build_initial_edges_via_cosine_similarity(window_data, device):
    grouped = window_data.groupby('Stock')[feature_cols1]
    stock_arrays = np.array([group.values.T for _, group in grouped])  # (N, F1, D)
    n_stocks = stock_arrays.shape[0]

    stock_tensor = torch.tensor(stock_arrays, dtype=torch.float32, device=device)
    cos_matrix = gpu_featurewise_cosine(stock_tensor).cpu().numpy()  # (N, N)

    pos_edges = []
    neg_edges = []

    for i in range(n_stocks):
        strong_pos = np.where(cos_matrix[i] > sim_threshold_pos)[0]
        if len(strong_pos) < min_neighbors:
            cos_vals = cos_matrix[i].copy()
            top_pos = np.argsort(-cos_vals)[:min_neighbors]
            for j in top_pos:
                if i != j and cos_matrix[i, j] > 0:
                    pos_edges.append((i, j))
                    pos_edges.append((j, i))
        else:
            for j in strong_pos:
                if i != j:
                    pos_edges.append((i, j))
                    pos_edges.append((j, i))

        strong_neg = np.where(cos_matrix[i] < sim_threshold_neg)[0]
        if len(strong_neg) < min_neighbors:
            cos_vals = cos_matrix[i].copy()
            top_neg = np.argsort(cos_vals)[:min_neighbors]
            for j in top_neg:
                if i != j and cos_matrix[i, j] < 0:
                    neg_edges.append((i, j))
                    neg_edges.append((j, i))
        else:
            for j in strong_neg:
                if i != j:
                    neg_edges.append((i, j))
                    neg_edges.append((j, i))

    pos_edges = list(set(pos_edges))
    neg_edges = list(set(neg_edges))
    pos_edges_t = torch.LongTensor(list(zip(*pos_edges))) if pos_edges else torch.empty((2, 0), dtype=torch.long)
    neg_edges_t = torch.LongTensor(list(zip(*neg_edges))) if neg_edges else torch.empty((2, 0), dtype=torch.long)
    return pos_edges_t, neg_edges_t

def build_initial_edges_via_correlation(window_data, device):
    grouped = window_data.groupby('Stock')[feature_cols1]
    stock_arrays = np.array([group.values.T for name, group in grouped])  # (n_stocks, n_features, n_days)
    n_stocks = stock_arrays.shape[0]

    stock_tensor = torch.tensor(stock_arrays, dtype=torch.float32, device=device)  # (n_stocks, n_features, n_days)

    stock_tensor = stock_tensor - stock_tensor.mean(dim=2, keepdim=True)
    stock_tensor = stock_tensor / (stock_tensor.std(dim=2, keepdim=True) + 1e-8)

    corr_matrices = []
    for f in range(stock_tensor.shape[1]):
        X = stock_tensor[:, f, :]  # (n_stocks, n_days)
        corr = torch.matmul(X, X.T) / (X.shape[1] - 1)
        corr_matrices.append(corr)
    corr_stack = torch.stack(corr_matrices, dim=2)  # (n_stocks, n_stocks, n_features)
    corr_matrix = corr_stack.mean(dim=2)  # (n_stocks, n_stocks)

    corr_matrix.fill_diagonal_(0)
    corr_matrix = corr_matrix.cpu().numpy()

    pos_edges = []
    neg_edges = []

    for i in range(n_stocks):
        strong_pos = np.where(corr_matrix[i] > sim_threshold_pos)[0]
        if len(strong_pos) < min_neighbors:
            corrs = corr_matrix[i].copy()
            top_pos = np.argsort(-corrs)[:min_neighbors]
            for j in top_pos:
                if corr_matrix[i,j] > 0:
                    pos_edges.append((i, j))
                    pos_edges.append((j, i))
        else:
            for j in strong_pos:
                pos_edges.append((i, j))
                pos_edges.append((j, i))
        
        strong_neg = np.where(corr_matrix[i] < sim_threshold_neg)[0]
        if len(strong_neg) < min_neighbors:
            corrs = corr_matrix[i].copy()
            top_neg = np.argsort(corrs)[:min_neighbors]
            for j in top_neg:
                if corr_matrix[i,j] < 0:
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


def sign_semantics_aggregation(num_nodes, edge_list_pos, edge_list_neg, device):
    A_pos = torch.zeros((num_nodes, num_nodes), dtype=torch.float32, device=device)
    A_neg = torch.zeros((num_nodes, num_nodes), dtype=torch.float32, device=device)
    if edge_list_pos.numel() > 0:
        A_pos[edge_list_pos[0], edge_list_pos[1]] = 1.0
    if edge_list_neg.numel() > 0:
        A_neg[edge_list_neg[0], edge_list_neg[1]] = 1.0
    A_pos.fill_diagonal_(0.0)
    A_neg.fill_diagonal_(0.0)

    P1 = A_pos @ A_pos
    P2 = A_neg @ A_neg
    P3 = A_pos @ A_neg
    P4 = A_neg @ A_pos

    existing = (A_pos + A_neg) > 0
    suggested_pos = P1 + P2
    suggested_neg = P3 + P4
    delta = suggested_pos - suggested_neg
    cand_pos = (delta > margin) & (~existing)
    cand_neg = (delta < -margin) & (~existing)

    delta_A_pos = A_pos.clone()
    delta_A_neg = A_neg.clone()

    delta_A_pos[cand_pos] = 1.0
    delta_A_neg[cand_neg] = 1.0

    delta_A_pos = torch.maximum(delta_A_pos, delta_A_pos.t())
    delta_A_neg = torch.maximum(delta_A_neg, delta_A_neg.t())
    delta_A_pos.fill_diagonal_(0.0)
    delta_A_neg.fill_diagonal_(0.0)

    return delta_A_pos, delta_A_neg


def calculate_label(raw_df, current_date):
    date_idx = raw_df[raw_df['Date'] == current_date].index[0]
    close_today = raw_df.iloc[date_idx]['Close']
    close_tomorrow = raw_df.iloc[date_idx+1]['Close']
    return (close_tomorrow / close_today) - 1


def edges_to_adj_matrix(edges, num_nodes, device):
    adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float32, device=device)
    if edges.numel() > 0:
        adj[edges[0], edges[1]] = 1.0
    return adj

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Device: {device}")

    dataset_dir = os.path.abspath(args.dataset_dir)
    normalised_dir = os.path.join(dataset_dir, "normaliseddailydata")
    raw_dir = os.path.join(dataset_dir, "stockdata")

    relation_dir = os.path.join(dataset_dir, "relation_SSA-CS")
    out_pkl_dir = os.path.join(dataset_dir, "data_train_predict_SSA-CS")
    out_daily_dir = os.path.join(dataset_dir, "daily_stock_SSA-CS")

    os.makedirs(relation_dir, exist_ok=True)
    os.makedirs(out_pkl_dir, exist_ok=True)
    os.makedirs(out_daily_dir, exist_ok=True)

    log_path = os.path.join(relation_dir, "snapshot_log.csv")
    eval_log_path = os.path.join(relation_dir, "edge_evaluation_log.csv")

    stock_data = load_all_stocks(normalised_dir)
    all_dates = sorted(stock_data['Date'].unique())
    unique_stocks = sorted(stock_data['Stock'].unique())
    n_stocks = len(unique_stocks)
    print(f"[INFO] {len(all_dates)} datums, {n_stocks} stocks gevonden.")

    raw_data = load_raw_stocks(raw_dir, all_dates)

    write_snapshot_header = not os.path.exists(log_path) or os.path.getsize(log_path) == 0
    if write_snapshot_header:
        with open(log_path, "w") as f:
            f.write("date,nodes,pos_edges_cos,neg_edges_cos,pos_edges_ssa,neg_edges_ssa\n")

    for i in tqdm(range(prev_date_num - 1, len(all_dates)), desc="Building static snapshots"):
        end_date = all_dates[i]
        start_i = i - prev_date_num + 1
        if start_i < 0:
            continue

        window_dates = all_dates[start_i:i+1]
        window_data = stock_data[stock_data['Date'].isin(window_dates)]

        pos_pairs, neg_pairs = build_initial_edges_via_cosine_similarity(window_data, device=device)
        # pos_pairs, neg_pairs = build_initial_edges_via_correlation(window_data, device=device)

        pos_pairs_t = pos_pairs.to(device)
        neg_pairs_t = neg_pairs.to(device)
        delta_A_pos, delta_A_neg = sign_semantics_aggregation(n_stocks, pos_pairs_t, neg_pairs_t, device=device)

        pos_cos_count = pos_pairs.shape[1]
        neg_cos_count = neg_pairs.shape[1]
        pos_ssa = torch.nonzero(delta_A_pos).t().cpu()
        neg_ssa = torch.nonzero(delta_A_neg).t().cpu()
        pos_ssa_count = pos_ssa.shape[1]
        neg_ssa_count = neg_ssa.shape[1]

        with open(log_path, "a") as f:
            f.write(f"{end_date},{n_stocks},{pos_cos_count},{neg_cos_count},{pos_ssa_count},{neg_ssa_count}\n")

        window_grouped = window_data.groupby('Stock')
        features_list = []
        labels_list = []
        mask_list = []

        for stock_name in unique_stocks:
            grp = window_grouped.get_group(stock_name)
            feat = grp[feature_cols2].values.astype(np.float32)
            features_list.append(feat)
            raw_df = raw_data.get(stock_name)
            lbl = calculate_label(raw_df, end_date)
            labels_list.append(float(lbl))
            mask_list.append(True)

        features_np = np.stack(features_list, axis=0)  # (N, prev_date_num, F2)
        labels_np = np.array(labels_list, dtype=np.float32)  # (N,)
        mask_np = np.array(mask_list, dtype=bool)  # (N,)

        pos_adj = delta_A_pos.detach().cpu()
        neg_adj = delta_A_neg.detach().cpu()

        sample = {
            'pos_adj': pos_adj,                                                # Tensor [N, N]
            'neg_adj': neg_adj,                                                # Tensor [N, N]
            'features': torch.from_numpy(features_np),                         # Tensor [N, T, F]
            'labels': torch.from_numpy(labels_np),                             # Tensor [N]
            'mask': mask_np                                                    # list/bool array [N]
        }

        out_pkl_path = os.path.join(out_pkl_dir, f"{end_date}.pkl")
        with open(out_pkl_path, "wb") as f:
            pickle.dump(sample, f)

        day_rows = [[code, end_date] for code in unique_stocks]
        pd.DataFrame(day_rows, columns=['code', 'dt']).to_csv(
            os.path.join(out_daily_dir, f"{end_date}.csv"), index=False
        )

    print(f"[DONE] Static snapshots geschreven naar:\n  - {out_pkl_dir}\n  - {out_daily_dir}")

if __name__ == "__main__":
    class Args:
        pass
    args = Args()
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_path, "data", "CSI300")
    args.dataset_dir = data_path

    main(args)

    class Args:
        pass
    args = Args()
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_path, "data", "S&P500")
    args.dataset_dir = data_path

    main(args)