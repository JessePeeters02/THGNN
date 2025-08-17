import os
import pickle
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

# ====== Hyperparameters (gelijk aan jouw DynamiSE) ======
prev_date_num = 20
feature_cols1 = ['Open', 'High', 'Low', 'Close']
feature_cols2 = ['Open', 'High', 'Low', 'Close', 'Volume', 'Turnover']

# Cosine-similarity → signed edges
min_neighbors = 3
sim_threshold_pos = 0.4
sim_threshold_neg = -0.4
margin = 2

# ====== Helpers ======

def load_all_stocks(normalised_dir, restrict_last_n_days=None):
    """Laad alle genormaliseerde dagelijkse data (zelfde als DynamiSE)."""
    all_stock_data = []
    files = [f for f in os.listdir(normalised_dir) if f.endswith('.csv')]
    # for file in tqdm(files, desc="Loading normalised data"):
    for file in files:
        df = pd.read_csv(os.path.join(normalised_dir, file))
        # Verwacht kolommen: Date, Stock, + feature_cols2
        all_stock_data.append(df[['Date', 'Stock'] + feature_cols2])
    if not all_stock_data:
        raise FileNotFoundError(f"Geen .csv bestanden gevonden in {normalised_dir}")
    all_stock_data = pd.concat(all_stock_data, ignore_index=True)

    if restrict_last_n_days is not None:
        all_dates = sorted(all_stock_data['Date'].unique())
        last_dates = all_dates[-restrict_last_n_days:]
        all_stock_data = all_stock_data[all_stock_data['Date'].isin(last_dates)]

    # Zorg voor vaste sortering
    all_stock_data = all_stock_data.sort_values(['Stock', 'Date']).reset_index(drop=True)
    return all_stock_data


def load_raw_stocks(raw_dir, all_dates):
    """Laad ruwe data per stock om labels te berekenen (zelfde structuur als DynamiSE)."""
    raw_files = [f for f in os.listdir(raw_dir) if f.endswith('.csv')]
    if not raw_files:
        raise FileNotFoundError(f"Geen .csv bestanden gevonden in {raw_dir}")
    raw_data = {}
    # for file in tqdm(raw_files, desc="Loading raw data for labels"):
    for file in raw_files:
        stock_name = file.split('.')[0]
        df = pd.read_csv(os.path.join(raw_dir, file), parse_dates=['Date'])
        # Filter op dezelfde datums
        df = df[df['Date'].astype(str).isin(all_dates)].reset_index(drop=True)
        raw_data[stock_name] = df[['Date', 'Stock'] + feature_cols1]
    return raw_data


@torch.no_grad()
def gpu_featurewise_cosine(stock_tensor: torch.Tensor):
    """
    stock_tensor: (n_stocks, n_features, n_days)
    Retourneert mean cosine similarity over features: (n_stocks, n_stocks)
    Identiek principe als in DynamiSE (gemiddelde over feature-dimensie).
    """
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
    """
    Volgt de logica uit jouw DynamiSE:
    - groepeer per stock
    - per-feature cosine similarity
    - minimum neighbors boosten
    - drempels voor pos/neg
    """
    grouped = window_data.groupby('Stock')[feature_cols1]
    stock_arrays = np.array([group.values.T for _, group in grouped])  # (N, F1, D)
    n_stocks = stock_arrays.shape[0]

    stock_tensor = torch.tensor(stock_arrays, dtype=torch.float32, device=device)
    cos_matrix = gpu_featurewise_cosine(stock_tensor).cpu().numpy()  # (N, N)

    pos_edges = []
    neg_edges = []

    for i in range(n_stocks):
        # Positieve edges
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

        # Negatieve edges
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

    # Uniek maken
    pos_edges = list(set(pos_edges))
    neg_edges = list(set(neg_edges))

    pos_edges_t = torch.LongTensor(list(zip(*pos_edges))) if pos_edges else torch.empty((2, 0), dtype=torch.long)
    neg_edges_t = torch.LongTensor(list(zip(*neg_edges))) if neg_edges else torch.empty((2, 0), dtype=torch.long)
    return pos_edges_t, neg_edges_t


def sign_semantics_aggregation(num_nodes, edge_list_pos, edge_list_neg, device):
    """
    Identiek aan jouw DynamiSE-SSA (balance theory triad closure).
    Geeft delta_A_pos en delta_A_neg terug, symmetrisch gemaakt.
    """
    # Directe edges
    A_pos = torch.zeros((num_nodes, num_nodes), dtype=torch.float32, device=device)
    A_neg = torch.zeros((num_nodes, num_nodes), dtype=torch.float32, device=device)
    if edge_list_pos.numel() > 0:
        A_pos[edge_list_pos[0], edge_list_pos[1]] = 1.0
    if edge_list_neg.numel() > 0:
        A_neg[edge_list_neg[0], edge_list_neg[1]] = 1.0
    A_pos.fill_diagonal_(0.0)
    A_neg.fill_diagonal_(0.0)

    # Triad closure via matmul
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

    # Symmetriseren
    delta_A_pos = torch.maximum(delta_A_pos, delta_A_pos.t())
    delta_A_neg = torch.maximum(delta_A_neg, delta_A_neg.t())
    delta_A_pos.fill_diagonal_(0.0)
    delta_A_neg.fill_diagonal_(0.0)

    overlap = (delta_A_pos > 0) & (delta_A_neg > 0)
    if overlap.any():
        print(f"Waarschuwing: Er zijn {overlap.sum().item()} overlappingen tussen positieve en negatieve edges!")
        print("Overlapping indices:", torch.nonzero(overlap, as_tuple=True))

    return delta_A_pos, delta_A_neg


def calculate_label(raw_df, current_date):
    """
    Zelfde labeldefinitie als DynamiSE: return van t->t+1 (Close).
    """
    # raw_df['Date'] is datetime64; current_date komt als string 'YYYY-MM-DD'
    idx = raw_df[raw_df['Date'].astype(str) == current_date].index
    if len(idx) == 0 or idx[0] + 1 >= len(raw_df):
        # Geen label beschikbaar (laatste dag of ontbrekende dag)
        return None
    i = idx[0]
    close_today = raw_df.iloc[i]['Close']
    close_yesterday = raw_df.iloc[i-1]['Close']
    return float(close_today / close_yesterday - 1.0)


def edges_to_adj_matrix(edges, num_nodes, device):
    adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float32, device=device)
    if edges.numel() > 0:
        adj[edges[0], edges[1]] = 1.0
    return adj


# ====== Hoofd-pipeline ======

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Device: {device}")

    dataset_dir = os.path.abspath(args.dataset_dir)
    normalised_dir = os.path.join(dataset_dir, "normaliseddailydata")
    raw_dir = os.path.join(dataset_dir, "stockdata")

    # Output directories (STATIC, om verwarring te vermijden)
    relation_dir = os.path.join(dataset_dir, "relation_STATIC_t1")                 # log e.d.
    snapshot_dir = os.path.join(dataset_dir, "intermediate_snapshots_STATIC_t1")   # optioneel, hier niet gebruikt
    out_pkl_dir = os.path.join(dataset_dir, "data_train_predict_STATIC_t1")
    out_daily_dir = os.path.join(dataset_dir, "daily_stock_STATIC_t1")

    os.makedirs(relation_dir, exist_ok=True)
    os.makedirs(snapshot_dir, exist_ok=True)
    os.makedirs(out_pkl_dir, exist_ok=True)
    os.makedirs(out_daily_dir, exist_ok=True)

    # Logging (zelfde stijl als DynamiSE)
    log_path = os.path.join(relation_dir, "snapshot_log.csv")
    eval_log_path = os.path.join(relation_dir, "edge_evaluation_log.csv")

    # 1) Data inladen
    stock_data = load_all_stocks(normalised_dir)
    all_dates = sorted(stock_data['Date'].unique())
    unique_stocks = sorted(stock_data['Stock'].unique())
    n_stocks = len(unique_stocks)
    print(f"[INFO] {len(all_dates)} datums, {n_stocks} stocks gevonden.")

    # Snel index voor window slices
    date_to_idx = {d: i for i, d in enumerate(all_dates)}

    # Ruwe data voor labels
    raw_data = load_raw_stocks(raw_dir, all_dates)

    # Voor logging
    write_snapshot_header = not os.path.exists(log_path) or os.path.getsize(log_path) == 0
    if write_snapshot_header:
        with open(log_path, "w") as f:
            f.write("date,nodes,pos_edges_cos,neg_edges_cos,pos_edges_ssa,neg_edges_ssa\n")

    # 2) Itereer over alle mogelijke windows
    for i in range(prev_date_num - 1, len(all_dates)):
    # for i in tqdm(range(prev_date_num - 1, len(all_dates)), desc="Building static snapshots"):
        end_date = all_dates[i]
        start_i = i - prev_date_num + 1
        if start_i < 0:
            continue

        window_dates = all_dates[start_i:i+1]
        # Filter data voor window en huidige dag
        window_data = stock_data[stock_data['Date'].isin(window_dates)]
        today_data = stock_data[stock_data['Date'] == end_date]

        # Zet feature-matrix (N, prev_date_num, len(feature_cols2)) in exact dezelfde volgorde
        grouped_today = today_data.groupby("Stock")
        # Check dat alle stocks aanwezig zijn
        missing = set(unique_stocks) - set(grouped_today.groups.keys())
        if missing:
            # Als er stocks missen op bepaalde dagen, skip of vul mask=0
            # Voor eenvoud: we vereisen complete panel voor statische benchmark
            print(f"[WARN] {end_date}: {len(missing)} stocks missen vandaag, snapshot wordt overgeslagen.")
            continue

        # Cosine edges op window_data (gebruik alleen feature_cols1 zoals DynamiSE)
        pos_pairs, neg_pairs = build_initial_edges_via_cosine_similarity(window_data, device=device)

        # Balance theory (SSA) voor triad closure (identiek aan DynamiSE)
        pos_pairs_t = pos_pairs.to(device)
        neg_pairs_t = neg_pairs.to(device)
        delta_A_pos, delta_A_neg = sign_semantics_aggregation(n_stocks, pos_pairs_t, neg_pairs_t, device=device)

        # Tellen voor logging
        pos_cos_count = pos_pairs.shape[1]
        neg_cos_count = neg_pairs.shape[1]
        pos_ssa = torch.nonzero(delta_A_pos).t().cpu()
        neg_ssa = torch.nonzero(delta_A_neg).t().cpu()
        pos_ssa_count = pos_ssa.shape[1]
        neg_ssa_count = neg_ssa.shape[1]

        with open(log_path, "a") as f:
            f.write(f"{end_date},{n_stocks},{pos_cos_count},{neg_cos_count},{pos_ssa_count},{neg_ssa_count}\n")

        # 3) Features & labels opbouwen
        # Maak mapping stock -> volledige window (prev_date_num rijen)
        window_grouped = window_data.groupby('Stock')
        features_list = []
        labels_list = []
        mask_list = []

        for stock_name in unique_stocks:
            grp = window_grouped.get_group(stock_name)
            if len(grp) != prev_date_num:
                # onvolledig venster
                features_list.append(np.zeros((prev_date_num, len(feature_cols2)), dtype=np.float32))
                labels_list.append(0.0)
                mask_list.append(False)
                continue

            # features: exact zoals DynamiSE → (prev_date_num, len(feature_cols2))
            feat = grp[feature_cols2].values.astype(np.float32)
            features_list.append(feat)

            # label op end_date
            raw_df = raw_data.get(stock_name)
            if raw_df is None:
                labels_list.append(0.0)
                mask_list.append(False)
                continue
            lbl = calculate_label(raw_df, end_date)
            if lbl is None:
                labels_list.append(0.0)
                mask_list.append(False)
            else:
                labels_list.append(float(lbl))
                mask_list.append(True)

        features_np = np.stack(features_list, axis=0)  # (N, prev_date_num, F2)
        labels_np = np.array(labels_list, dtype=np.float32)  # (N,)
        mask_np = np.array(mask_list, dtype=bool)  # (N,)

        # 4) Adjacency-matrices (dense) naar CPU voor opslag
        pos_adj = delta_A_pos.detach().cpu()
        neg_adj = delta_A_neg.detach().cpu()

        sample = {
            'pos_adj': pos_adj,                                                # Tensor [N, N]
            'neg_adj': neg_adj,                                                # Tensor [N, N]
            'features': torch.from_numpy(features_np),                         # Tensor [N, T, F]
            'labels': torch.from_numpy(labels_np),                             # Tensor [N]
            'mask': mask_np                                                    # list/bool array [N]
        }

        # 5) Wegschrijven in STATIC outputmappen
        out_pkl_path = os.path.join(out_pkl_dir, f"{end_date}.pkl")
        with open(out_pkl_path, "wb") as f:
            pickle.dump(sample, f)

        # Mapping code/dt voor dezelfde datum (zoals DynamiSE doet)
        day_rows = [[code, end_date] for code in unique_stocks]
        pd.DataFrame(day_rows, columns=['code', 'dt']).to_csv(
            os.path.join(out_daily_dir, f"{end_date}.csv"), index=False
        )

    print(f"[DONE] Static snapshots geschreven naar:\n  - {out_pkl_dir}\n  - {out_daily_dir}")
    print("[TIP] Zet in jouw main_remake.py de base_dir naar deze STATIC map om THGNN hierop te trainen.")


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