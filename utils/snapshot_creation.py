import torch
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
import os
import itertools
from collections import defaultdict
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity as cosine_similarity_skl


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu') 
print(f"Device: {device}")

# alle paden relatief aanmaken
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(base_path, "data", "testbatch_mini")
daily_data_path = os.path.join(data_path, "normaliseddailydata")
raw_data_path = os.path.join(data_path, "stockdata")
# kies hieronder de map waarin je de resultaten wilt opslaan
snapshot_path= os.path.join(data_path, "intermediate_snapshots_mini")
os.makedirs(snapshot_path, exist_ok=True)

# Hyperparameters
prev_date_num = 20
feature_cols1 = ['Open', 'High', 'Low', 'Close']
feature_cols2 = ['Open', 'High', 'Low', 'Close', 'Volume', 'Turnover']
restrict_last_n_days= 100 # None of bv 80 om da laatse 60 dagen te nemen (20-day time window geraak je in begin altijd kwijt)
relevance_threshold = 0
max_age = 5
threshold = 0.4
min_neighbors = 3

sim_threshold_pos = 0.6
sim_threshold_neg = -0.4

class EdgeAgingManager:
    def __init__(self, relevance_threshold, max_age, date_to_idx):
        self.relevance_threshold = relevance_threshold
        self.max_age = max_age
        self.date_to_idx = date_to_idx

    def update_edge(self, edge_info, src, dst, new_score, current_date):
        key = (src, dst)
        if key in edge_info:
            edge_info[key]['last_score'] = new_score
            edge_info[key]['last_update'] = current_date
        else:
            edge_info[key] = {
                'created_at': current_date,
                'last_score': new_score,
                'last_update': current_date
            }

    def prune_edges(self, edge_info, current_date, price_data):
        to_delete = []
        for (src, dst), info in edge_info.items():
            age = self._calculate_age(info['last_update'], current_date)
            if (age >= self.max_age):
                vec_i = price_data[src]
                vec_k = price_data[dst]
                sim = cosine_similarity(vec_i, vec_k)

                # Get the edge score to determine if it's positive or negative
                edge_score = info['last_score']
                if edge_score > 0:  # Positive edge
                    if sim > sim_threshold_pos:
                        # Reset age by updating created_at to current date
                        edge_info[(src,dst)]['last_update'] = current_date
                    else:
                        to_delete.append((src,dst))
                else:  # Negative edge
                    if sim < sim_threshold_neg:
                        # Reset age by updating created_at to current date
                        edge_info[(src,dst)]['last_update'] = current_date
                    else:
                        to_delete.append((src,dst))

        # Delete edges that didn't meet the criteria
        for edge in to_delete:
            del edge_info[edge]

    def _calculate_age(self, last_update, current_date):
        idx_created = self.date_to_idx[last_update]
        idx_current = self.date_to_idx[current_date]
        return idx_current - idx_created

def edge_info_to_tensor(edge_info):
    if not edge_info:
        return torch.empty((2, 0), dtype=torch.long).to(device)
    edges = list(edge_info.keys())
    return torch.LongTensor(list(zip(*edges))).to(device)

def warn_nodes_without_neighbors(adj_pos, adj_neg, num_nodes, snapshot_date=None):
    all_nodes = set(range(num_nodes))

    nodes_with_pos = set(adj_pos.keys())
    nodes_with_neg = set(adj_neg.keys())

    # nodes_without_pos = all_nodes - nodes_with_pos
    # nodes_without_neg = all_nodes - nodes_with_neg

    # date_str = snapshot_date if snapshot_date else "unknown"

    # if nodes_without_pos:
    #     print(f"[WARN] Snapshot {date_str}: {len(nodes_without_pos)} nodes hebben GEEN positieve buren (bv. {list(nodes_without_pos)[:5]})")
    # if nodes_without_neg:
    #     print(f"[WARN] Snapshot {date_str}: {len(nodes_without_neg)} nodes hebben GEEN negatieve buren (bv. {list(nodes_without_neg)[:5]})")
    isolated_nodes = all_nodes - (nodes_with_pos | nodes_with_neg)
    # if isolated_nodes:
    #     print(f"[WARN] Snapshot {date_str}: {len(isolated_nodes)} nodes hebben GEEN buren (positief noch negatief)")

    # return (nodes_without_neg|nodes_without_pos)
    return isolated_nodes


def cosine_similarity(vec1, vec2):
    cos = []
    for i in range(vec1.shape[0]):
        sim = F.cosine_similarity(vec1[i].unsqueeze(0), vec2[i].unsqueeze(0)).item()
        cos.append(sim)
    if len(cos) != len(feature_cols1):
        print("[Warn] lengte van cos is verschillend van feature lengte")
    return np.mean(cos)

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

def build_initial_edges_via_cosine_similarity(window_data, threshold):
    grouped = window_data.groupby('Stock')[feature_cols1]
    stock_arrays = np.array([group.values.T for name, group in grouped])
    n_stocks = stock_arrays.shape[0]
    cos_matrix = np.zeros((n_stocks, n_stocks))

    # Bereken cosine similarities tussen alle paren van stocks
    for i in tqdm(range(n_stocks), desc="Calculating cosine similarities"):
        for j in range(i+1, n_stocks):
            feature_cosines = []
            for f in range(len(feature_cols1)):
                cos_sim = cosine_similarity_skl([stock_arrays[i,f,:]], [stock_arrays[j,f,:]])[0][0]
                feature_cosines.append(cos_sim)
            avg_cos = np.nanmean(feature_cosines)
            cos_matrix[i,j] = avg_cos
            cos_matrix[j,i] = avg_cos

    # Bouw edges op basis van drempelwaarde
    pos_edges = []
    neg_edges = []

    # Garandeer minimum aantal buren
    for i in range(n_stocks):
        # Positieve edges
        strong_pos = np.where(cos_matrix[i] > threshold)[0]
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
        strong_neg = np.where(cos_matrix[i] < -threshold)[0]
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

def build_edges_via_balance_theory(prev_pos_edges, prev_neg_edges, num_nodes, price_data, current_date):#, close_data_raw):
    adj_pos = defaultdict(set)
    adj_neg = defaultdict(set)

    # Bouw adjacency lists
    if prev_pos_edges.numel() > 0:
        for src, dst in prev_pos_edges.T.tolist():
            adj_pos[src].add(dst)
    if prev_neg_edges.numel() > 0:
        for src, dst in prev_neg_edges.T.tolist():
            adj_neg[src].add(dst)

    # start_isolated_nodes = time.time()
    isolated_nodes = warn_nodes_without_neighbors(adj_pos, adj_neg, num_nodes, snapshot_date=current_date)
    # end_isolated_nodes = time.time()
    # print(f" isolated nodes duurde {end_isolated_nodes-start_isolated_nodes:.4f} seconden")
    pos_edges_set = set()
    neg_edges_set = set()
    triangle_count = 0
    confirmed_edges = set()
    attempted_edges = defaultdict(set)

    # start_triangles = time.time()
    # for j in tqdm(range(num_nodes), desc="triangles"):
    for j in range(num_nodes):
        neighbors = adj_pos[j] | adj_neg[j]
        for i, k in itertools.combinations(neighbors, 2):
            if i == k:
                continue

            edge_key = frozenset((i, k))
            if edge_key in confirmed_edges:
                continue

            # Check bestaande edges
            if (k in adj_pos[i]) or (k in adj_neg[i]):
                continue

            # Debug: Tel triadische checks
            triangle_count += 1

            # Triadische check  
            signs = []
            for u, v in [(i, j), (j, k)]:
                if v in adj_pos[u]:
                    signs.append('+')
                elif v in adj_neg[u]:
                    signs.append('-')
                else:
                    break
            else:
                neg_count = signs.count('-')
                signature = 'positive' if neg_count % 2 == 0 else 'negative'

                if signature in attempted_edges[edge_key]:
                    continue  # deze context al geprobeerd
                attempted_edges[edge_key].add(signature)

                vec_i = price_data[i]
                vec_k = price_data[k]
                sim = cosine_similarity(vec_i, vec_k)

                # Balance theory toepassen
                if signature == 'positive' and sim > sim_threshold_pos:
                    pos_edges_set.add((i, k))
                    pos_edges_set.add((k, i))
                    confirmed_edges.add(edge_key)
                elif signature == 'negative' and sim < sim_threshold_neg:
                    neg_edges_set.add((i, k))
                    neg_edges_set.add((k, i))
                    confirmed_edges.add(edge_key)

    # end_triangles = time.time()
    # print(f" triangles duurde {end_triangles-start_triangles:.4f} seconden")

    # start_isolated_new = time.time()
    if isolated_nodes:
        print(f"Adding fallback connections for {len(isolated_nodes)} isolated nodes")
        edges_added = 0
        for i in isolated_nodes:
            vec_i = price_data[i]
            simmilarities = []
            for j in range(num_nodes):
                if j == i:
                    continue
                vec_j = price_data[j]
                sim = cosine_similarity(vec_i, vec_j)
                simmilarities.append(sim)
                if sim > sim_threshold_pos:
                    pos_edges_set.add((i, j))
                    pos_edges_set.add((j, i))
                    edges_added += 1
                elif sim < sim_threshold_neg:
                    neg_edges_set.add((i, j))
                    neg_edges_set.add((j, i))
                    edges_added += 1
            print(f"    For node {i}, {edges_added} edges where added.")
    #         print(f"     sim min: {min(simmilarities)}, sim max: {max(simmilarities)}")
    # end_isolated_new = time.time()
    # print(f" fallback duurde {end_isolated_new-start_isolated_new:.4f} seconden")

    # Converteer naar tensors
    pos_edges = torch.LongTensor(list(zip(*pos_edges_set))) if pos_edges_set else torch.empty((2, 0), dtype=torch.long)
    neg_edges = torch.LongTensor(list(zip(*neg_edges_set))) if neg_edges_set else torch.empty((2, 0), dtype=torch.long)    
    return pos_edges, neg_edges

def prepare_dynamic_data(stock_data, window_size=prev_date_num):
    snapshots = []
    bool_eerste = True
    already_done = set(fname.replace('.pkl', '') for fname in os.listdir(snapshot_path) if fname.endswith('.pkl'))

    aging_manager = EdgeAgingManager(relevance_threshold, max_age, date_to_idx)

    edge_info_pos = {}
    edge_info_neg = {}

    for i in tqdm(range(window_size-1, len(date_to_idx)), desc="Preparing snapshots"):
        # print('dit is i:', i)
        current_date = all_dates[i]
        # print('current date: ', current_date)
        if current_date in already_done:
            bool_eerste = False
            with open(os.path.join(snapshot_path, f"{current_date}.pkl"), 'rb') as f:
                loaded_snapshot = pickle.load(f)
                snapshots.append(loaded_snapshot)
                edge_info_pos = dict(loaded_snapshot['pos_edges_info'])
                edge_info_neg = dict(loaded_snapshot['neg_edges_info'])
            continue

        window_dates = all_dates[i-window_size+1:i+1]
        # print('dit is windowdates', len(window_dates), window_dates)
        window_data = stock_data[stock_data['Date'].isin(window_dates)]
        current_date_data = stock_data[stock_data['Date'] == current_date]

        feature_matrix = []
        for stock in unique_stocks:
            stock_data_current = current_date_data[current_date_data['Stock'] == stock]
            if len(stock_data_current) != 1:
                print(f"Fout bij feature matrix van {stock} op {current_date}")
            feature_matrix.append(stock_data_current[feature_cols2].values[0])
        feature_matrix = torch.FloatTensor(np.array(feature_matrix)).to(device)

        if bool_eerste:
            pos_pairs, neg_pairs = build_initial_edges_via_cosine_similarity(window_data, threshold)
            bool_eerste = False
            for src, dst in pos_pairs.T.tolist():
                aging_manager.update_edge(edge_info_pos, src, dst, 1.0, current_date)
            for src, dst in neg_pairs.T.tolist():
                aging_manager.update_edge(edge_info_neg, src, dst, -1.0, current_date)
        else:
            price_df = []
            for stock in unique_stocks:
                stockdf = window_data[window_data['Stock'] == stock]
                price_df.append(stockdf[feature_cols1].values.T)
            price_df = torch.FloatTensor(np.array(price_df))
            
            aging_manager.prune_edges(edge_info_pos, current_date, price_df)
            aging_manager.prune_edges(edge_info_neg, current_date, price_df)

            pos_pairs, neg_pairs = build_edges_via_balance_theory(
                edge_info_to_tensor(edge_info_pos),
                edge_info_to_tensor(edge_info_neg),
                len(unique_stocks),
                price_df,
                current_date
            )

            for src, dst in pos_pairs.T.tolist():
                aging_manager.update_edge(edge_info_pos, src, dst, 1.0, current_date)
            for src, dst in neg_pairs.T.tolist():
                aging_manager.update_edge(edge_info_neg, src, dst, -1.0, current_date)

        snapshots.append({
            'date': current_date,
            'features': feature_matrix.detach().cpu(),
            'pos_edges_info': dict(edge_info_pos),
            'neg_edges_info': dict(edge_info_neg),
            'tickers': unique_stocks,
            'full_window_data': window_data
        })

        with open(os.path.join(snapshot_path, f"{current_date}.pkl"), 'wb') as f:
            pickle.dump(snapshots[-1], f)

        write_header = not os.path.exists(log_path) or os.path.getsize(log_path) == 0
        with open(log_path, "a") as log_f:
            if write_header:
                log_f.write("date,nodes,pos_edges,neg_edges\n")
            pos_count = len(edge_info_pos)
            neg_count = len(edge_info_neg)
            log_f.write(f"{current_date},{len(unique_stocks)},{pos_count},{neg_count}\n")

    return snapshots

# eenmalig inladen van alle data
stock_data = load_all_stocks(daily_data_path)
all_dates = sorted(stock_data['Date'].unique())
date_to_idx = {date: idx for idx, date in enumerate(all_dates)}
raw_data = load_raw_stocks(raw_data_path, all_dates)
unique_stocks = sorted(stock_data['Stock'].unique())
stock_data = stock_data.sort_values(['Stock', 'Date'])

# start model

log_path = os.path.join(data_path, f"snapshot_log_mini.csv")
os.makedirs(os.path.dirname(log_path), exist_ok=True)
prepare_dynamic_data(stock_data)