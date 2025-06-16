import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torchdiffeq import odeint
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
import os
import itertools
from collections import defaultdict
import torch.nn.functional as F
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu') 
print(f"Device: {device}")

# alle paden relatief aanmaken
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(base_path, "data", "testbatch_mini")
daily_data_path = os.path.join(data_path, "normaliseddailydata")
raw_data_path = os.path.join(data_path, "stockdata")
# kies hieronder de map waarin je de resultaten wilt opslaan
relation_path = os.path.join(data_path, "relation_dynamiSE_mini")
os.makedirs(relation_path, exist_ok=True)
snapshot_path= os.path.join(data_path, "intermediate_snapshots_mini")
os.makedirs(snapshot_path, exist_ok=True)
data_train_predict_path = os.path.join(data_path, "data_train_predict_mini")
os.makedirs(data_train_predict_path, exist_ok=True)
daily_stock_path = os.path.join(data_path, "daily_stock_mini")
os.makedirs(daily_stock_path, exist_ok=True)

# Hyperparameters
prev_date_num = 20
feature_cols1 = ['Open', 'High', 'Low', 'Close']
feature_cols2 = ['Open', 'High', 'Low', 'Close', 'Volume', 'Turnover']
hidden_dim = 32
num_epochs = 30
restrict_last_n_days= 100 # None of bv 80 om da laatse 60 dagen te nemen (20-day time window geraak je in begin altijd kwijt)
relevance_threshold = 0
max_age = 5
learning_rate = 0.0001
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

class DynamiSE(nn.Module):
    def __init__(self, num_features, hidden_dim):
        super(DynamiSE, self).__init__()
        self.hidden_dim = hidden_dim

        self.feature_encoder = nn.Linear(num_features, hidden_dim).to(device)
        self.feature_norm = nn.LayerNorm(hidden_dim)

        # Positieve en negatieve convoluties
        self.pos_conv = GCNConv(hidden_dim, hidden_dim).to(device)
        self.neg_conv = GCNConv(hidden_dim, hidden_dim).to(device)

        self.pair_norm = nn.LayerNorm(hidden_dim)
        self.concat_norm = nn.LayerNorm(2 * hidden_dim) 

        # Combinatiefunctie Ψ
        self.psi = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim, eps=1e-6),
            nn.ReLU()
        ).to(device)

        self.ode_func = ODEFunc(hidden_dim, self.pos_conv, self.neg_conv).to(device)

        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        ).to(device)

    def forward(self, x, edge_index_pos, edge_index_neg, t, method='dopri5'):
        
        # print(x.shape)
        x = x.to(device)
        h = self.feature_norm(self.feature_encoder(x))
        # print(f"\nFeature encoder out - mean: {h.mean().item():.4f}, std: {h.std().item():.4f}")
        # print(h.shape)
        if torch.isnan(h).any() or torch.isinf(h).any():
            print(f"h bevat NaN of Inf op snapshot {self.snapshot_date if hasattr(self, 'snapshot_date') else '??'}")
            print(h)
            raise ValueError("h bevat NaN of Inf")
        edge_index_pos = edge_index_pos.to(device)
        edge_index_neg = edge_index_neg.to(device)
        self.ode_func.set_graph(edge_index_pos, edge_index_neg)
        t = t.to(device)
        h = odeint(self.ode_func, h, t, 
               method=method,
               rtol=1e-3,
               atol=1e-4,
               options={'max_num_steps': 100})[1]
        # print(f"ODE out - mean: {h.mean().item():.4f}, std: {h.std().item():.4f}")
        return h

    def predict_edge_weight(self, h, edge_index, combine_method='hadamard'):
        src, dst = edge_index.long()
        h_src, h_dst = h[src], h[dst]
        
        if combine_method == 'hadamard':
            h_pair = self.pair_norm(h_src * h_dst)
        elif combine_method == 'concat':
            h_pair = self.concat_norm(torch.cat([h_src, h_dst], dim=1))
        elif combine_method == 'average':
            h_pair = (h_src + h_dst) / 2
        elif combine_method == 'subtract':
            h_pair = h_src - h_dst
        else:
            raise ValueError("Ongeldige combinatiemethode")
        
        return torch.tanh(self.predictor(h_pair).squeeze())

    def full_loss(self, h, pos_edges, neg_edges, alpha=0.1, beta=0.001):
        # Reconstructieverlies (RMSE)

        w_hat_pos = self.predict_edge_weight(h, pos_edges)
        w_true_pos = torch.full_like(w_hat_pos, +1, dtype=torch.float32)
        loss_pos = (w_hat_pos - w_true_pos).pow(2)

        w_hat_neg = self.predict_edge_weight(h, neg_edges)
        w_true_neg = torch.full_like(w_hat_neg, -1, dtype=torch.float32)
        loss_neg = (w_hat_neg - w_true_neg).pow(2)

        # print(f"loss_pos range: [{loss_pos.min().item():.4f}, {loss_pos.max().item():.4f}]")
        # print(f"loss_neg range: [{loss_neg.min().item():.4f}, {loss_neg.max().item():.4f}]")
        recon_loss = torch.cat([loss_pos, loss_neg]).mean()
        # print(loss_pos.shape())

        # Teken-constraint (paper Eq.7)
        # print(f"w_hat_pos range: [{w_hat_pos.min().item():.4f}, {w_hat_pos.max().item():.4f}]")
        # print(f"w_hat_neg range: [{w_hat_neg.min().item():.4f}, {w_hat_neg.max().item():.4f}]")
        # log_pos = torch.log(1 + w_hat_pos)
        # log_neg = torch.log(1 - w_hat_neg)
        # print(f"log_pos range: [{log_pos.min().item():.4f}, {log_pos.max().item():.4f}]")
        # print(f"log_neg range: [{log_neg.min().item():.4f}, {log_neg.max().item():.4f}]")
        sign_loss = -alpha * (
            torch.log(1 + w_hat_pos).mean() + 
            torch.log(1 - w_hat_neg).mean()
        )
        #print("w_hat_pos:", w_hat_pos.min().item(), w_hat_pos.max().item(), w_hat_pos.mean().item())
        #print("w_hat_neg:", w_hat_neg.min().item(), w_hat_neg.max().item(), w_hat_neg.mean().item())

        # Regularisatie
        reg_loss = beta * h.norm(p=2).mean()
        total_loss = recon_loss + sign_loss + reg_loss
        # print(f"Loss components - recon: {recon_loss.item():.4f}, sign: {sign_loss.item():.4f}, reg: {reg_loss.item():.4f}")
        if torch.isnan(total_loss):
            print("NaN in loss! Breaking down components:")
            print("recon_loss:", recon_loss)
            print("sign_loss:", sign_loss)
            print("reg_loss:", reg_loss)
            return torch.tensor(0.0, requires_grad=True)
        return total_loss


class ODEFunc(nn.Module):
    # def __init__(self, hidden_dim, pos_conv, neg_conv, psi):
    #     super(ODEFunc, self).__init__()
    #     self.pos_conv = pos_conv
    #     self.neg_conv = neg_conv
    #     self.psi = psi
    #     self.edge_index_pos = None
    #     self.edge_index_neg = None

    #     # Stabilisatielagen
    #     self.layer_norm = nn.LayerNorm(hidden_dim)
    #     self.dropout = nn.Dropout(0.1)
    def __init__(self, hidden_dim, pos_conv, neg_conv, damping = 0.1):
        super(ODEFunc, self).__init__()
        self.pos_conv = pos_conv
        self.neg_conv = neg_conv
        self.psi = nn.Sequential(
        nn.Linear(hidden_dim*2, hidden_dim, bias=False),
        nn.Tanh()
        )
        self.edge_index_pos = None
        self.edge_index_neg = None

        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.damping = damping

    def set_graph(self, edge_index_pos, edge_index_neg):
        self.edge_index_pos = edge_index_pos.long()
        self.edge_index_neg = edge_index_neg.long()

    def forward(self, t, h):
        # Voeg layer normalization toe voor stabiliteit
        h = self.layer_norm(h)

        # Pos/Neg-specifieke propagatie (paper Eq.3-4)
        h_pos = self.pos_conv(h, self.edge_index_pos.long())
        h_neg = self.neg_conv(h, self.edge_index_neg.long())
        
        # Dropout voor regularisatie
        # h_pos = self.dropout(h_pos)
        # h_neg = self.dropout(h_neg)
        delta = self.psi(torch.cat([h_pos, h_neg], dim=1))
        # Lineaire combinatie (paper Eq.5)
        delta_h = delta - self.damping * h
        # print(f"ODE delta_h range: [{delta_h.min():.2f}, {delta_h.max():.2f}]")
        return delta_h.clamp(-50, 50)

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
                cos_sim = cosine_similarity(stock_arrays[i,f,:], stock_arrays[j,f,:])
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
        strong_pos = np.where(cos_matrix[i] > threshold)
        print(strong_pos)
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

def build_initial_edges_via_correlation(window_data, threshold):
    grouped = window_data.groupby('Stock')[feature_cols1]
    stock_arrays = np.array([group.values.T for name, group in grouped])
    n_stocks = stock_arrays.shape[0]
    corr_matrix = np.zeros((n_stocks, n_stocks))

    # Bereken correlaties tussen alle paren van stocks
    for i in tqdm(range(n_stocks), desc="Calculating correlations"):
        for j in range(i+1, n_stocks):
            feature_correlations = []
            for f in range(len(feature_cols1)):
                corr = np.corrcoef(stock_arrays[i,f,:], stock_arrays[j,f,:])[0,1]
                feature_correlations.append(corr)           
            avg_corr = np.nanmean(feature_correlations)
            corr_matrix[i,j] = avg_corr
            corr_matrix[j,i] = avg_corr

    # Bouw edges op basis van drempelwaarde
    pos_edges = []
    neg_edges = []

    # Garandeer minimum aantal buren
    for i in range(n_stocks):
        # Positieve edges
        strong_pos = np.where(corr_matrix[i] > threshold)[0]
        if len(strong_pos) < min_neighbors:
            # Voeg extra buren toe als er te weinig zijn
            corrs = corr_matrix[i].copy()
            top_pos = np.argsort(-corrs)[:min_neighbors]
            for j in top_pos:
                if corr_matrix[i,j] > 0:  # Alleen positieve correlaties toevoegen
                    pos_edges.append((i, j))
                    pos_edges.append((j, i))
        else:
            # Gebruik alleen de sterke correlaties
            for j in strong_pos:
                pos_edges.append((i, j))
                pos_edges.append((j, i))
        
        # Negatieve edges
        strong_neg = np.where(corr_matrix[i] < -threshold)[0]
        if len(strong_neg) < min_neighbors:
            # Voeg extra buren toe als er te weinig zijn
            corrs = corr_matrix[i].copy()
            top_neg = np.argsort(corrs)[:min_neighbors]
            for j in top_neg:
                if corr_matrix[i,j] < 0:  # Alleen negatieve correlaties toevoegen
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

def prepare_dynamic_data(stock_data, window_size=20):
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
                edge_info_pos[(src, dst)] = {'created_at': current_date, 'last_score': 1.0, 'last_update': current_date}
            for src, dst in neg_pairs.T.tolist():
                edge_info_neg[(src, dst)] = {'created_at': current_date, 'last_score': -1.0, 'last_update': current_date}
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


def edge_prediction_loss(h, edge_index, sign, model):
    """Loss op basis van voorspelde ŵ_{ij} tegenover ground truth sign."""
    w_hat = model.predict_edge_weight(h, edge_index)
    # print(f"w_hat range: [{w_hat.min():.2f}, {w_hat.max():.2f}]")
    w_true = torch.full_like(w_hat, sign, dtype=torch.float32)
    return ((w_hat - w_true) ** 2).mean()

def edges_to_adj_matrix(edges, num_nodes):
    """Converteer edges naar adjacency matrix"""
    adj = torch.zeros((num_nodes, num_nodes))
    if edges.size(1) > 0:
        adj[edges[0], edges[1]] = 1.0
    return adj

def calculate_label(raw_df, current_date):
    date_idx = raw_df[raw_df['Date'] == current_date].index[0]
    close_today = raw_df.iloc[date_idx]['Close']
    close_yesterday = raw_df.iloc[date_idx-1]['Close']
    return (close_today / close_yesterday) - 1


def main1_generate():
    print(f"Aantal snapshots: {len(date_to_idx)}")

    model = DynamiSE(num_features=len(feature_cols2), hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_loss = float('inf')
    training_results = []

    for epoch in range(num_epochs):
        model.train()
        epoch_losses = []

        for date in tqdm(all_dates[prev_date_num-1:], desc=f"Epoch {epoch+1} van de {num_epochs}"):
            snapshot_file = os.path.join(snapshot_path, f"{date}.pkl")
            if not os.path.exists(snapshot_file):
                print(f"Error: {snapshot_file} for date {date} not found.")
                continue
            with open(snapshot_file, 'rb') as f:
                snapshot = pickle.load(f)

            optimizer.zero_grad()

            features = snapshot['features'].float().to(device)
            pos_edges_tensor = edge_info_to_tensor(snapshot['pos_edges_info']).to(device)
            neg_edges_tensor = edge_info_to_tensor(snapshot['neg_edges_info']).to(device)
            t = torch.tensor([0.0, 1.0], device=device)

            # print("\n", snapshot['date'])
            # print("Input features stats - min:", features.min(), "max:", features.max())
            # print("pos edge shape: ", pos_edges_tensor.shape, "\nneg edge shape: ", neg_edges_tensor.shape)

            # with torch.autograd.set_detect_anomaly(True):
            embeddings = model(
                features,
                pos_edges_tensor,
                neg_edges_tensor,
                t
            )
            loss = model.full_loss(embeddings, pos_edges_tensor, neg_edges_tensor)
            if torch.isnan(loss):
                print("NaN loss detected!")
                for name, param in model.named_parameters():
                    if torch.isnan(param.grad).any():
                        print(f"NaN in gradients of {name}")
                raise ValueError("NaN in loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_losses.append(loss.item())

        avg_loss = np.average(epoch_losses, weights=np.arange(1, len(epoch_losses)+1))
        print(f"Epoch {epoch+1}, Avg Loss: {avg_loss}")
        training_results.append(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(relation_path, "best_model.pth"))
            print(f"Beste model opgeslagen in {relation_path} met loss {best_loss}")

    results_df = pd.DataFrame({'epoch': range(1, len(training_results)+1), 'loss': training_results})
    results_df.to_csv(os.path.join(relation_path, "training_results.csv"), index=False)
    print(f"Trainingsresultaten opgeslagen.")

            
def main1_load():
    model = DynamiSE(num_features=len(feature_cols2), hidden_dim=hidden_dim)
    model.load_state_dict(torch.load(os.path.join(relation_path, "best_model.pth"), map_location=device))
    model.eval()

    for date in tqdm(all_dates[prev_date_num-1:], desc="Generating outputs"):
        snapshot_file = os.path.join(snapshot_path, f"{date}.pkl")
        if not os.path.exists(snapshot_file):
            print(f"Error: {snapshot_file} for date {date} not found.")
            continue
            
        with open(snapshot_file, 'rb') as f:
            snapshot = pickle.load(f)

        with torch.no_grad():
            N = len(snapshot['tickers'])
            pos_edges_tensor = edge_info_to_tensor(snapshot['pos_edges_info']).to(device)
            neg_edges_tensor = edge_info_to_tensor(snapshot['neg_edges_info']).to(device)
            #vanaf hier is het vervangen:
            # pos_adj = edges_to_adj_matrix(pos_edges_tensor, N).to(device)
            # neg_adj = edges_to_adj_matrix(neg_edges_tensor, N).to(device)
            features = snapshot['features'].float().to(device)
            t = torch.tensor([0.0, 1.0], device=device)

            embeddings = model(features, pos_edges_tensor, neg_edges_tensor, t)

            # Combineer originele edges en voorspel w_hat
            all_edges_tensor = torch.cat([pos_edges_tensor, neg_edges_tensor], dim=1) # maar dit maakt dan zowel positief als negatief 1?
            edge_scores = model.predict_edge_weight(embeddings, all_edges_tensor) # is deze gemaakt voor negatief en positief tesamen te doen?
            num_pos = pos_edges_tensor.shape[1]
            scores_pos = edge_scores[:num_pos]
            scores_neg = edge_scores[num_pos:]

            # Filter edges op basis van model-output
            new_pos_edges = all_edges_tensor[:, :num_pos][:, scores_pos > 0.3]
            new_neg_edges = all_edges_tensor[:, num_pos:][:, scores_neg < -0.3]

            # Maak refined adjacencymatrices
            pos_adj = edges_to_adj_matrix(new_pos_edges, N).to(device)
            neg_adj = edges_to_adj_matrix(new_neg_edges, N).to(device)

            #hier stopt het vervangen

            end_date = snapshot['date']
            end_idx = date_to_idx[end_date]
            start_idx = end_idx - prev_date_num + 1
            if start_idx < 0:
                print(f"Skipping {end_date} - not enough history")
                continue

            features, labels, stock_info = [], [], []
            window_data = snapshot['full_window_data']
            grouped = window_data.groupby('Stock')
            stock_groups = {name: group for name, group in grouped}

            for stock_name in snapshot['tickers']:
                stock_data = stock_groups.get(stock_name)
                if len(stock_data) == prev_date_num:
                    features.append(stock_data[feature_cols2].values)
                    raw_df = raw_data[stock_name]
                    labels.append(calculate_label(raw_df, snapshot['date']))
                    stock_info.append([stock_name, end_date])
                else:
                    print(f"Window data klopt niet voor {stock_name} op {end_date}")

            with open(os.path.join(data_train_predict_path, f"{end_date}.pkl"), 'wb') as f:
                pickle.dump({
                    'pos_adj': pos_adj,
                    'neg_adj': neg_adj,
                    'features': torch.FloatTensor(np.array(features)),
                    'labels': torch.FloatTensor(labels),
                    'mask': [True] * len(labels)
                }, f)

            pd.DataFrame(stock_info, columns=['code', 'dt']).to_csv(
                os.path.join(daily_stock_path, f"{end_date}.csv"), index=False)


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
snapshots = prepare_dynamic_data(stock_data)


main1_generate()
main1_load()