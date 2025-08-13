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
hidden_dim = 32
num_epochs = 30
restrict_last_n_days= None # None of bv 80 om da laatse 60 dagen te nemen (20-day time window geraak je in begin altijd kwijt)
relevance_threshold = 0
max_age = 5
learning_rate = 0.0001

min_neighbors = 3
sim_threshold_pos = 0.4
sim_threshold_neg = -0.4
threshold = 0.4

edge_evaluation = True

def load_all_stocks(stock_data_path):
    all_stock_data = []
    for file in tqdm(os.listdir(stock_data_path), desc="Loading normalised data"):
    # for file in os.listdir(stock_data_path):
        if file.endswith('.csv'):
            df = pd.read_csv(os.path.join(stock_data_path, file))
            all_stock_data.append(df[['Date', 'Stock'] + feature_cols2])
    all_stock_data = pd.concat(all_stock_data, ignore_index=True)
    # print(all_stock_data.head()) # kleine test om te zien of data deftig is ingeladen

        # Enkel laatste X dagen
    if restrict_last_n_days is not None:
        all_dates = sorted(all_stock_data['Date'].unique())
        last_dates = all_dates[-restrict_last_n_days:]
        all_stock_data = all_stock_data[all_stock_data['Date'].isin(last_dates)]

    # print(all_stock_data.head())  # test of data deftig is

    return all_stock_data

def load_raw_stocks(raw_stock_path, all_dates):
    raw_files = [f for f in os.listdir(raw_stock_path) if f.endswith('.csv')]
    raw_data = {}
    for file in tqdm(raw_files, desc="Loading raw data for label creation"):
    # for file in raw_files:
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
        self.feature_norm = nn.LayerNorm(hidden_dim).to(device)

        # Positieve en negatieve convoluties
        self.pos_conv = GCNConv(hidden_dim, hidden_dim).to(device)
        self.neg_conv = GCNConv(hidden_dim, hidden_dim).to(device)

        self.pair_norm = nn.LayerNorm(hidden_dim).to(device)
        self.concat_norm = nn.LayerNorm(2 * hidden_dim).to(device) 

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
        sign_loss = -alpha * torch.cat([
            torch.log(1 + w_hat_pos),
            torch.log(1 - w_hat_neg)
        ]).mean()
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

def sign_semantics_aggregation(num_nodes, edge_list_pos, edge_list_neg, balance_theory_triads=True):
    """
    GPU-versnelde SSA: produceert delta_A_pos en delta_A_neg (shape: [N, N]) volgens balance theory.
    """
    device = edge_list_pos.device

    # 1. Directe edges
    A_pos = torch.zeros((num_nodes, num_nodes), dtype=torch.float32, device=device)
    A_neg = torch.zeros((num_nodes, num_nodes), dtype=torch.float32, device=device)
    if edge_list_pos.numel() > 0:
        A_pos[edge_list_pos[0], edge_list_pos[1]] = 1.0
    if edge_list_neg.numel() > 0:
        A_neg[edge_list_neg[0], edge_list_neg[1]] = 1.0

    if not balance_theory_triads:
        return A_pos, A_neg

    # 2. Triad closure via matrixvermenigvuldiging
    # Maak binaire maskers voor "positieve paden" en "negatieve paden"
    P1 = A_pos @ A_pos   # i → k → j met + +
    P2 = A_neg @ A_neg   # i → k → j met - -
    P3 = A_pos @ A_neg   # i → k → j met + -
    P4 = A_neg @ A_pos   # i → k → j met - +

    # Nieuwe kandidaten (nog geen directe verbinding)
    existing = (A_pos + A_neg) > 0
    suggested_pos = ((P1 + P2) > 0) & (~existing)
    suggested_neg = ((P3 + P4) > 0) & (~existing)

    # Maak uiteindelijke delta-matrices
    delta_A_pos = A_pos.clone()
    delta_A_neg = A_neg.clone()

    delta_A_pos[suggested_pos] = 1.0
    delta_A_neg[suggested_neg] = 1.0

    # Zorg dat de matrices symmetrisch zijn (zoals jouw origineel)
    delta_A_pos = torch.maximum(delta_A_pos, delta_A_pos.T)
    delta_A_neg = torch.maximum(delta_A_neg, delta_A_neg.T)

    return delta_A_pos, delta_A_neg

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

def build_initial_edges_via_correlation(window_data, threshold):
    grouped = window_data.groupby('Stock')[feature_cols1]
    stock_arrays = np.array([group.values.T for name, group in grouped])
    n_stocks = stock_arrays.shape[0]
    corr_matrix = np.zeros((n_stocks, n_stocks))

    # Bereken correlaties tussen alle paren van stocks
    for i in tqdm(range(n_stocks), desc="Calculating correlations"):
    # for i in range(n_stocks):
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

def evaluate_edges(model,snapshot, N, pred_pos, pred_neg):
    pos_edges_cos = snapshot['pos_edges_cos'].to(device)
    neg_edges_cos = snapshot['neg_edges_cos'].to(device)
    # pos_edges_ssa = snapshot['pos_edges_ssa'].to(device)
    # neg_edges_ssa = snapshot['neg_edges_ssa'].to(device)

    def edge_set(edges):
        return set(map(tuple, edges.T.cpu().numpy()))

    sets = {
        'cos_pos': edge_set(pos_edges_cos),
        # 'ssa_pos': edge_set(pos_edges_ssa),
        'pred_pos': edge_set(pred_pos),
        'cos_neg': edge_set(neg_edges_cos),
        # 'ssa_neg': edge_set(neg_edges_ssa),
        'pred_neg': edge_set(pred_neg),
    }

    log_row = {
        'date': snapshot['date'],
        'n_nodes': N,
        'cos_pos': len(sets['cos_pos']),
        # 'ssa_pos': len(sets['ssa_pos']),
        'pred_pos': len(sets['pred_pos']),
        'overlap_cos_pred_pos': len(sets['cos_pos'] & sets['pred_pos']),
        # 'overlap_ssa_pred_pos': len(sets['ssa_pos'] & sets['pred_pos']),
        # 'overlap_cos_ssa_pos': len(sets['cos_pos'] & sets['ssa_pos']),
        'cos_neg': len(sets['cos_neg']),
        # 'ssa_neg': len(sets['ssa_neg']),
        'pred_neg': len(sets['pred_neg']),
        'overlap_cos_pred_neg': len(sets['cos_neg'] & sets['pred_neg']),
        # 'overlap_ssa_pred_neg': len(sets['ssa_neg'] & sets['pred_neg']),
        # 'overlap_cos_ssa_neg': len(sets['cos_neg'] & sets['ssa_neg']),
        # 'ssa_pos_neg_overlap': len(sets['ssa_pos'] & sets['ssa_neg']),
        'cos_pos_to_pred_neg': len(sets['cos_pos'] & sets['pred_neg']),
        'cos_neg_to_pred_pos': len(sets['cos_neg'] & sets['pred_pos']),
    }

    eval_log_file = os.path.join(relation_path, "edge_evaluation_log.csv")
    write_header = not os.path.exists(eval_log_file)
    with open(eval_log_file, "a") as f:
        if write_header:
            f.write(','.join(log_row.keys()) + '\n')
        f.write(','.join(str(v) for v in log_row.values()) + '\n')

def prepare_dynamic_data(stock_data, window_size=20):
    # bool_eerste = True
    already_done = set(fname.replace('.pkl', '') for fname in os.listdir(snapshot_path) if fname.endswith('.pkl'))

    for i in tqdm(range(window_size-1, len(date_to_idx)), desc="Preparing snapshots"):
    # for i in range(window_size-1, len(date_to_idx)):

        current_date = all_dates[i]

        if current_date in already_done:
            continue

        window_dates = all_dates[i-window_size+1:i+1]
        window_data = stock_data[stock_data['Date'].isin(window_dates)]
        current_date_data = stock_data[stock_data['Date'] == current_date]

        grouped = current_date_data.groupby("Stock")
        feature_matrix = np.stack([
            grouped.get_group(stock)[feature_cols2].values[0] for stock in unique_stocks
        ])

        pos_pairs, neg_pairs = build_initial_edges_via_cosine_similarity(window_data)

        # pos_pairs_tensor = pos_pairs.to(device)
        # neg_pairs_tensor = neg_pairs.to(device)

        # delta_A_pos, delta_A_neg = sign_semantics_aggregation(
        #     len(unique_stocks), pos_pairs_tensor, neg_pairs_tensor
        # )
        # pos_edges_ssa = torch.nonzero(delta_A_pos).T.cpu()
        # neg_edges_ssa = torch.nonzero(delta_A_neg).T.cpu()

        snapshot_data = {
            'date': current_date,
            'features': feature_matrix,
            'pos_edges_cos': pos_pairs.cpu(),
            'neg_edges_cos': neg_pairs.cpu(),
            # 'pos_edges_ssa': pos_edges_ssa,
            # 'neg_edges_ssa': neg_edges_ssa,
            'tickers': unique_stocks,
            'full_window_data': window_data
        }
        
        with open(os.path.join(snapshot_path, f"{current_date}.pkl"), 'wb') as f:
            pickle.dump(snapshot_data, f)

        write_header = not os.path.exists(log_path) or os.path.getsize(log_path) == 0
        with open(log_path, "a") as log_f:
            if write_header:
                log_f.write("date,nodes,pos_edges_cos,neg_edges_cos,pos_edges_ssa,neg_edges_ssa\n")
            pos_count = pos_pairs.shape[1]
            neg_count = neg_pairs.shape[1]
            # pos_ssa_count = pos_edges_ssa.shape[1]
            # neg_ssa_count = neg_edges_ssa.shape[1]
            log_f.write(f"{current_date},{len(unique_stocks)},{pos_count},{neg_count}\n")

def edges_to_adj_matrix(edges, num_nodes):
    """Converteer edges naar adjacency matrix"""
    adj = torch.zeros((num_nodes, num_nodes))
    if edges.size(1) > 0:
        adj[edges[0], edges[1]] = 1.0
    return adj

def calculate_label(raw_df, current_date):
    date_idx = raw_df[raw_df['Date'] == current_date].index[0]
    # close_today = raw_df.iloc[date_idx]['Close']
    # close_tomorrow = raw_df.iloc[date_idx+1]['Close']
    # return (close_tomorrow / close_today) - 1
    close_today = raw_df.iloc[date_idx]['Close']
    close_yesterday = raw_df.iloc[date_idx-1]['Close']
    return (close_today / close_yesterday) - 1


def main1_generate():
    num_snapshots = len([fname for fname in os.listdir(snapshot_path) if fname.endswith('.pkl')])

    model = DynamiSE(num_features=len(feature_cols2), hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_loss = float('inf')
    training_results = []

    for epoch in range(num_epochs):
        model.train()
        epoch_losses = []
        early_stop_due_to_nan = False  # voeg deze flag toe

        for date in tqdm(all_dates[prev_date_num-1:], desc=f"Epoch {epoch+1} van de {num_epochs}"):
        # for date in all_dates[prev_date_num-1:]:
            snapshot_file = os.path.join(snapshot_path, f"{date}.pkl")
            if not os.path.exists(snapshot_file):
                print(f"Error: {snapshot_file} for date {date} not found.")
                continue
            with open(snapshot_file, 'rb') as f:
                snapshot = pickle.load(f)

            optimizer.zero_grad()
            features = torch.from_numpy(snapshot['features']).float().to(device)
            edge_index_pos_cos = snapshot['pos_edges_cos'].to(device)
            edge_index_neg_cos = snapshot['neg_edges_cos'].to(device)
            t = torch.tensor([0.0, 1.0], device=device)

            try:
                embeddings = model(features, edge_index_pos_cos, edge_index_neg_cos, t)
                loss = model.full_loss(embeddings, edge_index_pos_cos, edge_index_neg_cos)

                if torch.isnan(loss):
                    print(f" NaN loss detected during epoch {epoch+1}, date {date}.")
                    raise ValueError("NaN in loss")

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_losses.append(loss.item())

            except ValueError as e:
                print(f" Training stopped early due to numerical issue: {e}")
                print(" Saving best model so far and exiting training loop.")
                early_stop_due_to_nan = True
                break  # verlaat binnenste loop

        if early_stop_due_to_nan:
            training_results.append("NaN detected")
            break  # verlaat de outer loop

        if not epoch_losses:
            break

        avg_loss = np.average(epoch_losses, weights=np.arange(1, len(epoch_losses)+1))
        training_results.append(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(relation_path, "best_model.pth"))
            print(f" Best model so far saved with loss {best_loss:.4f} (epoch {epoch+1})")
        else:
            print(f"Epoch {epoch+1} completed — avg loss: {avg_loss:.4f}")

    # log resultaten zelfs als training vroegtijdig stopt
    results_df = pd.DataFrame({'epoch': range(1, len(training_results)+1), 'loss': training_results})
    results_df.to_csv(os.path.join(relation_path, "training_results.csv"), index=False)
    print(" Training results saved.")

            
def main1_load():
    model = DynamiSE(num_features=len(feature_cols2), hidden_dim=hidden_dim)
    model.load_state_dict(torch.load(os.path.join(relation_path, "best_model.pth"), map_location=device))
    model.eval()

    for date in tqdm(all_dates[prev_date_num-1:-1], desc="Generating outputs"):
    # for date in all_dates[prev_date_num-1:-1]:
        snapshot_file = os.path.join(snapshot_path, f"{date}.pkl")
        if not os.path.exists(snapshot_file):
            print(f"Error: {snapshot_file} for date {date} not found.")
            continue
            
        with open(snapshot_file, 'rb') as f:
            snapshot = pickle.load(f)

        with torch.no_grad():
            N = len(snapshot['tickers'])
            features = torch.from_numpy(snapshot['features']).float().to(device)
            t = torch.tensor([0.0, 1.0], device=device)

            
            embeddings = model(features, snapshot['pos_edges_cos'].to(device), snapshot['neg_edges_cos'].to(device), t)

            # Combineer originele edges en voorspel w_hat
            N = embeddings.shape[0]
            candidate_edges = torch.combinations(torch.arange(N), r=2).T.to(device)
            edge_scores = model.predict_edge_weight(embeddings, candidate_edges) # is deze gemaakt voor negatief en positief tesamen te doen?

            # Filter edges op basis van model-output
            pos_mask = edge_scores > threshold
            neg_mask = edge_scores < -threshold

            new_pos_edges = candidate_edges[:, pos_mask]
            new_neg_edges = candidate_edges[:, neg_mask]
            # symmetrie
            pos_pairs = torch.cat([new_pos_edges, new_pos_edges[[1, 0], :]], dim=1)
            neg_pairs = torch.cat([new_neg_edges, new_neg_edges[[1, 0], :]], dim=1)
            # Maak refined adjacencymatrices
            pos_adj = edges_to_adj_matrix(pos_pairs, N).to(device)
            neg_adj = edges_to_adj_matrix(neg_pairs, N).to(device)

            if edge_evaluation == True:
                evaluate_edges(model, snapshot, N, new_pos_edges, new_neg_edges)

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
    relation_path = os.path.join(data_path, "relation_cosineDSC_t1")
    os.makedirs(relation_path, exist_ok=True)
    snapshot_path= os.path.join(data_path, "intermediate_snapshots_cosineDSC_t1")
    os.makedirs(snapshot_path, exist_ok=True)
    data_train_predict_path = os.path.join(data_path, "data_train_predict_cosineDSC_t1")
    os.makedirs(data_train_predict_path, exist_ok=True)
    daily_stock_path = os.path.join(data_path, "daily_stock_cosineDSC_t1")
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

    main1_generate()
    main1_load()

def SP500():
    # alle paden relatief aanmaken
    global base_path, data_path, daily_data_path, raw_data_path, relation_path, snapshot_path, data_train_predict_path, daily_stock_path, log_path, stock_data, all_dates, date_to_idx, raw_data, unique_stocks
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_path, "data", "S&P500")
    daily_data_path = os.path.join(data_path, "normaliseddailydata")
    raw_data_path = os.path.join(data_path, "stockdata")
    # kies hieronder de map waarin je de resultaten wilt opslaan
    relation_path = os.path.join(data_path, "relation_cosineDSC_t1")
    os.makedirs(relation_path, exist_ok=True)
    snapshot_path= os.path.join(data_path, "intermediate_snapshots_cosineDSC_t1")
    os.makedirs(snapshot_path, exist_ok=True)
    data_train_predict_path = os.path.join(data_path, "data_train_predict_cosineDSC_t1")
    os.makedirs(data_train_predict_path, exist_ok=True)
    daily_stock_path = os.path.join(data_path, "daily_stock_cosineDSC_t1")
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

    main1_generate()
    main1_load()

def testbatch_mini():
    # alle paden relatief aanmaken
    global base_path, data_path, daily_data_path, raw_data_path, relation_path, snapshot_path, data_train_predict_path, daily_stock_path, log_path, stock_data, all_dates, date_to_idx, raw_data, unique_stocks
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_path, "data", "testbatch_mini")
    daily_data_path = os.path.join(data_path, "normaliseddailydata")
    raw_data_path = os.path.join(data_path, "stockdata")
    # kies hieronder de map waarin je de resultaten wilt opslaan
    relation_path = os.path.join(data_path, "relation_cosineDSC")
    os.makedirs(relation_path, exist_ok=True)
    snapshot_path= os.path.join(data_path, "intermediate_snapshots_cosineDSC")
    os.makedirs(snapshot_path, exist_ok=True)
    data_train_predict_path = os.path.join(data_path, "data_train_predict_cosineDSC")
    os.makedirs(data_train_predict_path, exist_ok=True)
    daily_stock_path = os.path.join(data_path, "daily_stock_cosineDSC")
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

    main1_generate()
    main1_load()

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
        relation_path = os.path.join(data_path, "relation_cosineDSC")
        os.makedirs(relation_path, exist_ok=True)
        snapshot_path= os.path.join(data_path, "intermediate_snapshots_cosineDSC")
        os.makedirs(snapshot_path, exist_ok=True)
        data_train_predict_path = os.path.join(data_path, "data_train_predict_cosineDSC")
        os.makedirs(data_train_predict_path, exist_ok=True)
        daily_stock_path = os.path.join(data_path, "daily_stock_cosineDSC")
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

        main1_generate()
        main1_load()

SP500()
CSI300()
# testbatch_mini()
# nasdaq5batches()