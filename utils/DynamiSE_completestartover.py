import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from torchdiffeq import odeint_adjoint as odeint
import numpy as np
import pandas as pd
from itertools import combinations
import pickle
from tqdm import tqdm
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu') 
print(f"Device: {device}")

# alle paden relatief aanmaken
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(base_path, "data", "testbatch_mini")
daily_data_path = os.path.join(data_path, "normaliseddailydata")
raw_data_path = os.path.join(data_path, "stockdata")
# kies hieronder de map waarin je de resultaten wilt opslaan
relation_path = os.path.join(data_path, "relation_dynamiSE_completestartovertest")
os.makedirs(relation_path, exist_ok=True)
snapshot_path= os.path.join(data_path, "intermediate_snapshots_mini")
os.makedirs(snapshot_path, exist_ok=True)
data_train_predict_path = os.path.join(data_path, "data_train_predict_completestartovertest")
os.makedirs(data_train_predict_path, exist_ok=True)
daily_stock_path = os.path.join(data_path, "daily_stock_completestartovertest")
os.makedirs(daily_stock_path, exist_ok=True)

# Hyperparameters
prev_date_num = 20
feature_cols1 = ['Open', 'High', 'Low', 'Close']
feature_cols2 = ['Open', 'High', 'Low', 'Close', 'Volume', 'Turnover']
hidden_dim = 32
num_epochs = 30
restrict_last_n_days= 100 # None of bv 80 om da laatse 60 dagen te nemen (20-day time window geraak je in begin altijd kwijt)
learning_rate = 0.0001
score_pos_threshold = 0.5
score_neg_threshold = 0.5

def edge_info_to_tensor(edge_info):
    if not edge_info:
        return torch.empty((2, 0), dtype=torch.long).to(device)
    edges = list(edge_info.keys())
    return torch.LongTensor(list(zip(*edges))).to(device)

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

# 1. Data Preparation (Onveranderd uit originele code)
# ---------------------------------------------------
# Deze sectie blijft identiek omdat de data-verwerking goed was
# Bevat paden, hyperparameters en hulpfuncties voor data loading

class SignSemanticsAggregator(nn.Module):
    """
    Verbeterde SSA-unit volgens paper Sectie 3.2
    Combineert elementen uit beide versies met balance theory implementatie
    """
    def __init__(self):
        super().__init__()
        
    def find_indirect_relations(self, A_pos, A_neg):
        """Geïmplementeerd volgens balance theory (Figuur 2 in paper)"""
        N = max(A_pos.max().item() + 1 if A_pos.numel() > 0 else 0, 
                A_neg.max().item() + 1 if A_neg.numel() > 0 else 0)
        
        # Bouw adjacency matrices
        pos_adj = torch.zeros((N, N), device=A_pos.device)
        neg_adj = torch.zeros((N, N), device=A_neg.device)
        
        if A_pos.size(1) > 0:
            pos_adj[A_pos[0], A_pos[1]] = 1
        if A_neg.size(1) > 0:
            neg_adj[A_neg[0], A_neg[1]] = 1
            
        new_pos_edges = []
        new_neg_edges = []
        
        # Check alle mogelijke driehoeken volgens balance theory
        for i, j, k in combinations(range(N), 3):
            # Vier balance theory regels (paper Sectie 2.2)
            for a, b, c in [(i, j, k), (i, k, j), (j, i, k)]:
                if (pos_adj[a, b] or neg_adj[a, b]) and (pos_adj[b, c] or neg_adj[b, c]):
                    sign_ab = 1 if pos_adj[a, b] else -1
                    sign_bc = 1 if pos_adj[b, c] else -1
                    predicted_sign = sign_ab * sign_bc
                    
                    if predicted_sign == 1 and not pos_adj[a, c]:
                        new_pos_edges.append((a, c))
                    elif predicted_sign == -1 and not neg_adj[a, c]:
                        new_neg_edges.append((a, c))
                        
        return (torch.tensor(new_pos_edges).T if new_pos_edges else torch.empty((2, 0)),
                torch.tensor(new_neg_edges).T if new_neg_edges else torch.empty((2, 0)))

    def forward(self, A_pos_t, A_pos_tp1, A_neg_t, A_neg_tp1):
        """Paper Eq. 1: ΔA±ᵗ = SSA_Unit(ΔAᵗ)"""
        ΔA_pos = self.edge_index_diff(A_pos_tp1, A_pos_t)
        ΔA_neg = self.edge_index_diff(A_neg_tp1, A_neg_t)
        
        indirect_pos, indirect_neg = self.find_indirect_relations(A_pos_tp1, A_neg_tp1)
        
        ΔA_pos = torch.cat([ΔA_pos, indirect_pos], dim=1) if ΔA_pos.size(1) > 0 else indirect_pos
        ΔA_neg = torch.cat([ΔA_neg, indirect_neg], dim=1) if ΔA_neg.size(1) > 0 else indirect_neg
        
        return ΔA_pos.to(device), ΔA_neg.to(device)

    @staticmethod
    def edge_index_diff(new_edges, old_edges):
        """Bereken edge verschillen tussen tijdstappen"""
        new_set = set(map(tuple, new_edges.t().tolist())) if new_edges.size(1) > 0 else set()
        old_set = set(map(tuple, old_edges.t().tolist())) if old_edges.size(1) > 0 else set()
        diff = list(new_set.symmetric_difference(old_set))
        return torch.tensor(diff, dtype=torch.long).t().to(new_edges.device) if diff else torch.empty((2, 0), device=new_edges.device)

class GCNODEFunc(nn.Module):
    """
    ψ-functie uit paper Eq. 3
    Combineert GCN met tijdsafhankelijkheid uit tweede versie
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.gcn = GCNConv(in_dim, out_dim)
        self.time_weights = nn.Linear(1, out_dim, bias=False)  # Tijdsafhankelijkheid

    def forward(self, t, x, edge_index):
        # Paper Eq. 3: dΔH±ᵗ/dt = σ(ΔA±ᵗHᵗW)
        time_factor = self.time_weights(t.view(1, 1)).sigmoid()  # [0,1]
        return F.relu(self.gcn(x, edge_index)) * time_factor

class DynamicSignCollaboration(nn.Module):
    """
    DSC-unit uit paper Sectie 3.3
    Behoudt ODE-integratie uit eerste versie maar met tijdsafhankelijkheid
    """
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.odefunc_pos = GCNODEFunc(in_dim, hidden_dim)
        self.odefunc_neg = GCNODEFunc(in_dim, hidden_dim)

    def forward(self, x, pos_edge_index, neg_edge_index, steps=5):
        """Paper Eq. 4: ODE-integratie voor beide edge typen"""
        t_span = torch.linspace(0, 1, steps).to(x.device)
        
        # Positieve edges
        z_pos = odeint(self.odefunc_pos, x, t_span, 
                      method='rk4', options={'step_size':0.1},
                      adjoint=False, edge_index=pos_edge_index)[-1] if pos_edge_index.size(1) > 0 else torch.zeros_like(x)
        
        # Negatieve edges
        z_neg = odeint(self.odefunc_neg, x, t_span,
                      method='rk4', options={'step_size':0.1},
                      adjoint=False, edge_index=neg_edge_index)[-1] if neg_edge_index.size(1) > 0 else torch.zeros_like(x)
        
        return z_pos, z_neg

class DynamiSE(nn.Module):
    """
    Hoofdmodel volgens paper Sectie 3
    Combineert beste elementen uit beide versies:
    - SSA-unit met balance theory
    - DSC-unit met tijdsafhankelijke ODE
    - Init GCN uit tweede versie
    """
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        # Init embedding volgens paper Sectie 4.3
        self.init_gcn = GCNConv(in_dim, hidden_dim)
        
        # Core componenten
        self.ssa = SignSemanticsAggregator()
        self.dsc = DynamicSignCollaboration(hidden_dim, hidden_dim)
        
        # Combineren volgens paper Eq. 5
        self.combine = nn.Linear(2 * hidden_dim, hidden_dim)
        
        # Predictie
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1))
        
        # Normalisatie
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, H_t, A_pos_t, A_pos_tp1, A_neg_t, A_neg_tp1):
        """Paper Eq. 1-5 gecombineerd"""
        # Init embedding
        H_t = F.relu(self.init_gcn(H_t, torch.cat([A_pos_t, A_neg_t], dim=1)))
        
        # SSA fase
        ΔA_pos, ΔA_neg = self.ssa(A_pos_t, A_pos_tp1, A_neg_t, A_neg_tp1)
        
        # DSC fase
        Z_pos, Z_neg = self.dsc(H_t, ΔA_pos, ΔA_neg)
        
        # Combineren
        return self.norm(self.combine(torch.cat([Z_pos, Z_neg], dim=-1)))

    def predict_link(self, Z, edge_index, combine_op='hadamard'):
        """Paper Sectie 3.4 met opties uit Table 2"""
        src, dst = edge_index
        z_i, z_j = Z[src], Z[dst]
        
        if combine_op == 'hadamard':
            pair = z_i * z_j
        elif combine_op == 'concat':
            pair = torch.cat([z_i, z_j], dim=-1)
        elif combine_op == 'average':
            pair = (z_i + z_j) / 2
        elif combine_op == 'subtract':
            pair = z_i - z_j
            
        return torch.tanh(self.predictor(pair)).squeeze()

    def full_loss(self, Z, pos_edges, neg_edges, alpha=0.1, beta=0.001):
        """Paper Eq. 7 met stabiliteitsverbeteringen uit tweede versie"""
        w_hat_pos = self.predict_link(Z, pos_edges)
        w_hat_neg = self.predict_link(Z, neg_edges)

        # Reconstructieverlies
        recon_loss = (torch.cat([
            (w_hat_pos - torch.ones_like(w_hat_pos)).pow(2),
            (w_hat_neg - torch.ones_like(w_hat_neg)).pow(2)
        ])).mean()

        # Sign-constraint met clamping voor stabiliteit
        sign_loss = -alpha * (
            torch.log(1 + w_hat_pos.clamp(max=0.999)).mean() +
            torch.log(1 - w_hat_neg.clamp(min=-0.999)).mean()
        )

        # Regularisatie
        reg_loss = beta * Z.norm(p=2).mean()

        total_loss = recon_loss + sign_loss + reg_loss
        if torch.isnan(total_loss):
            print("NaN loss detected in components:")
            print("Recon:", recon_loss.item(), "Sign:", sign_loss.item(), "Reg:", reg_loss.item())
            return torch.tensor(0.0, requires_grad=True)
            
        return total_loss
    

def load_snapshots(stock_data, window_size=20):
    snapshots = []
    already_done = set(fname.replace('.pkl', '') for fname in os.listdir(snapshot_path) if fname.endswith('.pkl'))

    for i in tqdm(range(window_size-1, len(date_to_idx)), desc="Preparing snapshots"):
        current_date = all_dates[i]
        if current_date in already_done:
            with open(os.path.join(snapshot_path, f"{current_date}.pkl"), 'rb') as f:
                loaded_snapshot = pickle.load(f)
                print(loaded_snapshot)
                snapshots.append(loaded_snapshot)
            continue
    
    return snapshots

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


def main1_generate(model):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    training_results = []
    best_loss = float('inf')

    print("Start training (realistic temporal setup)...")
    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_losses = []

        for t in tqdm(range(1, len(snapshots)), desc=f"Epoch {epoch}/{num_epochs}"):
            # Gebruik t-1 en t (ipv t en t+1) want we kunnen alleen naar het verleden kijken
            prev = snapshots[t-1]  # gisteren
            curr = snapshots[t]    # vandaag
            
            # Converteer edges - vergelijk gisteren met vandaag
            A_pos_prev = edge_info_to_tensor(prev['pos_edges_info']).to(device)
            A_neg_prev = edge_info_to_tensor(prev['neg_edges_info']).to(device)
            A_pos_curr = edge_info_to_tensor(curr['pos_edges_info']).to(device)
            A_neg_curr = edge_info_to_tensor(curr['neg_edges_info']).to(device)

            features = curr['features'].float().to(device)
            
            # Forward pass met temporeel correcte delta's
            optimizer.zero_grad()
            embeddings = model(
                features,  # H_t met window context
                A_pos_prev,  # Pos edges gisteren
                A_pos_curr,  # Pos edges vandaag (voor ΔA = vandaag - gisteren)
                A_neg_prev,  # Neg edges gisteren
                A_neg_curr   # Neg edges vandaag (voor ΔA)
            )
            
            # Loss berekenen op huidige edges
            loss = model.full_loss(
                embeddings,
                A_pos_curr,  # Vergelijk met huidige pos edges
                A_neg_curr   # Vergelijk met huidige neg edges
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_losses.append(loss.item())

        avg_loss = np.mean(epoch_losses)
        training_results.append(avg_loss)
        print(f"Epoch {epoch}: Avg Loss = {avg_loss:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(relation_path, "best_model.pth"))

    results_df = pd.DataFrame({'epoch': range(1, len(training_results)+1), 'loss': training_results})
    results_df.to_csv(os.path.join(relation_path, "training_results.csv"), index=False)

            
def main1_load(model):
    print("Laad beste model voor nieuwe edge-generatie...")

    model.load_state_dict(torch.load(os.path.join(relation_path, "best_model.pth"), map_location=device))
    model.eval()

    for date in tqdm(all_dates[prev_date_num - 1:], desc="Predicting full graph from embeddings"):
        snapshot_file = os.path.join(snapshot_path, f"{date}.pkl")
        if not os.path.exists(snapshot_file):
            print(f"Snapshot ontbreekt: {snapshot_file}")
            continue

        with open(snapshot_file, 'rb') as f:
            snapshot = pickle.load(f)

        with torch.no_grad():
            N = len(snapshot['tickers'])
            features = snapshot['features'].float().to(device)
            pos_edges_tensor = torch.empty((2, 0), dtype=torch.long, device=device)  # dummy
            neg_edges_tensor = torch.empty((2, 0), dtype=torch.long, device=device)  # dummy

            # Enkel features gebruiken om embedding te maken via model
            embeddings = model(features, pos_edges_tensor, neg_edges_tensor)

            candidate_edges = torch.combinations(torch.arange(N), r=2).T.to(device)
            scores = model.predict_link(embeddings, candidate_edges)

            pos_mask = scores > score_pos_threshold
            neg_mask = scores < score_neg_threshold

            pos_edges = candidate_edges[:, pos_mask]
            neg_edges = candidate_edges[:, neg_mask]

            pos_adj = edges_to_adj_matrix(pos_edges, N).to(device)
            neg_adj = edges_to_adj_matrix(neg_edges, N).to(device)

        # Gebruik jouw opslagstructuur voor downstream
        end_date = snapshot['date']
        end_idx = date_to_idx[end_date]
        start_idx = end_idx - prev_date_num + 1
        if start_idx < 0:
            print(f"Skipping {end_date} - onvoldoende geschiedenis")
            continue

        features_list, labels, stock_info = [], [], []
        window_data = snapshot['full_window_data']
        grouped = window_data.groupby('Stock')
        stock_groups = {name: group for name, group in grouped}

        for stock_name in snapshot['tickers']:
            stock_data = stock_groups.get(stock_name)
            if stock_data is not None and len(stock_data) == prev_date_num:
                features_list.append(stock_data[feature_cols2].values)
                raw_df = raw_data[stock_name]
                labels.append(calculate_label(raw_df, snapshot['date']))
                stock_info.append([stock_name, end_date])
            else:
                print(f"Data niet volledig voor {stock_name} op {end_date}")

        # Opslaan
        with open(os.path.join(data_train_predict_path, f"{end_date}.pkl"), 'wb') as f:
            pickle.dump({
                'pos_adj': pos_adj,
                'neg_adj': neg_adj,
                'features': torch.FloatTensor(np.array(features_list)),
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
snapshots = load_snapshots(stock_data)
model = DynamiSE(len(feature_cols2), hidden_dim).to(device)
main1_generate(model)
main1_load(model)