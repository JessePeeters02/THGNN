# commit om 11u55
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
print('hallo')
# alle paden relatief aanmaken
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(base_path, "data", "testbatch1")
daily_data_path = os.path.join(data_path, "normaliseddailydata")
# kies hieronder de map waarin je de resultaten wilt opslaan
relation_path = os.path.join(data_path, "relation_dynamiSE_noknn2")
model_path = os.path.join(relation_path, "best_model.pth")
os.makedirs(relation_path, exist_ok=True)

# Hyperparameters
prev_date_num = 20
feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
hidden_dim = 64
num_epochs = 10

def load_all_stocks(stock_data_path):
    all_stock_data = []
    for file in tqdm(os.listdir(stock_data_path), desc="Loading data"):
        if file.endswith('.csv'):
            df = pd.read_csv(os.path.join(stock_data_path, file))
            all_stock_data.append(df[['Date', 'Stock', 'Open', 'High', 'Low', 'Close', 'Volume']])
    all_stock_data = pd.concat(all_stock_data, ignore_index=True)
    print(all_stock_data.head()) # kleine test om te zien of data deftig is ingeladen
    return all_stock_data

class DynamiSE(nn.Module):
    def __init__(self, num_features, hidden_dim):
        super(DynamiSE, self).__init__()
        self.hidden_dim = hidden_dim

        self.feature_encoder = nn.Linear(num_features, hidden_dim)

        # Positieve en negatieve convoluties
        self.pos_conv = GCNConv(hidden_dim, hidden_dim)
        self.neg_conv = GCNConv(hidden_dim, hidden_dim)

        # Combinatiefunctie Ψ
        self.psi = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.ode_func = ODEFunc(hidden_dim, self.pos_conv, self.neg_conv, self.psi)

        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, edge_index_pos, edge_index_neg, t, method='dopri5'):
        h = self.feature_encoder(x)
        self.ode_func.set_graph(edge_index_pos, edge_index_neg)
        h = odeint(self.ode_func, h, t, method=method)[1]
        return h

    def predict_edge_weight(self, h, edge_index, combine_method='hadamard'):
        src, dst = edge_index
        h_src, h_dst = h[src], h[dst]
        
        if combine_method == 'hadamard':
            h_pair = h_src * h_dst
        elif combine_method == 'concat':
            h_pair = torch.cat([h_src, h_dst], dim=1)
        elif combine_method == 'average':
            h_pair = (h_src + h_dst) / 2
        elif combine_method == 'subtract':
            h_pair = h_src - h_dst
        else:
            raise ValueError("Ongeldige combinatiemethode")
        
        return self.predictor(h_pair).squeeze()

    def full_loss(self, h, pos_edges, neg_edges, alpha=1.0, beta=0.001):
        # Reconstructieverlies (RMSE)
        loss_pos = self.edge_loss(h, pos_edges, +1)
        loss_neg = self.edge_loss(h, neg_edges, -1)
        recon_loss = loss_pos + loss_neg
        
        # Teken-constraint (paper Eq.7)
        w_hat_pos = self.predict_edge_weight(h, pos_edges)
        w_hat_neg = self.predict_edge_weight(h, neg_edges)
        sign_loss = -alpha * (
            torch.log(1 + w_hat_pos).mean() + 
            torch.log(1 - w_hat_neg).mean()
        )
        
        # Regularisatie
        reg_loss = beta * h.norm(p=2).mean()
        
        return recon_loss + sign_loss + reg_loss

    def edge_loss(self, h, edge_index, sign):
        w_hat = self.predict_edge_weight(h, edge_index)
        w_true = torch.full_like(w_hat, sign, dtype=torch.float32)
        recon = (w_hat - w_true).pow(2)
        log_term = torch.log1p(torch.exp(-w_true * w_hat))
        return recon.mean() + log_term.mean()

class ODEFunc(nn.Module):
    def __init__(self, hidden_dim, pos_conv, neg_conv, psi):
        super(ODEFunc, self).__init__()
        self.pos_conv = pos_conv
        self.neg_conv = neg_conv
        self.psi = psi
        self.edge_index_pos = None
        self.edge_index_neg = None
        self.psi_pos = nn.Linear(hidden_dim, hidden_dim)
        self.psi_neg = nn.Linear(hidden_dim, hidden_dim)

    def set_graph(self, edge_index_pos, edge_index_neg):
        self.edge_index_pos = edge_index_pos
        self.edge_index_neg = edge_index_neg

    def forward(self, t, h):
        # Pos/Neg-specifieke propagatie (paper Eq.3-4)
        h_pos = self.pos_conv(h, self.edge_index_pos)
        h_neg = self.neg_conv(h, self.edge_index_neg)
        
        # Lineaire combinatie (paper Eq.5)
        delta_h = self.psi_pos(h_pos) + self.psi_neg(h_neg)  # Aparte lineaire lagen
        return delta_h

# ΔA_t edgeverschillen berekenen

def compute_delta_edges(
    current_edges: torch.LongTensor,
    previous_edges: torch.LongTensor
) -> torch.LongTensor:
    
    curr = set(map(tuple, current_edges.T.tolist()))
    if len(previous_edges) > 0:
        prev = set(map(tuple, previous_edges.T.tolist()))
    else:
        prev = set()
        print("Geen vorige edges gevonden, dus geen delta.")    
    delta = curr - prev
    if delta:
        return torch.LongTensor(list(zip(*list(delta))))
    else:
        return torch.empty((2, 0), dtype=torch.long)

def build_initial_edges_via_correlation(window_data, threshold=0.6):
    # Groepeer per stock en bereken correlaties per feature
    grouped = window_data.groupby('Stock')[feature_cols]
    
    # Maak een 3D array van features (stocks x features x time)
    stock_arrays = np.stack([group.values.T for name, group in grouped])
    n_stocks = stock_arrays.shape[0]
    print(f"aantal stocks: {n_stocks}")
    # Initialiseer correlatiematrix
    corr_matrix = np.zeros((n_stocks, n_stocks))
    print(f"corr_matrix shape: {corr_matrix.shape}")
    # Bereken correlaties tussen alle paren van stocks
    for i in tqdm(range(n_stocks), desc="Calculating correlations"):
        for j in range(i+1, n_stocks):
            # Bereken correlatie voor elke feature apart
            feature_correlations = []
            for f in range(len(feature_cols)):
                corr = np.corrcoef(stock_arrays[i,f,:], stock_arrays[j,f,:])[0,1]
                feature_correlations.append(corr)
            
            # Gemiddelde correlatie over alle features
            avg_corr = np.nanmean(feature_correlations)
            corr_matrix[i,j] = avg_corr
            corr_matrix[j,i] = avg_corr
    
    # Bouw edges op basis van drempelwaarde
    pos_edges = []
    neg_edges = []
    
    for i in range(n_stocks):
        for j in range(i+1, n_stocks):
            if corr_matrix[i,j] > threshold:
                pos_edges.append((i, j))
                pos_edges.append((j, i))  # Maak ongericht
            elif corr_matrix[i,j] < -threshold:
                neg_edges.append((i, j))
                neg_edges.append((j, i))  # Maak ongericht
    print(f"Pos edges: {len(pos_edges)/2}, Neg edges: {len(neg_edges)/2}")

    # Converteer naar torch Tensors
    pos_edges = torch.LongTensor(list(zip(*pos_edges))) if pos_edges else torch.empty((2, 0), dtype=torch.long)
    neg_edges = torch.LongTensor(list(zip(*neg_edges))) if neg_edges else torch.empty((2, 0), dtype=torch.long)

    return pos_edges, neg_edges

def build_edges_via_balance_theory(prev_pos_edges, prev_neg_edges, num_nodes):
    # Debug: Print input edges
    print(f"\nInput pos edges: {prev_pos_edges.shape}, neg edges: {prev_neg_edges.shape}")
    
    adj_pos = defaultdict(set)
    adj_neg = defaultdict(set)

    # Bouw adjacency lists
    if prev_pos_edges.numel() > 0:
        for src, dst in prev_pos_edges.T.tolist():
            adj_pos[src].add(dst)
            adj_pos[dst].add(src)  # Maak ongericht
    if prev_neg_edges.numel() > 0:
        for src, dst in prev_neg_edges.T.tolist():
            adj_neg[src].add(dst)
            adj_neg[dst].add(src)  # Maak ongericht

    # Debug: Tel nodes met neighbors
    nodes_with_neighbors = sum(1 for j in range(num_nodes) if adj_pos[j] or adj_neg[j])
    print(f"Nodes with neighbors: {nodes_with_neighbors}/{num_nodes}")

    pos_edges_set = set()
    neg_edges_set = set()
    triangle_count = 0

    for j in tqdm(range(num_nodes), desc="triangles"):
        neighbors = set(adj_pos[j]) | set(adj_neg[j])
        for i, k in itertools.combinations(neighbors, 2):
            if i == k:
                continue
                
            # Debug: Tel triadische checks
            triangle_count += 1
            
            # Check bestaande edges
            if (k in adj_pos[i]) or (k in adj_neg[i]):
                continue

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
                # Balance theory toepassen
                neg_count = signs.count('-')
                if neg_count % 2 == 0:
                    pos_edges_set.add((i, k))
                    pos_edges_set.add((k, i))  # Ongericht
                else:
                    neg_edges_set.add((i, k))
                    neg_edges_set.add((k, i))  # Ongericht

    # Debug prints
    print(f"Triangles checked: {triangle_count}")
    print(f"New pos edges: {len(pos_edges_set)//2}, New neg edges: {len(neg_edges_set)//2}")

    # Converteer naar tensors
    pos_edges = torch.LongTensor(list(zip(*pos_edges_set))) if pos_edges_set else torch.empty((2, 0))
    neg_edges = torch.LongTensor(list(zip(*neg_edges_set))) if neg_edges_set else torch.empty((2, 0))
    
    return pos_edges, neg_edges

def prepare_dynamic_data(stock_data, window_size=20):
    """Prepare time-evolving graph data without full correlation matrices"""
    dates = sorted(stock_data['Date'].unique())
    unique_stocks = sorted(stock_data['Stock'].unique())
    snapshots = []
    bool_eerste = True
    snapshot_path = os.path.join(data_path, "intermediate_snapshots")
    os.makedirs(snapshot_path, exist_ok=True)
    already_done = set(fname.replace('.pkl', '') for fname in os.listdir(snapshot_path) if fname.endswith('.pkl'))

    for i in tqdm(range(window_size, len(dates)), desc="Preparing snapshots"):
        if dates[i] in already_done:
            print(f"Snapshot {dates[i]} bestaat al, overslaan...")
            with open(os.path.join(snapshot_path, f"{dates[i]}.pkl"), 'rb') as f:
                snapshots.append(pickle.load(f))
            continue

        window_dates = dates[i-window_size:i]
        window_data = stock_data[stock_data['Date'].isin(window_dates)]
        
        # Normalize features per stock
        print('1')
        # Find approximate neighbors
        if bool_eerste:
            pos_pairs, neg_pairs = build_initial_edges_via_correlation(window_data, threshold=0.6)
            bool_eerste = False
            feature_matrix = (
                window_data.groupby('Stock')[feature_cols]
                .apply(lambda df: df.values.mean(axis=0))
                .reindex(sorted(window_data['Stock'].unique()))
                .values
            )
            # Voeg toe: sla initiële edges op als vorige edges voor volgende snapshot
            snapshots.append({
                'date': dates[i],
                'features': torch.FloatTensor(feature_matrix),
                'pos_edges': pos_pairs,
                'neg_edges': neg_pairs,
                'tickers': unique_stocks
            })
            continue
        else:
            prev_snapshot = snapshots[-1]
            pos_pairs, neg_pairs = build_edges_via_balance_theory(
                prev_snapshot['pos_edges'], prev_snapshot['neg_edges'], len(unique_stocks)
            )
            prev_edge_count = (
                prev_snapshot['pos_edges'].shape[1] +
                prev_snapshot['neg_edges'].shape[1]
            ) 
            new_edge_count = pos_pairs.shape[1] + neg_pairs.shape[1]
            growth_ratio = new_edge_count / max(prev_edge_count, 1) 
            if growth_ratio > 1.2:
                print(f"[Te veel groei] snapshot {dates[i]} overslaan (ratio={growth_ratio:.2f})")
                pos_pairs, neg_pairs = build_initial_edges_via_correlation(window_data, threshold=0.6)

                feature_matrix = (
                    window_data.groupby('Stock')[feature_cols]
                    .apply(lambda df: df.values.mean(axis=0))
                    .reindex(sorted(window_data['Stock'].unique()))
                    .values
                )     
            # Check: levert balance theory iets op?
            new_edges_found = pos_pairs.size(1) + neg_pairs.size(1)

            if new_edges_found == 0:
                print(f"[Fallback] Geen nieuwe edges gevonden op dag {dates[i]} — fallback naar correlatie.")
                pos_pairs, neg_pairs = build_initial_edges_via_correlation(window_data, threshold=0.6)
                feature_matrix = (
                    window_data.groupby('Stock')[feature_cols]
                    .apply(lambda df: df.values.mean(axis=0))
                    .reindex(sorted(window_data['Stock'].unique()))
                    .values
                )
            else:
                print(f"Balance theory gevonden: {pos_pairs.size(1)} pos, {neg_pairs.size(1)} neg edges.")
        print('2')
        # Bereken ΔA_t voor de huidige snapshot (verandering in de graaf)
        prev_pos_edges = prev_snapshot['pos_edges'] if snapshots else []
        prev_neg_edges = prev_snapshot['neg_edges'] if snapshots else []
        # delta_pos = compute_delta_edges(pos_pairs, prev_pos_edges)
        # delta_neg = compute_delta_edges(neg_pairs, prev_neg_edges)
        print('3')
        # Zet de delta's om naar edge_index tensors
        edge_index_pos = pos_pairs
        edge_index_neg = neg_pairs

        feature_matrix = (
            window_data.groupby('Stock')[feature_cols]
            .apply(lambda df: df.values.mean(axis=0))
            .reindex(sorted(window_data['Stock'].unique()))
            .values
        )
        # Voeg de snapshot toe met dynamische veranderingen in de graaf
        snapshots.append({
            'date': dates[i],
            'features': torch.FloatTensor(feature_matrix),
            'pos_edges': edge_index_pos,
            'neg_edges': edge_index_neg,
            'tickers': unique_stocks
        })
        # On-the-fly snapshot opslaan
        with open(os.path.join(snapshot_path, f"{dates[i]}.pkl"), 'wb') as f:
            pickle.dump(snapshots[-1], f)
        # Append snapshot metadata naar CSV
        log_path = os.path.join(data_path, "snapshot_log.csv")
        write_header = not os.path.exists(log_path) or os.path.getsize(log_path) == 0
        with open(log_path, "a") as log_f:
            if write_header:
                log_f.write("date,nodes,pos_edges,neg_edges\n")
            line = f"{dates[i]},{len(unique_stocks)},{edge_index_pos.shape[1]},{edge_index_neg.shape[1]}\n"
            log_f.write(line)
        print(f"Vorige pos_edges shape: {prev_pos_edges.shape if isinstance(prev_pos_edges, torch.Tensor) else 'Geen'}")
        print(f"Vorige neg_edges shape: {prev_neg_edges.shape if isinstance(prev_neg_edges, torch.Tensor) else 'Geen'}")
    
    return snapshots

def edge_prediction_loss(h, edge_index, sign, model):
    """Loss op basis van voorspelde ŵ_{ij} tegenover ground truth sign."""
    w_hat = model.predict_edge_weight(h, edge_index)
    w_true = torch.full_like(w_hat, sign, dtype=torch.float32)
    return ((w_hat - w_true) ** 2).mean()

def edges_to_adj_matrix(edges, num_nodes):
    """Converteer edges naar adjacency matrix"""
    adj = torch.zeros((num_nodes, num_nodes))
    if edges.size(1) > 0:
        adj[edges[0], edges[1]] = 1.0
    return adj

# def evaluate(model, snapshots):
#     model.eval()
#     rmse, mae = 0.0, 0.0
#     for snapshot in snapshots:
#         with torch.no_grad():
#             h = model(...)
#             w_hat = model.predict_edge_weight(h, snapshot['edges'])
#             w_true = get_ground_truth(snapshot)  # Implementeer dit
#             rmse += torch.sqrt(((w_hat - w_true)**2).mean()).item()
#             mae += torch.abs(w_hat - w_true).mean().item()
#     print(f"RMSE: {rmse/len(snapshots):.4f}, MAE: {mae/len(snapshots):.4f}")

# Main
# if __name__ == "__main__":
def main1_generate():
    # 1. prepare data
    stock_data = load_all_stocks(daily_data_path)
    snapshots = prepare_dynamic_data(stock_data)
    
    # test prints
    print(f"Aantal snapshots: {len(snapshots)}")
    print(f"Gemiddelde nodes per snapshot: {np.mean([s['features'].shape[0] for s in snapshots]):.0f}")
    print(f"Gemiddelde edges per snapshot: {np.mean([s['pos_edges'].shape[1] + s['neg_edges'].shape[1] for s in snapshots]):.0f}")

    # 2. Initialize model
    model = DynamiSE(num_features=len(feature_cols), hidden_dim=hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop met opslag van model en resultaten
    best_loss = float('inf')
    training_results = []

    # 3. Training loop
    for epoch in range(num_epochs):
        model.train()
        epoch_losses = []

        for snapshot in tqdm(snapshots, desc=f"Epoch {epoch+1} van de {num_epochs}"):
            optimizer.zero_grad()
            
            # Forward pass
            embeddings = model(
                snapshot['features'],
                snapshot['pos_edges'],
                snapshot['neg_edges'],
                torch.tensor([0.0, 1.0])  # Time steps
            )
            loss = model.full_loss(embeddings, snapshot['pos_edges'], snapshot['neg_edges'])

            # #DyanmiSE loss function 
            # loss = dynamiSE_loss(
            #     embeddings,
            #     snapshot['pos_edges'],
            #     snapshot['neg_edges'],
            #     alpha=1.0,
            #     beta=0.001
            # )
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
            weights =  np.arange(1, len(epoch_losses)+1)
            weighted_avg_loss = np.average(epoch_losses, weights=weights)

        # Bereken gemiddeld verlies voor deze epoch
        avg_loss = weighted_avg_loss  # np.mean(epoch_losses)
        print(f"Epoch {epoch+1}, Avg Loss: {avg_loss}")
        training_results.append(avg_loss)

        # Sla het beste model op
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(relation_path, "best_model.pth"))
            print(f"Beste model opgeslagen in {model_path} met loss {best_loss}")

    # Sla de trainingsresultaten op
    results_df = pd.DataFrame({
        'epoch': range(1, len(training_results) + 1),
        'loss': training_results
    })
    results_path = os.path.join(relation_path, "training_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"Trainingsresultaten opgeslagen in {results_path}")

def main1_load():

    stock_data = load_all_stocks(daily_data_path)
    snapshots = prepare_dynamic_data(stock_data)
    # 4. Resultaatgeneratie
    model = DynamiSE(num_features=len(feature_cols), hidden_dim=hidden_dim)
    model.load_state_dict(torch.load(os.path.join(relation_path, "best_model.pth")))
    model.eval()

    # voor efficientie later
    stock_data = stock_data.sort_values(['Stock', 'Date'])  # Sorteer eenmalig
    stock_groups = stock_data.groupby('Stock')  # Maak snelle lookup groups

    for snapshot in tqdm(snapshots, desc="Generating outputs"):
        with torch.no_grad():
            # Adjacency matrices
            N = len(snapshot['tickers'])
            pos_adj = edges_to_adj_matrix(snapshot['pos_edges'], N)
            neg_adj = edges_to_adj_matrix(snapshot['neg_edges'], N)
            
            # Features en labels
            features, labels, stock_info = [], [], []
            for stock_name in snapshot['tickers']:
                window_data = stock_groups.get_group(stock_name)
                window_data = window_data[window_data['Date'] <= snapshot['date']].tail(prev_date_num)
                if len(window_data) == prev_date_num:
                    features.append(window_data[feature_cols].values)
                    labels.append(window_data.iloc[-1]['Close'])
                    stock_info.append([stock_name, snapshot['date']])
            
            # Opslag
            os.makedirs(os.path.join(data_path, "data_train_predict_DSE_noknn1"), exist_ok=True)
            with open(os.path.join(data_path, "data_train_predict_DSE_noknn1", f"{snapshot['date']}.pkl"), 'wb') as f:
                pickle.dump({
                    'pos_adj': pos_adj,
                    'neg_adj': neg_adj,
                    'features': torch.FloatTensor(np.array(features)),
                    'labels': torch.FloatTensor(labels),
                    'mask': [True] * len(labels)
                }, f)
            
            os.makedirs(os.path.join(data_path, "daily_stock_DSE_noknn1"), exist_ok=True)
            pd.DataFrame(stock_info, columns=['code', 'dt']).to_csv(
                os.path.join(data_path, "daily_stock_DSE_noknn1", f"{snapshot['date']}.csv"), index=False)
            

main1_generate()
main1_load()