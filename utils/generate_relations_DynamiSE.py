#Laatste commit: 
#j generate_relation.py herwerkt voor het gebruiken van DynamiSE
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torchdiffeq import odeint
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
import os
from sklearn.neighbors import NearestNeighbors

# alle paden relatief aanmaken
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(base_path, "data")
daily_data_path = os.path.join(data_path, "NASDAQ_per_dag")
# kies hieronder de map waarin je de resultaten wilt opslaan
relation_path = os.path.join(data_path, "relation_dynamiSE1")
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
        # Feature transformation
        self.feature_encoder = nn.Linear(num_features, hidden_dim)
        # GNN layers
        self.pos_conv = GCNConv(hidden_dim, hidden_dim)
        self.neg_conv = GCNConv(hidden_dim, hidden_dim)
        # ODE function
        self.ode_func = ODEFunc(hidden_dim)
        
    def forward(self, x, pos_edge_index, neg_edge_index, t):
        # Encode features
        h = self.feature_encoder(x)
        # Message passing
        h_pos = self.pos_conv(h, pos_edge_index)
        h_neg = self.neg_conv(h, neg_edge_index)
        h = torch.cat([h_pos, h_neg], dim=1)
        # Temporal dynamics
        h = odeint(self.ode_func, h, t, method='dopri5')[1]
        return h

class ODEFunc(nn.Module):
    def __init__(self, hidden_dim):
        super(ODEFunc, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.Tanh()
        )
        
    def forward(self, t, x):
        return self.net(x)

def find_neighbors(features, k=5, metric='correlation'):
    """Find k-nearest and k-farthest neighbors using correlation"""
    nbrs = NearestNeighbors(n_neighbors=k+1, metric=metric).fit(features)
    distances, indices = nbrs.kneighbors(features)
    
    # Positive neighbors (most similar)
    pos_pairs = []
    for i in range(len(indices)):
        for j in indices[i][1:]:  # Skip self
            pos_pairs.append([i, j])
    
    # Negative neighbors (most dissimilar)
    neg_pairs = []
    for i in range(len(features)):
        # Get indices sorted by correlation in ascending order
        anti_indices = np.argsort(distances[i])[::-1][:k]
        for j in anti_indices:
            if j != i:  # Skip self
                neg_pairs.append([i, j])
    
    return pos_pairs, neg_pairs

def prepare_dynamic_data(stock_data, window_size=20, k_neighbors=5):
    """Prepare time-evolving graph data without full correlation matrices"""
    dates = sorted(stock_data['Date'].unique())
    unique_stocks = sorted(stock_data['Stock'].unique())
    snapshots = []
    
    for i in tqdm(range(window_size, len(dates)), desc="Preparing snapshots"):
        window_dates = dates[i-window_size:i]
        window_data = stock_data[stock_data['Date'].isin(window_dates)]
        
        # Normalize features per stock
        feature_matrix = window_data.groupby('Stock')[feature_cols].mean().values
        
        # Find approximate neighbors
        pos_pairs, neg_pairs = find_neighbors(feature_matrix, k=k_neighbors)
        
        if not pos_pairs or not neg_pairs:
            print('huh, no pos or neg pairs found?')
            continue
            
        snapshots.append({
            'date': dates[i],
            'features': torch.FloatTensor(feature_matrix),
            'pos_edges': torch.LongTensor(pos_pairs).T,
            'neg_edges': torch.LongTensor(neg_pairs).T,
            'tickers': unique_stocks
        })
    
    return snapshots

def dynamiSE_loss(embeddings, pos_edges, neg_edges, alpha=1.0, beta=0.001):
    """
    embeddings: Tensor [N, D]
    pos_edges: Tensor [2, E_pos] (source, target)
    neg_edges: Tensor [2, E_neg]
    """
    def edge_loss(edge_index, sign):
        src, dst = edge_index
        emb_src = embeddings[src]
        emb_dst = embeddings[dst]

        emb_src = torch.nn.functional.normalize(emb_src, p=2, dim=1)
        emb_dst = torch.nn.functional.normalize(emb_dst, p=2, dim=1)

        # Predicted edge weight: dot product
        w_hat = (emb_src * emb_dst).sum(dim=1)  # shape [E]
        w_true = torch.full_like(w_hat, sign, dtype=torch.float32)

        # Term 1: squared error
        recon_loss = (w_hat - w_true).pow(2)

        # Term 2: sign consistency
        log_term = torch.log1p((w_hat * w_true).clamp(min=1e-6))  # log(1 + x), safe clamp

        return recon_loss - alpha * log_term

    loss_pos = edge_loss(pos_edges, sign=+1).mean()
    loss_neg = edge_loss(neg_edges, sign=-1).mean()

    # Regularization on embeddings
    l_reg = embeddings.norm(p=2).mean()

    return loss_pos + loss_neg + beta * l_reg

def edges_to_adj_matrix(edges, num_nodes):
    """Converteer edges naar adjacency matrix"""
    adj = torch.zeros((num_nodes, num_nodes))
    if edges.size(1) > 0:
        adj[edges[0], edges[1]] = 1.0
    return adj

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
            #DyanmiSE loss function 
            loss = dynamiSE_loss(
                embeddings,
                snapshot['pos_edges'],
                snapshot['neg_edges'],
                alpha=1.0,
                beta=0.001
            )
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
            os.makedirs(os.path.join(data_path, "data_train_predict_DSE1"), exist_ok=True)
            with open(os.path.join(data_path, "data_train_predict_DSE1", f"{snapshot['date']}.pkl"), 'wb') as f:
                pickle.dump({
                    'pos_adj': pos_adj,
                    'neg_adj': neg_adj,
                    'features': torch.FloatTensor(np.array(features)),
                    'labels': torch.FloatTensor(labels),
                    'mask': [True] * len(labels)
                }, f)
            
            os.makedirs(os.path.join(data_path, "daily_stock_DSE1"), exist_ok=True)
            pd.DataFrame(stock_info, columns=['code', 'dt']).to_csv(
                os.path.join(data_path, "daily_stock_DSE1", f"{snapshot['date']}.csv"), index=False)
            


def main2_generate():
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
            #DyanmiSE loss function 
            loss = dynamiSE_loss(
                embeddings,
                snapshot['pos_edges'],
                snapshot['neg_edges'],
                alpha=1.0,
                beta=0.001
            )
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        # Bereken gemiddeld verlies voor deze epoch
        avg_loss = np.mean(epoch_losses)
        training_results.append(avg_loss)
        print(f"Epoch {epoch+1}, Avg Loss: {avg_loss}")
        
        # Sla het beste model op
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(relation_path, "best_model.pth"))
            print(f"Beste model opgeslagen in {model_path} met loss {best_loss}")

    # Sla de trainingsresultaten op
    results_df = pd.DataFrame({
        'epoch': range(1, len(training_results)+1),
        'loss': training_results
    })
    results_path = os.path.join(relation_path, "training_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"Trainingsresultaten opgeslagen in {results_path}")

def main2_load():

    stock_data = load_all_stocks(daily_data_path)
    snapshots = prepare_dynamic_data(stock_data)
    # 4. Resultaatgeneratie
    model = DynamiSE(num_features=len(feature_cols), hidden_dim=hidden_dim)
    model.load_state_dict(torch.load(os.path.join(relation_path, "best_model.pth")))
    model.eval()
    stock_data = stock_data.sort_values(['Stock', 'Date']) 
    stock_groups = stock_data.groupby('Stock')
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
            date_str = snapshot['date']
            os.makedirs(os.path.join(data_path, "data_train_predict_DSE2"), exist_ok=True)
            with open(os.path.join(data_path, "data_train_predict_DSE2", f"{date_str}.pkl"), 'wb') as f:
                pickle.dump({
                    'pos_adj': pos_adj,
                    'neg_adj': neg_adj,
                    'features': torch.FloatTensor(np.array(features)),
                    'labels': torch.FloatTensor(labels),
                    'mask': [True] * len(labels)
                }, f)
            
            os.makedirs(os.path.join(data_path, "daily_stock_DSE2"), exist_ok=True)
            pd.DataFrame(stock_info, columns=['code', 'dt']).to_csv(
                os.path.join(data_path, "daily_stock_DSE2", f"{date_str}.csv"), index=False)
            


# main1_generate() # heeft gewerkt
# main1_load()   

relation_path = os.path.join(data_path, "relation_dynamiSE2")
model_path = os.path.join(relation_path, "best_model.pth")
os.makedirs(relation_path, exist_ok=True)

# main2_generate()
main2_load()