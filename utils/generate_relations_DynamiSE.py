import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torchdiffeq import odeint
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
from sklearn.neighbors import NearestNeighbors

class EfficientDynamicSE(nn.Module):
    def __init__(self, num_features, hidden_dim):
        super(EfficientDynamicSE, self).__init__()
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
    dates = sorted(list(set(stock_data['Date'])))
    snapshots = []
    
    for i in tqdm(range(window_size, len(dates))):
        window_dates = dates[i-window_size:i]
        window_data = stock_data[stock_data['Date'].isin(window_dates)]
        
        # Normalize features per stock
        features = window_data.groupby('Stock').apply(
            lambda x: x[['Open', 'High', 'Low', 'Close', 'Volume']].values
        )
        
        # For each stock, use mean features over window
        mean_features = features.apply(lambda x: x.mean(axis=0))
        feature_matrix = np.stack(mean_features.values)
        
        # Find approximate neighbors
        pos_pairs, neg_pairs = find_neighbors(feature_matrix, k=k_neighbors)
        
        if not pos_pairs or not neg_pairs:
            continue
            
        snapshots.append({
            'date': dates[i],
            'features': torch.FloatTensor(feature_matrix),
            'pos_edges': torch.LongTensor(pos_pairs).T,
            'neg_edges': torch.LongTensor(neg_pairs).T
        })
    
    return snapshots

# Usage example
if __name__ == "__main__":
    # 1. Load your stock data (assuming it's in a DataFrame)
    stock_data = pd.read_csv('your_stock_data.csv')
    
    # 2. Prepare dynamic graph snapshots
    snapshots = prepare_dynamic_data(stock_data)
    
    # 3. Initialize model
    model = EfficientDynamicSE(num_features=5, hidden_dim=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 4. Training loop
    for epoch in range(10):
        for snapshot in snapshots:
            optimizer.zero_grad()
            
            # Forward pass
            embeddings = model(
                snapshot['features'],
                snapshot['pos_edges'],
                snapshot['neg_edges'],
                torch.tensor([0.0, 1.0])  # Time steps
            )
            
            # Custom loss function for your task
            loss = embeddings.pow(2).mean()  # Placeholder - replace with your loss
            loss.backward()
            optimizer.step()
        
        print(f"Epoch {epoch}, Loss: {loss.item()}")



# import pandas as pd
# import os
# import torch
# import torch.nn as nn
# from torch_geometric.nn import GCNConv
# from torch_geometric.data import Data
# from torchdiffeq import odeint

# class DynamicSE(nn.Module):
#     def __init__(self, num_nodes, hidden_dim):
#         super(DynamicSE, self).__init__()
#         self.num_nodes = num_nodes
#         self.hidden_dim = hidden_dim
        
#         # Initial embedding layer
#         self.embedding = nn.Embedding(num_nodes, hidden_dim)
        
#         # Sign-aware GNN layers
#         self.pos_conv = GCNConv(hidden_dim, hidden_dim)
#         self.neg_conv = GCNConv(hidden_dim, hidden_dim)
        
#         # ODE function
#         self.ode_func = ODEFunc(hidden_dim)
        
#         # Prediction layer
#         self.predict = nn.Linear(hidden_dim * 2, 1)
        
#     def forward(self, data, t):
#         # Get initial embeddings
#         x = self.embedding(torch.arange(self.num_nodes))
        
#         # Separate positive and negative edges
#         pos_edge_index = data.edge_index[:, data.edge_attr > 0]
#         neg_edge_index = data.edge_index[:, data.edge_attr < 0]
        
#         # Message passing for positive and negative edges
#         x_pos = self.pos_conv(x, pos_edge_index)
#         x_neg = self.neg_conv(x, neg_edge_index)
        
#         # Combine sign information
#         x = torch.cat([x_pos, x_neg], dim=1)
        
#         # Solve ODE to model dynamics
#         x = odeint(self.ode_func, x, t, method='dopri5')
        
#         return x[-1]  # Return final embeddings

# class ODEFunc(nn.Module):
#     def __init__(self, hidden_dim):
#         super(ODEFunc, self).__init__()
#         self.net = nn.Sequential(
#             nn.Linear(hidden_dim * 2, hidden_dim * 2),
#             nn.Tanh(),
#             nn.Linear(hidden_dim * 2, hidden_dim * 2)
#         )
        
#     def forward(self, t, x):
#         return self.net(x)

# def create_signed_network(correlation_matrix, pos_threshold=0.5, neg_threshold=-0.3):
#     """
#     Convert correlation matrix to signed network with positive and negative edges
#     based on correlation thresholds.
#     """
#     n = correlation_matrix.shape[0]
#     edge_index = []
#     edge_attr = []
    
#     # Create edges for significant correlations
#     for i in range(n):
#         for j in range(i+1, n):
#             corr = correlation_matrix.iloc[i,j]
#             if corr > pos_threshold:
#                 edge_index.append([i, j])
#                 edge_attr.append(1)  # Positive edge
#                 edge_index.append([j, i])
#                 edge_attr.append(1)
#             elif corr < neg_threshold:
#                 edge_index.append([i, j])
#                 edge_attr.append(-1)  # Negative edge
#                 edge_index.append([j, i])
#                 edge_attr.append(-1)
    
#     if not edge_index:  # If no edges meet thresholds
#         return None
    
#     return Data(
#         edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
#         edge_attr=torch.tensor(edge_attr, dtype=torch.float)
#     )

# def process_correlation_matrices(relation_path, model_path="dynamicse_model.pth"):
#     # Load all correlation matrices
#     corr_files = [f for f in os.listdir(relation_path) if f.endswith('.csv')]
#     corr_files.sort()  # Sort by date
    
#     # Initialize model
#     num_nodes = len(pd.read_csv(os.path.join(relation_path, corr_files[0])['Unnamed: 0'])
#     model = DynamicSE(num_nodes, hidden_dim=64)
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
#     # Train on historical data
#     for i, file in enumerate(corr_files[:-1]):  # Use all but last for training
#         corr_matrix = pd.read_csv(os.path.join(relation_path, file), index_col=0)
#         data = create_signed_network(corr_matrix)
#         if data is None:
#             continue
            
#         # Add node features (identity matrix)
#         data.x = torch.eye(num_nodes)
        
#         # Train step
#         optimizer.zero_grad()
#         embeddings = model(data, torch.tensor([0.0, 1.0]))  # Time steps
#         loss = torch.mean(embeddings)  # Placeholder loss - replace with your task
#         loss.backward()
#         optimizer.step()
        
#         print(f"Processed {file}, Loss: {loss.item()}")
    
#     # Save model
#     torch.save(model.state_dict(), model_path)
#     return model

# # Main execution
# if __name__ == "__main__":
#     base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#     data_path = os.path.join(base_path, "data")
#     relation_path = os.path.join(data_path, "relation")
    
#     # Ensure relation directory exists
#     os.makedirs(relation_path, exist_ok=True)
    
#     # Process the correlation matrices with DynamicSE approach
#     model = process_correlation_matrices(relation_path)