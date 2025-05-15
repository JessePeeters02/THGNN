import os
import torch
import pickle
import numpy as np
from tqdm import tqdm
import networkx as nx
import pandas as pd
from torch.autograd import Variable

# Definieer de kolommen die we willen gebruiken uit de CSV-bestanden
# feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Turnover']
prev_date_num = 20

threshold = 0.4
min_neighbors = 3

# Basis pad naar de data-map
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Huidige scriptmap
print(base_path)
data_path = os.path.join(base_path, "data", "S&P500")
print(data_path)
relation_path = os.path.join(data_path, "correlations")
print(relation_path)
raw_data_path = os.path.join(data_path, "stockdata")
print(raw_data_path)
stock_data_path = os.path.join(data_path, "normalisedstockdata")  # Map waar de CSV-bestanden staan
print(stock_data_path)


# Functie om de CSV-bestanden in te lezen en om te zetten naar een DataFrame
def load_stock_data(raw_stock_path, stock_data_path):

    raw_files = [f for f in os.listdir(raw_stock_path) if f.endswith('.csv')]
    raw_data = {}
    for file in tqdm(raw_files, desc="Loading raw data"):
        stock_name = file.split('.')[0]
        df = pd.read_csv(os.path.join(raw_stock_path, file), parse_dates=['Date'])
        raw_data[stock_name] = df

    stock_files = [f for f in os.listdir(stock_data_path) if f.endswith('.csv')]
    stock_data = {}
    for stock_file in tqdm(stock_files, desc="Loading stock data"):
        stock_name = stock_file.split('.')[0]
        df = pd.read_csv(os.path.join(stock_data_path, stock_file))
        df['Date'] = pd.to_datetime(df['Date'])
        stock_data[stock_name] = df
    
    return raw_data, stock_data

def calculate_label(raw_df, current_date):
    date_idx = raw_df[raw_df['Date'] == current_date].index[0]
    # print(date_idx)
    close_today = raw_df.iloc[date_idx]['Close']
    close_yesterday = raw_df.iloc[date_idx-1]['Close']
    return (close_today / close_yesterday) - 1

# Laad de stock data
raw_data, stock_data = load_stock_data(raw_data_path, stock_data_path)
# print(stock_data)

# Bepaal de unieke datums uit de data
all_dates = sorted({date.strftime('%Y-%m-%d') for df in stock_data.values() for date in df['Date'].tolist()})
# print(f"Unique dates determined: {len(all_dates)}")
# print(all_dates)

# def prepare_adjacencymatrix(enddt):
#     relation_file = os.path.join(relation_path, f"{enddt}.csv")
#     adj_all = pd.read_csv(relation_file, index_col=0)

#     pos_adj = nx.adjacency_matrix(nx.Graph(adj_all > threshold)).toarray().astype(float)
#     pos_adj = torch.FloatTensor(pos_adj - np.diag(np.diag(pos_adj)))

#     neg_adj = nx.adjacency_matrix(nx.Graph(adj_all < -threshold)).toarray().astype(float)
#     neg_adj = torch.FloatTensor(neg_adj - np.diag(np.diag(neg_adj)))
    
#     # Tel het aantal verbindingen (1'en) in beide matrices
#     pos_connections = int(torch.sum(pos_adj).item())
#     neg_connections = int(torch.sum(neg_adj).item())
#     print(f"Aantal positieve verbindingen: {pos_connections}")
#     print(f"Aantal negatieve verbindingen: {neg_connections}")
    
#     # Bereken min/max aantal verbindingen per lijn
#     pos_connections_per_row = torch.sum(pos_adj, dim=1)
#     neg_connections_per_row = torch.sum(neg_adj, dim=1)
    
#     print(f"Min positieve verbindingen per lijn: {int(torch.min(pos_connections_per_row).item())}")
#     print(f"Max positieve verbindingen per lijn: {int(torch.max(pos_connections_per_row).item())}")
#     print(f"Min negatieve verbindingen per lijn: {int(torch.min(neg_connections_per_row).item())}")  
#     print(f"Max negatieve verbindingen per lijn: {int(torch.max(neg_connections_per_row).item())}")
    
#     # print('pos_adj shape: ', pos_adj.shape)
#     # print('neg_adj shape: ', neg_adj.shape)

#     return pos_adj, neg_adj

def prepare_adjacencymatrix(enddt, threshold, min_neighbors):
    relation_file = os.path.join(relation_path, f"{enddt}.csv")
    adj_all = pd.read_csv(relation_file, index_col=0)
    stock_names = adj_all.index.tolist()
    adj_values = adj_all.values
    
    # Initialiseer matrices met vaste threshold
    pos_adj = (adj_values > threshold).astype(float)
    neg_adj = (adj_values < -threshold).astype(float)
    
    # Verwijder zelf-connecties
    np.fill_diagonal(pos_adj, 0)
    np.fill_diagonal(neg_adj, 0)
    
    # Garandeer minimum aantal buren
    for i in range(len(stock_names)):
        # Positieve buren
        pos_neighbors = np.sum(pos_adj[i])
        if pos_neighbors < min_neighbors:
            corrs = adj_values[i].copy()
            corrs[i] = 0
            top_pos_indices = np.argsort(-corrs)[:min_neighbors]
            pos_adj[i, top_pos_indices] = 1
        
        # Negatieve buren
        neg_neighbors = np.sum(neg_adj[i])
        if neg_neighbors < min_neighbors:
            corrs = adj_values[i].copy()
            corrs[i] = 0
            top_neg_indices = np.argsort(corrs)[:min_neighbors]
            neg_adj[i, top_neg_indices] = 1
    
    return torch.FloatTensor(pos_adj), torch.FloatTensor(neg_adj)


# def prepare_adjacencymatrix(enddt, min_neighbors=min_neighbors):
#     relation_file = os.path.join(relation_path, f"{enddt}.csv")
#     adj_all = pd.read_csv(relation_file, index_col=0)
#     stock_names = adj_all.index.tolist()
#     adj_values = adj_all.values
    
#     # Initialiseer matrices met vaste threshold voor sterke connecties
#     original_pos_adj = (adj_values > threshold).astype(float)
#     original_neg_adj = (adj_values < -threshold).astype(float)
    
#     # Verwijder zelf-connecties
#     np.fill_diagonal(original_pos_adj, 0)
#     np.fill_diagonal(original_neg_adj, 0)
    
#     # Maak kopieën voor aanpassing
#     pos_adj = original_pos_adj.copy()
#     neg_adj = original_neg_adj.copy()
    
#     # Threshold logging
#     threshold_stats = {
#         'original_pos': [],
#         'added_pos': [],
#         'original_neg': [],
#         'added_neg': []
#     }
    
#     # Garandeer minimum aantal buren
#     for i in range(len(stock_names)):
#         # Positieve buren
#         pos_neighbors = np.sum(pos_adj[i])
#         if pos_neighbors < min_neighbors:
#             corrs = adj_values[i].copy()
#             corrs[i] = 0  # verwijder zelf-connectie
            
#             # Sorteer correlaties (hoog naar laag)
#             sorted_indices = np.argsort(-corrs)
#             sorted_corrs = corrs[sorted_indices]
            
#             # Threshold is de correlatie van de min_neighbors-de buur
#             new_threshold = sorted_corrs[min_neighbors-1] if len(sorted_corrs) >= min_neighbors else 0
            
#             # Log thresholds
#             original_threshold = threshold
#             threshold_stats['original_pos'].append(original_threshold)
#             threshold_stats['added_pos'].append(new_threshold)
            
#             # Selecteer nieuwe buren
#             needed = min_neighbors - pos_neighbors
#             added = 0
#             for idx in sorted_indices:
#                 if added >= needed:
#                     break
#                 if pos_adj[i, idx] == 0 and corrs[idx] > 0:  # Nog niet geselecteerd en positief
#                     pos_adj[i, idx] = 1
#                     added += 1
        
#         # Negatieve buren
#         neg_neighbors = np.sum(neg_adj[i])
#         if neg_neighbors < min_neighbors:
#             corrs = adj_values[i].copy()
#             corrs[i] = 0  # verwijder zelf-connectie
            
#             # Sorteer correlaties (laag naar hoog)
#             sorted_indices = np.argsort(corrs)
#             sorted_corrs = corrs[sorted_indices]
            
#             # Threshold is de correlatie van de min_neighbors-de buur
#             new_threshold = sorted_corrs[min_neighbors-1] if len(sorted_corrs) >= min_neighbors else 0
            
#             # Log thresholds
#             original_threshold = -threshold
#             threshold_stats['original_neg'].append(original_threshold)
#             threshold_stats['added_neg'].append(new_threshold)
            
#             # Selecteer nieuwe buren
#             needed = min_neighbors - neg_neighbors
#             added = 0
#             for idx in sorted_indices:
#                 if added >= needed:
#                     break
#                 if neg_adj[i, idx] == 0 and corrs[idx] < 0:  # Nog niet geselecteerd en negatief
#                     neg_adj[i, idx] = 1
#                     added += 1
    
#     # Print threshold statistieken
#     print("\nThreshold statistieken:")
#     print(f"Originele positieve threshold: {threshold}")
#     if threshold_stats['added_pos']:
#         print(f"Laagste toegevoegde positieve threshold: {min(threshold_stats['added_pos']):.4f}")
#         print(f"Gemiddelde toegevoegde positieve threshold: {np.mean(threshold_stats['added_pos']):.4f}")
    
#     print(f"\nOriginele negatieve threshold: {-threshold}")
#     if threshold_stats['added_neg']:
#         print(f"Hoogste toegevoegde negatieve threshold: {max(threshold_stats['added_neg']):.4f}")
#         print(f"Gemiddelde toegevoegde negatieve threshold: {np.mean(threshold_stats['added_neg']):.4f}")
    
#     # Converteer naar tensors
#     pos_adj = torch.FloatTensor(pos_adj)
#     neg_adj = torch.FloatTensor(neg_adj)
    
#     # Tel het aantal verbindingen
#     pos_connections = int(torch.sum(pos_adj).item())
#     neg_connections = int(torch.sum(neg_adj).item())
#     print(f"\nAantal positieve verbindingen: {pos_connections}")
#     print(f"Aantal negatieve verbindingen: {neg_connections}")
    
#     # Bereken min/max aantal verbindingen per lijn
#     pos_connections_per_row = torch.sum(pos_adj, dim=1)
#     neg_connections_per_row = torch.sum(neg_adj, dim=1)
    
#     print(f"\nMin positieve verbindingen per lijn: {int(torch.min(pos_connections_per_row).item())}")
#     print(f"Max positieve verbindingen per lijn: {int(torch.max(pos_connections_per_row).item())}")
#     print(f"Min negatieve verbindingen per lijn: {int(torch.min(neg_connections_per_row).item())}")  
#     print(f"Max negatieve verbindingen per lijn: {int(torch.max(neg_connections_per_row).item())}")
    
#     return pos_adj, neg_adj

# Functie om de relatiegrafieken te verwerken
def fun(iend, enddt, stock_data, pdn, tr, mn):
    istart = iend - pdn + 1
    startdt = all_dates[istart]
    # print(f"calculating window: {startdt} to {enddt}")
    
    pos_adj, neg_adj = prepare_adjacencymatrix(enddt, tr, mn)

    dts = all_dates[istart:iend+1]
    # print("Processing dates:", len(dts), dts)
        
    features = []
    labels = []
    day_last_code = []
        
    for stock_name, df in stock_data.items():
        df_window = df[(df['Date'] >= startdt) & (df['Date'] <= enddt)]
        # print(f"df window shape: {df_window.shape}")
        # print(f"df window: {df_window}")
        day_last_code.append([stock_name, enddt])
        raw_df = raw_data[stock_name]
        if len(df_window) == pdn:
            features.append(df_window[feature_cols].values)
            label = calculate_label(raw_df, end_date)
            labels.append(label)
        else:
            print(' huh, len df window pdn????')
            break
        
    output = {
        'pos_adj': torch.FloatTensor(pos_adj),
        'neg_adj': torch.FloatTensor(neg_adj),
        'features': torch.FloatTensor(np.array(features)),
        'labels': torch.FloatTensor(labels),
        'mask': [True] * len(labels)  # Alle samples zijn geldig
    }

    with open(os.path.join(data_train_predict_path, f"{enddt}.pkl"), 'wb') as f:
        pickle.dump(output, f)
    df = pd.DataFrame(columns=['code', 'dt'], data=day_last_code)
    df.to_csv(os.path.join(daily_stock_path, f"{enddt}.csv"), header=True, index=False, encoding='utf_8_sig')

# Voorbeeld van hoe je de functie kunt aanroepen
# fun('2022-11-30', '2022-11-01', '2022-11-30', stock_data)
# fun('2022-12-30', '2022-12-01', '2022-12-30', stock_data)


already_done = True
for i in tqdm(range(prev_date_num-1, len(all_dates)), desc=f"Processing dates"):
    end_date = all_dates[i]
    if already_done:
        if end_date == '2024-03-26':
            already_done = False
            continue
    data_train_predict_path = os.path.join(data_path, "data_train_predict_corr")
    os.makedirs(data_train_predict_path, exist_ok=True)
    daily_stock_path = os.path.join(data_path, "daily_stock_corr")
    os.makedirs(daily_stock_path, exist_ok=True)
    fun(i, end_date, stock_data, prev_date_num, threshold, min_neighbors)






# import os
# import torch
# import pickle
# import numpy as np
# from tqdm import tqdm
# import networkx as nx
# import pandas as pd
# from torch.autograd import Variable

# # feature_cols = ['open','high','low','close','to','vol']
# feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume']

# # origineel: path1 = "\THGNN\data\csi300.pkl"
# base_path = os.path.dirname(os.path.abspath(__file__))  # Huidige scriptmap
# data_path = os.path.join(base_path, "..", "data")
# relation_path = os.path.join(data_path, "relation")
# path1 = os.path.join(data_path, "csi300.pkl")  # Relatieve verwijzing naar csi300.pkl
# df1 = pickle.load(open(path1, 'rb'), encoding='utf-8')
# relation = os.listdir(relation_path)
# relation = sorted(relation)
# date_unique=df1['dt'].unique()
# stock_trade_data=date_unique.tolist()
# stock_trade_data.sort()

# df1['dt']=df1['dt'].astype('datetime64')

# def fun(relation_dt, start_dt_month, end_dt_month,df1):
#     prev_date_num = 20
#     relation_file = os.path.join(relation_path, f"{relation_dt}.csv")
#     adj_all = pd.read_csv(relation_file, index_col=0)
#     #origineel: adj_all = pd.read_csv('/home/THGNN-main/data/relation/'+relation_dt+'.csv', index_col=0)
#     adj_stock_set = list(adj_all.index)

#     pos_g = nx.Graph(adj_all > 0.1)
#     pos_adj = nx.adjacency_matrix(pos_g).toarray()
#     pos_adj = pos_adj - np.diag(np.diag(pos_adj))
#     pos_adj = torch.from_numpy(pos_adj).type(torch.float32)

#     neg_g = nx.Graph(adj_all < -0.1)
#     neg_adj = nx.adjacency_matrix(neg_g)
#     neg_adj.data = np.ones(neg_adj.data.shape)
#     neg_adj = neg_adj.toarray()
#     neg_adj = neg_adj - np.diag(np.diag(neg_adj))
#     neg_adj = torch.from_numpy(neg_adj).type(torch.float32)

#     print('neg_adj over')
#     print(neg_adj.shape)

#     dts = stock_trade_data[stock_trade_data.index(start_dt_month):stock_trade_data.index(end_dt_month)+1]
#     print(dts)

#     for i in tqdm(range(len(dts))):
#         end_data=dts[i]
#         start_data = stock_trade_data[stock_trade_data.index(end_data)-(prev_date_num - 1)]
#         df2 = df1.loc[df1['dt'] <= end_data]
#         df2 = df2.loc[df2['dt'] >= start_data]
#         code = adj_stock_set
#         feature_all = []
#         mask = []
#         labels = []
#         day_last_code = []
#         for j in range(len(code)):
#             df3 = df2.loc[df2['code'] == code[j]]
#             y = df3[feature_cols].values
#             if y.T.shape[1] == prev_date_num:
#                 one = []
#                 feature_all.append(y)
#                 mask.append(True)
#                 label = df3.loc[df3['dt'] == end_data]['label'].values
#                 labels.append(label[0])
#                 one.append(code[j])
#                 one.append(end_data)
#                 day_last_code.append(one)
#         feature_all = np.array(feature_all)
#         features = torch.from_numpy(feature_all).type(torch.float32)
#         mask = [True]*len(labels)
#         labels = torch.tensor(labels, dtype=torch.float32)
#         result = {'pos_adj': Variable(pos_adj), 'neg_adj': Variable(neg_adj),  'features': Variable(features),
#                   'labels': Variable(labels), 'mask': mask}
#         #origineel:
# #        with open('/home/THGNN-main/data/data_train_predict/'+end_data+'.pkl', 'wb') as f:
# #            pickle.dump(result, f)
#         with open(os.path.join(data_path, "data_train_predict", f"{end_data}.pkl"), 'wb') as f:
#             pickle.dump(result, f)
#         df = pd.DataFrame(columns=['code', 'dt'], data=day_last_code)
# #        df.to_csv('/home/THGNN-main/data/daily_stock/'+end_data+'.csv', header=True, index=False, encoding='utf_8_sig')
#         df.to_csv(os.path.join(data_path, "daily_stock", f"{end_data}.csv"), header=True, index=False, encoding='utf_8_sig')
        
# #The first parameter and third parameters indicate the last trading day of each month, and the second parameter indicates the first trading day of each month.
# # for i in ['2020','2021','2022']:
# #     for j in ['01','02','03','04','05','06','07','08','09','10','11','12']:
# #         stock_m=[k for k in stock_trade_data if k>i+'-'+j and k<i+'-'+j+'-32']
# #         fun(stock_m[-1], stock_m[0], stock_m[-1], df1)
# fun('2022-11-30','2022-11-01','2022-11-30',df1)
# fun('2022-12-30','2022-12-01','2022-12-30',df1)