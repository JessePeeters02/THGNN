import os
import torch
import pickle
import numpy as np
from tqdm import tqdm
import pandas as pd
import time

feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Turnover']
prev_date_num = 20

threshold = 0.4
min_neighbors = 3

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
    close_today = raw_df.iloc[date_idx]['Close']
    close_tomorrow = raw_df.iloc[date_idx+1]['Close']
    return (close_tomorrow / close_today) - 1

def prepare_adjacencymatrix(enddt, threshold, min_neighbors):
    relation_file = os.path.join(relation_path, f"{enddt}.csv")
    adj_all = pd.read_csv(relation_file, index_col=0)
    stock_names = adj_all.index.tolist()
    adj_values = adj_all.values
    
    pos_adj = (adj_values > threshold).astype(float)
    neg_adj = (adj_values < -threshold).astype(float)
    
    np.fill_diagonal(pos_adj, 0)
    np.fill_diagonal(neg_adj, 0)
    
    for i in range(len(stock_names)):
        pos_neighbors = np.sum(pos_adj[i])
        if pos_neighbors < min_neighbors:
            corrs = adj_values[i].copy()
            corrs[i] = 0
            top_pos_indices = np.argsort(-corrs)[:min_neighbors]
            pos_adj[i, top_pos_indices] = 1
        
        neg_neighbors = np.sum(neg_adj[i])
        if neg_neighbors < min_neighbors:
            corrs = adj_values[i].copy()
            corrs[i] = 0
            top_neg_indices = np.argsort(corrs)[:min_neighbors]
            neg_adj[i, top_neg_indices] = 1
    
    return torch.FloatTensor(pos_adj), torch.FloatTensor(neg_adj)


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
            label = calculate_label(raw_df, enddt)
            labels.append(label)
        else:
            print(' huh, len df window pdn????')
            break
        
    output = {
        'pos_adj': torch.FloatTensor(pos_adj),
        'neg_adj': torch.FloatTensor(neg_adj),
        'features': torch.FloatTensor(np.array(features)),
        'labels': torch.FloatTensor(labels),
        'mask': [True] * len(labels)
    }

    with open(os.path.join(data_train_predict_path, f"{enddt}.pkl"), 'wb') as f:
        pickle.dump(output, f)
    df = pd.DataFrame(columns=['code', 'dt'], data=day_last_code)
    df.to_csv(os.path.join(daily_stock_path, f"{enddt}.csv"), header=True, index=False, encoding='utf_8_sig')



def CSI300():
    print(f"Start CSI300: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    start_time = time.time()
    global base_path, data_path, relation_path, raw_data_path, stock_data_path, raw_data, stock_data, all_dates, data_train_predict_path, daily_stock_path
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print(base_path)
    data_path = os.path.join(base_path, "data", "CSI300")
    print(data_path)
    relation_path = os.path.join(data_path, "correlations")
    print(relation_path)
    raw_data_path = os.path.join(data_path, "stockdata")
    print(raw_data_path)
    stock_data_path = os.path.join(data_path, "normalisedstockdata")
    print(stock_data_path)

    raw_data, stock_data = load_stock_data(raw_data_path, stock_data_path)
    all_dates = sorted({date.strftime('%Y-%m-%d') for df in stock_data.values() for date in df['Date'].tolist()})
    for i in tqdm(range(prev_date_num-1, len(all_dates)-1), desc=f"Processing dates"):
        end_date = all_dates[i]
        data_train_predict_path = os.path.join(data_path, "data_train_predict_corr")
        os.makedirs(data_train_predict_path, exist_ok=True)
        daily_stock_path = os.path.join(data_path, "daily_stock_corr")
        os.makedirs(daily_stock_path, exist_ok=True)
        fun(i, end_date, stock_data, prev_date_num, threshold, min_neighbors)
    end_time = time.time()
    minutes_taken = round((end_time - start_time) / 60, 1)
    print(f"Done CSI300. Time taken: {minutes_taken} minutes") 

def SP500():
    print(f"Start SP500: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    start_time = time.time()
    global base_path, data_path, relation_path, raw_data_path, stock_data_path, raw_data, stock_data, all_dates, data_train_predict_path, daily_stock_path
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print(base_path)
    data_path = os.path.join(base_path, "data", "S&P500")
    print(data_path)
    relation_path = os.path.join(data_path, "correlations")
    print(relation_path)
    raw_data_path = os.path.join(data_path, "stockdata")
    print(raw_data_path)
    stock_data_path = os.path.join(data_path, "normalisedstockdata")
    print(stock_data_path)

    raw_data, stock_data = load_stock_data(raw_data_path, stock_data_path)
    all_dates = sorted({date.strftime('%Y-%m-%d') for df in stock_data.values() for date in df['Date'].tolist()})
    for i in tqdm(range(prev_date_num-1, len(all_dates)-1), desc=f"Processing dates"):
        end_date = all_dates[i]
        data_train_predict_path = os.path.join(data_path, "data_train_predict_corr")
        os.makedirs(data_train_predict_path, exist_ok=True)
        daily_stock_path = os.path.join(data_path, "daily_stock_corr")
        os.makedirs(daily_stock_path, exist_ok=True)
        fun(i, end_date, stock_data, prev_date_num, threshold, min_neighbors)
    end_time = time.time()
    minutes_taken = round((end_time - start_time) / 60, 1)
    print(f"Done SP500. Time taken: {minutes_taken} minutes")

# CSI300()
# SP500()


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
