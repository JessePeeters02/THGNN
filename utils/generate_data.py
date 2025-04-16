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
threshold = 0.6  # Drempelwaarde voor de correlatie

# Basis pad naar de data-map
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Huidige scriptmap
print(base_path)
data_path = os.path.join(base_path, "data", "testbatch1")
print(data_path)
relation_path = os.path.join(data_path, "relations")
print(relation_path)
stock_data_path = os.path.join(data_path, "normalisedstockdata")  # Map waar de CSV-bestanden staan
print(stock_data_path)

# Functie om de CSV-bestanden in te lezen en om te zetten naar een DataFrame
def load_stock_data(stock_data_path):
    stock_files = [f for f in os.listdir(stock_data_path) if f.endswith('.csv')]
    stock_data = {}
    for stock_file in tqdm(stock_files, desc="Loading stock data"):
        stock_name = stock_file.split('.')[0]
        df = pd.read_csv(os.path.join(stock_data_path, stock_file))
        df['Date'] = pd.to_datetime(df['Date'])
        stock_data[stock_name] = df
    return stock_data

# Laad de stock data
stock_data = load_stock_data(stock_data_path)
print(stock_data)

# Bepaal de unieke datums uit de data
all_dates = sorted({date.strftime('%Y-%m-%d') for df in stock_data.values() for date in df['Date'].tolist()})
print(f"Unique dates determined: {len(all_dates)}")
# print(all_dates)

def prepare_adjacencymatrix(enddt):
    relation_file = os.path.join(relation_path, f"{enddt}.csv")
    adj_all = pd.read_csv(relation_file, index_col=0)

    pos_adj = nx.adjacency_matrix(nx.Graph(adj_all > threshold)).toarray().astype(float)
    pos_adj = torch.FloatTensor(pos_adj - np.diag(np.diag(pos_adj)))

    neg_adj = nx.adjacency_matrix(nx.Graph(adj_all < -threshold)).toarray().astype(float)
    neg_adj = torch.FloatTensor(neg_adj - np.diag(np.diag(neg_adj)))
    
    print('pos_adj shape: ', pos_adj.shape)
    print('neg_adj shape: ', neg_adj.shape)

    return pos_adj, neg_adj

# Functie om de relatiegrafieken te verwerken
def fun(iend, enddt, stock_data, pdn):
    istart = iend - pdn + 1
    startdt = all_dates[istart]
    print(f"calculating window: {startdt} to {enddt}")
    
    pos_adj, neg_adj = prepare_adjacencymatrix(enddt)

    dts = all_dates[istart:iend+1]
    print("Processing dates:", len(dts), dts)
        
    features = []
    labels = []
    day_last_code = []
        
    for stock_name, df in stock_data.items():
        df_window = df[(df['Date'] >= startdt) & (df['Date'] <= enddt)]
        print(f"df window shape: {df_window.shape}")
        print(f"df window: {df_window}")
        day_last_code.append([stock_name, enddt])
        if len(df_window) == pdn:
            features.append(df_window[feature_cols].values)
            labels.append(df_window.iloc[-1]['Close'])
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

    with open(os.path.join(data_path, "data_train_predict", f"{enddt}.pkl"), 'wb') as f:
        pickle.dump(output, f)
    df = pd.DataFrame(columns=['code', 'dt'], data=day_last_code)
    df.to_csv(os.path.join(data_path, "daily_stock", f"{enddt}.csv"), header=True, index=False, encoding='utf_8_sig')
    
    #     feature_all = np.array(feature_all)
    #     features = torch.from_numpy(feature_all).type(torch.float32)
    #     mask = [True] * len(labels)
    #     labels = torch.tensor(labels, dtype=torch.float32)
        
    # result = {'pos_adj': Variable(pos_adj), 'neg_adj': Variable(neg_adj), 'features': Variable(features),
    #           'labels': Variable(labels), 'mask': mask}
        
    # with open(os.path.join(data_path, "data_train_predict", f"{enddt}.pkl"), 'wb') as f:
    #     pickle.dump(result, f)
        
    # df = pd.DataFrame(columns=['code', 'dt'], data=day_last_code)
    # df.to_csv(os.path.join(data_path, "daily_stock", f"{enddt}.csv"), header=True, index=False, encoding='utf_8_sig')


# Voorbeeld van hoe je de functie kunt aanroepen
# fun('2022-11-30', '2022-11-01', '2022-11-30', stock_data)
# fun('2022-12-30', '2022-12-01', '2022-12-30', stock_data)

for i in range(prev_date_num-1, len(all_dates)):
    end_date = all_dates[i]
    fun(i, end_date, stock_data, prev_date_num)






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