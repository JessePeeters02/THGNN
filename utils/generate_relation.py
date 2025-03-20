import time
import pickle
import multiprocessing as mp
import numpy as np
import pandas as pd
from tqdm import tqdm
import os

feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume']

def cal_pccs(x, y, n):
    sum_xy = np.sum(np.sum(x*y))
    sum_x = np.sum(np.sum(x))
    sum_y = np.sum(np.sum(y))
    sum_x2 = np.sum(np.sum(x*x))
    sum_y2 = np.sum(np.sum(y*y))
    pcc = (n*sum_xy-sum_x*sum_y)/np.sqrt((n*sum_x2-sum_x*sum_x)*(n*sum_y2-sum_y*sum_y))
    return pcc

def calculate_pccs(xs, yss, n):
    result = []
    for name in yss:
        ys = yss[name]
        tmp_res = []
        for pos, x in enumerate(xs):
            y = ys[pos]
            tmp_res.append(cal_pccs(x, y, n))
        result.append(tmp_res)
    return np.mean(result, axis=1)

def stock_cor_matrix(ref_dict, codes, n, processes=1):
    if processes > 1:
        pool = mp.Pool(processes=processes)
        args_all = [(ref_dict[code], ref_dict, n) for code in codes]
        results = [pool.apply_async(calculate_pccs, args=args) for args in args_all]
        output = [o.get() for o in results]
        data = np.stack(output)
        return pd.DataFrame(data=data, index=codes, columns=codes)
    data = np.zeros([len(codes), len(codes)])
    for i in tqdm(range(len(codes))):
        data[i, :] = calculate_pccs(ref_dict[codes[i]], ref_dict, n)
    return pd.DataFrame(data=data, index=codes, columns=codes)

# Basis pad naar de data-map
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Huidige scriptmap
# print(base_path)
data_path = os.path.join(base_path, "data")
# print(data_path)
relation_path = os.path.join(data_path, "relation")
# print(relation_path)
stock_data_path = os.path.join(os.path.dirname(base_path), "portfolio_construction", "data", "NASDAQ_data")  # Map waar de CSV-bestanden staan
# print(stock_data_path)

# Functie om de CSV-bestanden in te lezen en om te zetten naar een DataFrame
def load_stock_data(stock_data_path):
    stock_files = [f for f in os.listdir(stock_data_path) if f.endswith('.csv')]
    stock_data = {}
    for stock_file in stock_files:
        stock_name = stock_file.split('.')[0]
        stock_df = pd.read_csv(os.path.join(stock_data_path, stock_file))
        stock_df['Date'] = pd.to_datetime(stock_df['Date'])
        stock_data[stock_name] = stock_df
    return stock_data

# Laad de stock data
stock_data = load_stock_data(stock_data_path)

#prev_date_num Indicates the number of days in which stock correlation is calculated
prev_date_num = 20
date_unique = pd.date_range(start='2022-11-01', end='2022-12-30', freq='B')  # Werkdagen tussen de gegeven datums
stock_trade_data = date_unique.tolist()
stock_trade_data.sort()
stock_num = len(stock_data)

#dt is the last trading day of each month
dt = ['2022-11-30', '2022-12-30']

for i in range(len(dt)):
    df2 = stock_data.copy()
    end_data = dt[i]
    start_data = stock_trade_data[stock_trade_data.index(end_data)-(prev_date_num - 1)]
    code = sorted(list(stock_data.keys()))
    test_tmp = {}
    for j in tqdm(range(len(code))):
        df3 = df2[code[j]].loc[df2[code[j]]['Date'] <= end_data]
        df3 = df3.loc[df3['Date'] >= start_data]
        y = df3[feature_cols].values
        if y.T.shape[1] == prev_date_num:
            test_tmp[code[j]] = y.T
    t1 = time.time()
    result = stock_cor_matrix(test_tmp, list(test_tmp.keys()), prev_date_num, processes=1)
    result = result.fillna(0)
    for i in range(0, stock_num):
        result.iloc[i, i] = 1
    t2 = time.time()
    print('time cost', t2 - t1, 's')
    
    # Sla het bestand correct op
    result.to_csv(os.path.join(relation_path, f"{end_data}.csv"))



# import time
# import pickle
# import multiprocessing as mp
# import numpy as np
# import pandas as pd
# from tqdm import tqdm
# import os

# feature_cols = ['high','low','close','open','to','vol']

# def cal_pccs(x, y, n):
#     sum_xy = np.sum(np.sum(x*y))
#     sum_x = np.sum(np.sum(x))
#     sum_y = np.sum(np.sum(y))
#     sum_x2 = np.sum(np.sum(x*x))
#     sum_y2 = np.sum(np.sum(y*y))
#     pcc = (n*sum_xy-sum_x*sum_y)/np.sqrt((n*sum_x2-sum_x*sum_x)*(n*sum_y2-sum_y*sum_y))
#     return pcc

# def calculate_pccs(xs, yss, n):
#     result = []
#     for name in yss:
#         ys = yss[name]
#         tmp_res = []
#         for pos, x in enumerate(xs):
#             y = ys[pos]
#             tmp_res.append(cal_pccs(x, y, n))
#         result.append(tmp_res)
#     return np.mean(result, axis=1)

# def stock_cor_matrix(ref_dict, codes, n, processes=1):
#     if processes > 1:
#         pool = mp.Pool(processes=processes)
#         args_all = [(ref_dict[code], ref_dict, n) for code in codes]
#         results = [pool.apply_async(calculate_pccs, args=args) for args in args_all]
#         output = [o.get() for o in results]
#         data = np.stack(output)
#         return pd.DataFrame(data=data, index=codes, columns=codes)
#     data = np.zeros([len(codes), len(codes)])
#     for i in tqdm(range(len(codes))):
#         data[i, :] = calculate_pccs(ref_dict[codes[i]], ref_dict, n)
#     return pd.DataFrame(data=data, index=codes, columns=codes)

# # origineel: path1 = "\THGNN\data\csi300.pkl"
# base_path = os.path.dirname(os.path.abspath(__file__))  # Huidige scriptmap
# path1 = os.path.join(base_path, "..", "data", "csi300.pkl")  # Relatieve verwijzing naar csi300.pkl
# df1 = pickle.load(open(path1, 'rb'), encoding='utf-8')
# #prev_date_num Indicates the number of days in which stock correlation is calculated
# prev_date_num = 20
# date_unique=df1['dt'].unique()
# stock_trade_data=date_unique.tolist()
# stock_trade_data.sort()
# stock_num=df1.code.unique().shape[0]
# #dt is the last trading day of each month
# dt=['2022-11-30','2022-12-30']
# # for i in ['2020','2021','2022']:
# #     for j in ['01','02','03','04','05','06','07','08','09','10','11','12']:
# #         stock_m=[k for k in stock_trade_data if k>i+'-'+j and k<i+'-'+j+'-32']
# #         dt.append(stock_m[-1])
# df1['dt']=df1['dt'].astype('datetime64')

# for i in range(len(dt)):
#     df2 = df1.copy()
#     end_data = dt[i]
#     start_data = stock_trade_data[stock_trade_data.index(end_data)-(prev_date_num - 1)]
#     df2 = df2.loc[df2['dt'] <= end_data]
#     df2 = df2.loc[df2['dt'] >= start_data]
#     code = sorted(list(set(df2['code'].values.tolist())))
#     test_tmp = {}
#     for j in tqdm(range(len(code))):
#         df3 = df2.loc[df2['code'] == code[j]]
#         y = df3[feature_cols].values
#         if y.T.shape[1] == prev_date_num:
#             test_tmp[code[j]] = y.T
#     t1 = time.time()
#     result = stock_cor_matrix(test_tmp, list(test_tmp.keys()), prev_date_num, processes=1)
#     result=result.fillna(0)
#     for i in range(0,stock_num):
#         result.iloc[i,i]=1
#     t2 = time.time()
#     print('time cost', t2 - t1, 's')
    
#     # origineel: result.to_csv("/home/THGNN-main/data/relation/"+str(end_data)+".csv")
#     #gewijzigd:
#     base_path = os.path.dirname(os.path.abspath(__file__))  # Huidige scriptmap
#     relation_dir = os.path.join(base_path, "..", "data", "relation")  # Relatieve pad naar de juiste folder

#     # Maak de directory aan als deze niet bestaat
#     os.makedirs(relation_dir, exist_ok=True)

#     # Sla het bestand correct op
#     result.to_csv(os.path.join(relation_dir, f"{end_data}.csv"))

