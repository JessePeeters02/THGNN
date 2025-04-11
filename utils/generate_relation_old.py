import time
import pickle
import multiprocessing as mp
import numpy as np
import pandas as pd
from tqdm import tqdm
import os

feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume']

def cal_pccs(x, y, n):
    # print(f"Values of y: {y}")
    if np.isnan(x).any() or np.isnan(y).any():
        print(f"NaN values detected in x or y: x={x}, y={y}")
        return np.nan
    sum_xy = np.sum(np.sum(x*y))
    sum_x = np.sum(np.sum(x))
    sum_y = np.sum(np.sum(y))
    sum_x2 = np.sum(np.sum(x*x))
    sum_y2 = np.sum(np.sum(y*y))
    var_x = n * sum_x2 - sum_x * sum_x
    var_y = n * sum_y2 - sum_y * sum_y
    if var_x == 0 or var_y == 0:
        print(f"Zero variance detected: var_x={var_x}, var_y={var_y}")
        return np.nan
    denominator = np.sqrt((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y))
    if denominator == 0:
        print(f"Division by zero detected: denominator={denominator}")
        return np.nan  # Vermijd deling door nul
    pcc = (n*sum_xy-sum_x*sum_y)/denominator
    return pcc

def calculate_pccs(xs, yss, n):
    result = []
    for name in yss:
        ys = yss[name]
        # print(f"Values of ys for {name}: {ys}")
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
    for i in tqdm(range(len(codes)), desc="Calculating correlations"):
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
def load_stock_data(stock_data_path, filter_non_trading):
    stock_files = [f for f in os.listdir(stock_data_path) if f.endswith('.csv')]
    stock_data = {}
    non_trading_stocks = []
    for stock_file in stock_files:
        stock_name = stock_file.split('.')[0]
        stock_df = pd.read_csv(os.path.join(stock_data_path, stock_file))
        stock_df['Date'] = pd.to_datetime(stock_df['Date'])

        # Filter aandelen met niet-handelsdagen (indien gewenst)
        if filter_non_trading:
            if (stock_df['Volume'] == 0).any():  # Controleer of er minstens één dag is met volume = 0
                non_trading_stocks.append(stock_name)
                continue  # Sla dit aandeel over

        stock_data[stock_name] = stock_df
        # print(stock_data)

    if filter_non_trading:
        print(f"Aantal aandelen met niet-handelsdagen: {len(non_trading_stocks)}")
        print(f"Aantal aandelen in de huidige dataset: {len(stock_data)}")
    
    return stock_data

# Laad de stock data
stock_data = load_stock_data(stock_data_path, filter_non_trading=True)

# Bepaal de unieke datums uit de data
all_dates = []
for stock_name, df in stock_data.items():
    all_dates.extend(df['Date'].tolist())
date_unique = sorted(list(set(all_dates)))  # Unieke datums
stock_trade_data = date_unique
stock_trade_data.sort()
# print(stock_trade_data)
# print(len(stock_trade_data))

#prev_date_num geeft aan over hoeveel dagen de correlatie wordt berekend
prev_date_num = 20

# Genereer correlatiematrices voor elke maand
for i in range(prev_date_num, len(stock_trade_data)):
    t1 = time.time()
    end_data = stock_trade_data[i]
    start_data = stock_trade_data[i - prev_date_num+1]
    print(start_data, end_data)
    test_tmp = {}
    for stock_name, df in stock_data.items():
        # print(f"Unique dates for {stock_name}: {df['Date'].unique()}")  # Debug statement
        df2 = df.loc[df['Date'] <= end_data]
        df2 = df2.loc[df2['Date'] >= start_data]
        # print(f"Data for {stock_name} from {start_data} to {end_data}: {len(df2)} rows")  # Debug statement
        y = df2[feature_cols].values
        # print(f"Values of y for {stock_name}: {y}")
        # print(f"Shape of y.T: {y.T.shape}")  # Debug statement
        if y.T.shape[1] == prev_date_num:
            test_tmp[stock_name] = y.T
            # print("geslaagd")
    result = stock_cor_matrix(test_tmp, list(test_tmp.keys()), prev_date_num, processes=1)
    result = result.fillna(0)
    # print(result)
    for i in range(0, len(result)):
        result.iloc[i, i] = 1
    t2 = time.time()
    print('time cost', t2 - t1, 's')
    
    # Sla het bestand correct op
    result.to_csv(os.path.join(relation_path, f"{end_data.date()}.csv"))