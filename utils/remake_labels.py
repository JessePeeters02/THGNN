import os
import time
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch

def load_stock_data(raw_stock_path):

    raw_files = [f for f in os.listdir(raw_stock_path) if f.endswith('.csv')]
    raw_data = {}
    for file in tqdm(raw_files, desc="Loading raw data"):
        stock_name = file.split('.')[0]
        df = pd.read_csv(os.path.join(raw_stock_path, file), parse_dates=['Date'])
        df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
        raw_data[stock_name] = df
    return raw_data

def read_daily_codes(daily_csv_path):
    df = pd.read_csv(daily_csv_path)
    date_str = df.iloc[0]['dt']
    codes = df['code'].astype(str).tolist()
    return date_str, codes

def calculate_new_label(raw_df, current_date):
    date_idx = raw_df[raw_df['Date'] == current_date].index[0]
    if date_idx == 0:
        raise RuntimeError(f"Geen vorige dag beschikbaar voor {current_date}.")
    close_today = raw_df.iloc[date_idx]['Close']
    close_yesterday = raw_df.iloc[date_idx-1]['Close']
    return (close_today / close_yesterday) - 1

def process_file(file_path, filename):
    with open(file_path, 'rb') as f:
        file = pickle.load(f)

    date = filename.split('.')[0]
    daily_csv = os.path.join(daily_stock_path_old, f"{date}.csv")
    date_from_daily, codes = read_daily_codes(daily_csv)
    if date_from_daily != date:
        raise RuntimeError(f"Datum mismatch: pickle={date} vs daily={date_from_daily}")

    # print(f"Processing file: {file_path} for date: {date}")
    pos_adj = file['pos_adj']
    neg_adj = file['neg_adj']
    features = file['features']
    old_labels = file['labels']
    mask = file['mask']

    if len(old_labels) != len(codes):
        raise RuntimeError(f"Label count mismatch: {len(old_labels)} vs {len(codes)} for date {date}")

    new_labels = np.empty(len(codes), dtype=np.float32)
    for i, code in enumerate(codes):
        new_labels[i] = calculate_new_label(raw_data[code], date)

    with open(os.path.join(data_train_predict_path_new, filename), 'wb') as f:
        pickle.dump({
            'pos_adj': pos_adj,
            'neg_adj': neg_adj,
            'features': features,
            'labels': torch.FloatTensor(new_labels),
            'mask': mask
        }, f)


if __name__ == "__main__":
    global raw_data, daily_stock_path_old, data_train_predict_path_new

    for database in ["S&P500", "CSI300"]:
        for datatype in ["corr", "DSE", "onlycosine", "cosineDSC"]:
            print(f"Start {database}, {datatype}: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            start_time = time.time()
            # Basis pad naar de data-map
            base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Huidige scriptmap
            print(base_path)
            data_path = os.path.join(base_path, "data", database)
            print(data_path)
            raw_data_path = os.path.join(data_path, "stockdata")
            print(raw_data_path)

            data_train_predict_path_old = os.path.join(data_path, f"data_train_predict_{datatype}_old")
            os.makedirs(data_train_predict_path_old, exist_ok=True)
            daily_stock_path_old = os.path.join(data_path, f"daily_stock_{datatype}")
            os.makedirs(daily_stock_path_old, exist_ok=True)

            data_train_predict_path_new = os.path.join(data_path, f"data_train_predict_{datatype}")
            os.makedirs(data_train_predict_path_new, exist_ok=True)

            raw_data = load_stock_data(raw_data_path)

            for filename in tqdm(os.listdir(data_train_predict_path_old), desc="Processing files"):
                file_path = os.path.join(data_train_predict_path_old, filename)
                if os.path.isfile(file_path):
                    process_file(file_path, filename)

            end_time = time.time()
            minutes_taken = round((end_time - start_time) / 60, 1)
            print(f"Done {database}, {datatype}. Time taken: {minutes_taken} minutes")

    databasebig = "NASDAQ_batches_5_200"
    for database in ["batch_1", "batch_2", "batch_3", "batch_4", "batch_5"]:
        for datatype in ["corr", "DSE", "onlycosine", "cosineDSC"]:
            print(f"Start {database}, {datatype}: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            start_time = time.time()
            # Basis pad naar de data-map
            base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Huidige scriptmap
            print(base_path)
            data_path = os.path.join(base_path, "data", databasebig, database)
            print(data_path)
            raw_data_path = os.path.join(data_path, "stockdata")
            print(raw_data_path)

            data_train_predict_path_old = os.path.join(data_path, f"data_train_predict_{datatype}_old")
            os.makedirs(data_train_predict_path_old, exist_ok=True)
            daily_stock_path_old = os.path.join(data_path, f"daily_stock_{datatype}")
            os.makedirs(daily_stock_path_old, exist_ok=True)

            data_train_predict_path_new = os.path.join(data_path, f"data_train_predict_{datatype}")
            os.makedirs(data_train_predict_path_new, exist_ok=True)

            raw_data = load_stock_data(raw_data_path)

            for filename in tqdm(os.listdir(data_train_predict_path_old), desc="Processing files"):
                file_path = os.path.join(data_train_predict_path_old, filename)
                if os.path.isfile(file_path):
                    process_file(file_path, filename)

            end_time = time.time()
            minutes_taken = round((end_time - start_time) / 60, 1)
            print(f"Done {database}, {datatype}. Time taken: {minutes_taken} minutes")
