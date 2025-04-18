import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
from trainer.trainer import *
from data_loader import *
from model.Thgnn import *
from torch.utils.data import DataLoader
import torch.nn as nn


# Pad configuratie
base_path = os.path.dirname(os.path.abspath(__file__))
# print(base_path)
data_path = os.path.join(base_path, "data", "testbatch1")
# print(data_path)
daily_data_path = os.path.join(data_path, "intermediate_snapshots")
# print(daily_data_path)
stock_data_path = os.path.join(os.path.dirname(base_path), "portfolio_construction", "data", "NASDAQ_data")  # Map waar de CSV-bestanden staan
# print(stock_data_path)

""" alle testfuncties definieren """
file = os.path.join(data_path, "normaliseddailydata", "2020-11-25.csv")
df = pd.read_csv(file)
print(df.head())
print(df['Open'].max())
print(df['High'].max())
print(df['Low'].max())
print(df['Close'].max())
print(df['Volume'].max())

def check_pickles(nr, path, start):
    """ Controleer wat er in de eerste nr-aantal pkl-bestanden staat """
    bestandspad = os.path.join(daily_data_path)

    for file in os.listdir(bestandspad)[149:156]:
        print("Bestand:", file)
        file = os.path.join(bestandspad, file)
        with open(file, 'rb') as f:
            data = pickle.load(f)
        print(data.keys())
        features = data.get('features')
        # print(features)
        print("\nfeature shape: ",features.shape)
        print(features.min(), features.max())
        feature_encoder = nn.Linear(5, 64)
        h = feature_encoder(features)
        print(h)
        print("\nh shape", h.shape)
        # print(features)
        if features is None:
            if torch.isnan(features).any() or torch.isinf(features).any():  
                print(f" STOP: NaN of Inf gevonden in features van {file}")
                print("Feature shape:", features.shape)
                nan_rows = features[torch.isnan(features).any(dim=1)]
                print("Rijen met NaN:")
                print(nan_rows)
                raise ValueError(f"Corrupt feature data in {file}")
        # Print de keys van het opgeslagen dict
        # print("Keys in het bestand:", data.keys())

        # # Voor een idee van de inhoud:
        # print("\nVoorbeeldshape features:", data['features'].shape)
        # print("Aantal labels:", len(data['labels']))
        # print("Voorbeeldlabels:", data['labels'][:50])  # eerste 10 labels
        # print("Voorbeeldmask:", data['mask'][:10])  # eerste 10 mask-waarden
        # print("Positive adj shape:", data['pos_edges'].shape)
        # print("Negative adj shape:", data['neg_edges'].shape)

        # print("Positive adjacency (10x10):")
        # print(data['pos_edges'][:10, :10])

        # print("\nNegative adjacency (10x10):")
        # print(data['neg_edges'][:10, :10])
        # print(len(data["tickers"]))
        # N = len(data['tickers'])
        # pos_adj = torch.zeros((N, N))
        # if data['pos_edges'].size(1) > 0:
        #     pos_adj[data['pos_edges'][0], data['pos_edges'][1]] = 1.0
        # neg_adj = torch.zeros((N, N))
        # if data['neg_edges'].size(1) > 0:
        #     neg_adj[data['neg_edges'][0], data['neg_edges'][1]] = 1.0

        # pos_counts = pos_adj.sum(dim=1)  
        # neg_counts = neg_adj.sum(dim=1)

        # for i in range(N):
        #     print(f"Aandeel {i}: {int(pos_counts[i])} positieve buren, {int(neg_counts[i])} negatieve buren")
        #     if (int(pos_counts[i]) == 0) and (int(neg_counts[i]) == 0):
        #         print(f"STOP: Aandeel {i} heeft geen positieve of negatieve buren")
                # raise ValueError(f"Corrupt adjacency data in {file}")

        # Eventueel 1 feature sample inspecteren
        # print("\features:")
        # print(data['features'])

def check_csi300():
    """ Controleer wat er in de eerste nr-aantal pkl-bestanden staat """
    bestandspad = os.path.join(data_path)
    file = os.path.join(bestandspad, "csi300.pkl")
    with open(file, 'rb') as f:
        data = pickle.load(f)
    print(type(data))
    print(data['code'])
    df = data[data['code'] == '000001.SZ'].reset_index(drop=True)
    df['labeltest'] = df['close'].shift(-1) / df['close'] - 1
    print(df)
    df = data[data['code'] == '000002.SZ'].reset_index(drop=True)
    df['labeltest'] = df['close'].shift(-1) / df['close'] - 1
    print(df)
    df = data[data['code'] == '000063.SZ'].reset_index(drop=True)
    df['labeltest'] = df['close'].shift(-1) / df['close'] - 1
    print(df)
    df = data[data['code'] == '000069.SZ'].reset_index(drop=True)
    df['labeltest'] = df['close'].shift(-1) / df['close'] - 1
    print(df)
    # print("Keys in het bestand:", data.keys())

        

""" aanroepen van alle testfuncties"""
# check_pickles(1, "data_train_predict", 0)
# check_pickles(30, "data_train_predict", 20)
# check_csi300()