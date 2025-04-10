import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
from trainer.trainer import *
from data_loader import *
from model.Thgnn import *
from torch.utils.data import DataLoader

# Pad configuratie
base_path = os.path.dirname(os.path.abspath(__file__))
# print(base_path)
data_path = os.path.join(base_path, "data")
# print(data_path)
daily_data_path = os.path.join(data_path, "NASDAQ_per_dag")
# print(daily_data_path)
stock_data_path = os.path.join(os.path.dirname(base_path), "portfolio_construction", "data", "NASDAQ_data")  # Map waar de CSV-bestanden staan
# print(stock_data_path)

""" alle testfuncties definieren """

def check_pickles(nr, path, start):
    """ Controleer wat er in de eerste nr-aantal pkl-bestanden staat """
    bestandspad = os.path.join(data_path, path)

    for file in os.listdir(bestandspad)[start:start+nr]:
        print("Bestand:", file)
        file = os.path.join(bestandspad, file)
        with open(file, 'rb') as f:
            data = pickle.load(f)

        # Print de keys van het opgeslagen dict
        print("Keys in het bestand:", data.keys())

        # Voor een idee van de inhoud:
        print("\nVoorbeeldshape features:", data['features'].shape)
        print("Aantal labels:", len(data['labels']))
        print("Voorbeeldmask:", data['mask'][:10])  # eerste 10 mask-waarden
        print("Positive adj shape:", data['pos_adj'].shape)
        print("Negative adj shape:", data['neg_adj'].shape)

        print("Positive adjacency (10x10):")
        print(data['pos_adj'][:10, :10])

        print("\nNegative adjacency (10x10):")
        print(data['neg_adj'][:10, :10])

        pos_counts = data['pos_adj'].sum(dim=1)
        neg_counts = data['neg_adj'].sum(dim=1)

        for i in range(10):
            print(f"Aandeel {i}: {int(pos_counts[i])} positieve buren, {int(neg_counts[i])} negatieve buren")

        # Eventueel 1 feature sample inspecteren
        print("\nEerste feature sample:")
        print(data['features'][0])


""" aanroepen van alle testfuncties"""
check_pickles(1, "data_train_predict_DSE1", 0)
check_pickles(1, "data_train_predict", 19)