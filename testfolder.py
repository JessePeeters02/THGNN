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

def check_pickles(nr, path):
    """ Controleer wat er in de eerste nr-aantal pkl-bestanden staat """
    bestandspad = os.path.join(data_path, path)


""" aanroepen van alle testfuncties"""
check_pickles(10, "data_train_predict_DSE1")