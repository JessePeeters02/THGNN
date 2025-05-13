import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
from trainer.trainer import *
from data_loader import *
from model.Thgnn import *
from torch.utils.data import DataLoader
import torch
import psutil
import seaborn as sns
import torch.nn as nn


# Pad configuratie
base_path = os.path.dirname(os.path.abspath(__file__))
# print(base_path)
data_path = os.path.join(base_path, "data", "testbatch2")
# print(data_path)
daily_data_path = os.path.join(data_path, "NASDAQ_per_dag")
# print(daily_data_path)
stock_data_path = os.path.join(os.path.dirname(base_path), "portfolio_construction", "data", "NASDAQ_data")  # Map waar de CSV-bestanden staan
# print(stock_data_path)

""" alle testfuncties definieren """

def check_pickles(nr, path, start):
    """ Controleer wat er in de eerste nr-aantal pkl-bestanden staat """
    bestandspad = os.path.join(data_path, path)
    print("Bestandspad:", bestandspad)
    for file in os.listdir(bestandspad)[start:start+nr]:
        print("Bestand:", file)
        file = os.path.join(bestandspad, file)
        with open(file, 'rb') as f:
            data = pickle.load(f)

        # Print de keys van het opgeslagen dict
        print("Keys in het bestand:", data.keys())

        # Voor een idee van de inhoud:
        # print("\nVoorbeeldshape features:", data['features'].shape)
        # print("Aantal labels:", len(data['labels']))
        # print("Voorbeeldlabels:", data['labels'])  # eerste 10 labels
        # print("Voorbeeldmask:", data['mask'][:10])  # eerste 10 mask-waarden
        # print("Positive adj shape:", data['pos_adj'].shape)
        # print("Negative adj shape:", data['neg_adj'].shape)

        # print("Positive adjacency (10x10):")
        # print(data['pos_adj'][:15, :15])

        # print("\nNegative adjacency (10x10):")
        # print(data['neg_adj'][:15, :15])

        pos_counts = data['pos_adj'].sum(dim=1)
        neg_counts = data['neg_adj'].sum(dim=1)

        total_pos = pos_counts.sum()
        total_neg = neg_counts.sum()

        print(f"Totaal aantal positieve buren: {total_pos}")
        print(f"Totaal aantal negatieve buren: {total_neg}")

        # for i in range(len(pos_counts)):
        #     print(f"Aandeel {i}: {int(pos_counts[i])} positieve buren, {int(neg_counts[i])} negatieve buren")

        # Eventueel 1 feature sample inspecteren
        # print("\features:")
        # print(data['features'])

def evaluate_predictions(predictions, labels):
    mae = np.mean(np.abs(predictions - labels))
    mse = np.mean((predictions - labels) ** 2)
    # tpredictions = torch.tensor(predictions, dtype=torch.float32)
    # tlabels = torch.tensor(labels, dtype=torch.float32)
    # print('tlabels: ', tlabels)
    # print('tpredictions: ', tpredictions)
    # print(type(tpredictions), type(tlabels))
    # BCE = nn.BCELoss(reduction='mean')
    # bce = BCE(tpredictions, tlabels)
    return mae, mse#, bce

def direction_accuracy(predictions, labels, threshold=0.0000000):
    if threshold == 'mean':
        thresh_val = np.mean(labels)
    else:
        thresh_val = threshold

    pred_up = predictions > thresh_val
    label_up = labels > 0
    # print(pred_up)
    # print(label_up)
    print(f"Threshold: {thresh_val}")
    print("  === positieve ===")
    print(" positieve labels: ", np.sum(label_up))
    print(" positieve voorspellingen: ", np.sum(pred_up))
    print(" aantal positieve voorspellingen die ook positief zijn: ", np.sum(pred_up[label_up]))
    print("  === negatieve ===")
    print(" negatieve labels: ", np.sum(~label_up))
    print(" negatieve voorspellingen: ", np.sum(~pred_up))
    print(" aantal negatieve voorspellingen die ook negatief zijn: ", np.sum(~pred_up[~label_up]))
          
    acc = np.mean(pred_up == label_up)
    return acc

def check_labelsvsprediction(nr, path, start):
    """ Controleer wat er in de eerste nr-aantal pkl-bestanden staat"""
    bestandspad = os.path.join(data_path, path)
    predictionsdf = pd.read_csv(os.path.join(data_path, "prediction_corr", "0.6_5", "pred.csv"))
    predictions = predictionsdf["score"].values
    predictiondates = set(predictionsdf["dt"].values)
    predictions = predictions[:200]
    # print(f"predictions: {predictions}")
    print(f"predictiondates: {predictiondates}")
    print("Bestandspad:", bestandspad)
    labels = []
    startind = 1194
    for file in os.listdir(bestandspad)[startind:startind+1]:
        print("Bestand:", file)
        file = os.path.join(bestandspad, file)
        with open(file, 'rb') as f:
            data = pickle.load(f)
        print("labels keys: ",data.keys())
        labels.append(data['labels'].numpy())
    labels = np.concatenate(labels)
    # print(f"labels: {labels}")
    label_up = (labels >= 0).astype(int)
    tllabels = np.tanh(np.log(labels+1))
    print(len(labels), len(predictions))

    # label_stats = f"Labels - Gemiddelde: {np.mean(labels):.4f}, Std: {np.std(labels):.4f}, Max: {np.max(labels):.4f}, Min: {np.min(labels):.4f}"
    tllabel_stats = f"tanh log Labels - Gemiddelde: {np.mean(tllabels):.4f}, Std: {np.std(tllabels):.4f}, Max: {np.max(tllabels):.4f}, Min: {np.min(tllabels):.4f}"
    pred_stats = f"Voorspellingen - Gemiddelde: {np.mean(predictions):.4f}, Std: {np.std(predictions):.4f}, Max: {np.max(predictions):.4f}, Min: {np.min(predictions):.4f}"
    
    # print(label_stats)
    print(tllabel_stats)
    print(pred_stats)

    # mae, mse , bce = evaluate_predictions(predictions, labels)
    mae, mse = evaluate_predictions(predictions, tllabels)
    acc = direction_accuracy(predictions, tllabels)

    print(f"MAE: {mae:.6f}")
    print(f"MSE: {mse:.6f}")
    print(f"RMSE: {np.sqrt(mse):.6f}")
    # print(f"BCE: {bce:.6f}")
    print(f"Accuracy op richting: {acc:.2%}")

    # Plot optimalisaties
    bins = 50
    min_val = min(np.min(labels), np.min(predictions), np.min(tllabels))
    max_val = max(np.max(labels), np.max(predictions), np.max(tllabels))
    
    # Maak figuren parallel
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    
    # Histogram
    ax1.hist(predictions, bins=bins, range=(min_val, max_val), 
            alpha=0.5, label="Voorspellingen", density=True)
    # ax1.hist(labels, bins=bins, range=(min_val, max_val),
    #         alpha=0.5, label="Echte labels", density=True)
    ax1.hist(tllabels, bins=bins, range=(min_val, max_val),
            alpha=0.5, label="Echte labels", density=True)
    ax1.legend()
    ax1.set_title("Verdeling van returns")
    ax1.grid(True)
    
    # Boxplot
    ax2.boxplot([predictions, labels, tllabels], tick_labels=["Voorspellingen", "Echte labels", "tanh log labels"])
    ax2.set_title("Spreiding van returns")
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()


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

# def check_pickles(nr, path, start):
#     """ Controleer wat er in de eerste nr-aantal pkl-bestanden staat """
#     bestandspad = os.path.join(data_path, path)

#     for file in os.listdir(bestandspad)[start:start+nr]:
#         print("Bestand:", file)
#         file = os.path.join(bestandspad, file)
#         with open(file, 'rb') as f:
#             data = pickle.load(f)

#         # Print de keys van het opgeslagen dict
#         print("Keys in het bestand:", data.keys())

#         # Voor een idee van de inhoud:
#         print("\nVoorbeeldshape features:", data['features'].shape)
#         print("Aantal labels:", len(data['labels']))
#         print("Voorbeeldlabels:", data['labels'][:10])  # eerste 10 labels
#         print("Voorbeeldmask:", data['mask'][:10])  # eerste 10 mask-waarden
#         print("Positive adj shape:", data['pos_adj'].shape)
#         print("Negative adj shape:", data['neg_adj'].shape)

#         print("Positive adjacency (10x10):")
#         print(data['pos_adj'][:10, :10])

#         print("\nNegative adjacency (10x10):")
#         print(data['neg_adj'][:10, :10])

#         pos_counts = data['pos_adj'].sum(dim=1)
#         neg_counts = data['neg_adj'].sum(dim=1)

#         for i in range(10):
#             print(f"Aandeel {i}: {int(pos_counts[i])} positieve buren, {int(neg_counts[i])} negatieve buren")

        # Eventueel 1 feature sample inspecteren
        # print("\features:")
        # print(data['features'])

# Functie om geheugeninformatie weer te geven
def memory_info():
    virtual_memory = psutil.virtual_memory()
    print(f"Beschikbaar RAM: {virtual_memory.available / (1024 ** 3):.2f} GB")
    print(f"Totaal RAM: {virtual_memory.total / (1024 ** 3):.2f} GB")

# Functie om CPU-informatie weer te geven
def cpu_info():
    cpu_cores = os.cpu_count()
    print(f"Aantal CPU-cores: {cpu_cores}")
    cpu_usage = psutil.cpu_percent(interval=1)
    print(f"CPU-gebruik: {cpu_usage}%")

# Functie om GPU-informatie weer te geven (indien beschikbaar)
def gpu_info():
    if torch.cuda.is_available():
        print(f"CUDA beschikbaar: Ja")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA-geheugen beschikbaar: {torch.cuda.memory_allocated(0) / (1024 ** 3):.2f} GB")
        print(f"Totale GPU-geheugen: {torch.cuda.memory_reserved(0) / (1024 ** 3):.2f} GB")
    else:
        print("CUDA beschikbaar: Nee")


""" aanroepen van alle testfuncties"""
# check_pickles(3, "data_train_predict_DSE_noknn2", 7)
# check_pickles(3, "data_train_predict", len(os.listdir(os.path.join(data_path, "data_train_predict")))-3)
check_labelsvsprediction(2, os.path.join("data_train_predict_corr", "0.4_3"), 20)
# check_pickles(30, "data_train_predict", 20)
# check_csi300()
# memory_info()
# cpu_info()
# gpu_info()