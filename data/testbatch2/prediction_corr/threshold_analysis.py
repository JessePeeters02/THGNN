import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import torch
import psutil
import seaborn as sns
import torch.nn as nn
from sklearn.metrics import r2_score



# Pad configuratie
label_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "stock_labels.csv")
print(label_path)
prediction_map = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "prediction_overnight")
print(prediction_map)

def evaluate_predictions(predictions, labels):
    mae = np.mean(np.abs(predictions - labels))
    mse = np.mean((predictions - labels) ** 2)
    r2 = r2_score(labels, predictions)
    # tpredictions = torch.tensor(predictions, dtype=torch.float32)
    # tlabels = torch.tensor(labels, dtype=torch.float32)
    # print('tlabels: ', tlabels)
    # print('tpredictions: ', tpredictions)
    # print(type(tpredictions), type(tlabels))
    # BCE = nn.BCELoss(reduction='mean')
    # bce = BCE(tpredictions, tlabels)
    return mae, mse, r2#, bce

def direction_accuracy(predictions, labels, threshold=0.0000000):
    if threshold == 'mean':
        thresh_val = np.mean(labels)
    else:
        thresh_val = threshold

    pred_up = predictions > thresh_val
    label_up = labels > 0
    # print(pred_up)
    # print(label_up)
    # print(f"Threshold: {thresh_val}")
    # print("  === positieve ===")
    # print(" positieve labels: ", np.sum(label_up))
    # print(" positieve voorspellingen: ", np.sum(pred_up))
    # print(" aantal positieve voorspellingen die ook positief zijn: ", np.sum(pred_up[label_up]))
    # print("  === negatieve ===")
    # print(" negatieve labels: ", np.sum(~label_up))
    # print(" negatieve voorspellingen: ", np.sum(~pred_up))
    # print(" aantal negatieve voorspellingen die ook negatief zijn: ", np.sum(~pred_up[~label_up]))
          
    acc = np.mean(pred_up == label_up)
    return acc

def check_labelsvsprediction(path):
    """ Controleer wat er in de eerste nr-aantal pkl-bestanden staat"""

    predictionsdf = pd.read_csv(os.path.join(path, "pred.csv"))
    predictiondates = pd.unique(predictionsdf["dt"].values)
    # print(f"predictiondates: {predictiondates}")
    # print(predictionsdf.head())
    
    # print("Bestandspad:", label_path)
    labelsdf = pd.read_csv(label_path, index_col=0)
    labelsdf = labelsdf[predictiondates]
    labelsdf['stock'] = labelsdf.index
    # print(labelsdf.head())

    def lookup_label(row):
        stock = row['code']
        date = row['dt']
        try:
            return labelsdf.loc[stock, date]
        except KeyError:
            print('niet gevonden:', stock, date)
            return float('nan')  # of np.nan als je NumPy gebruikt

    predictionsdf['true_score'] = predictionsdf.apply(lookup_label, axis=1)

    predictions = torch.tensor(predictionsdf['score'].values, dtype=torch.float32).numpy()
    labels = torch.tensor(predictionsdf['true_score'].values, dtype=torch.float32).numpy()

    tllabels = np.tanh(np.log(labels+1))
    print(len(labels), len(predictions))

    tllabel_stats = f"tanh log Labels - Gemiddelde: {np.mean(tllabels):.4f}, Std: {np.std(tllabels):.4f}, Max: {np.max(tllabels):.4f}, Min: {np.min(tllabels):.4f}"
    pred_stats = f"Voorspellingen - Gemiddelde: {np.mean(predictions):.4f}, Std: {np.std(predictions):.4f}, Max: {np.max(predictions):.4f}, Min: {np.min(predictions):.4f}"
    
    print(tllabel_stats)
    print(pred_stats)

    # mae, mse , bce = evaluate_predictions(predictions, labels)
    mae, mse, r2 = evaluate_predictions(predictions, tllabels)
    acc = direction_accuracy(predictions, tllabels)

    # print(f"MAE: {mae:.6f}")
    # print(f"MSE: {mse:.6f}")
    # print(f"RMSE: {np.sqrt(mse):.6f}")
    # # print(f"BCE: {bce:.6f}")
    # print(f"Accuracy op richting: {acc:.2%}")

    for horizon, name in [(1, 'day1'), (5, 'day5'), (20, 'day20')]:
        horizon_df = predictionsdf.groupby("code").head(horizon)
        
        preds = np.tanh(np.log(horizon_df["score"].values + 1))
        labels = np.tanh(np.log(horizon_df["true_score"].values + 1))

        mae, mse, r2 = evaluate_predictions(preds, labels)
        acc = direction_accuracy(preds, labels)

        results.append({
            "positive_threshold": p,
            "negative_threshold": n,
            "horizon": name,
            "mae": mae,
            "mse": mse,
            "rmse": np.sqrt(mse),
            "r2": r2,
            "accuracy": acc
        })



# # results aanmaken
# results = []

# for map in os.listdir(prediction_map):
#     try:
#         p, n = map.split("_")
#         p = float(p)
#         n = float(n)
#         print(p, n)
#         check_labelsvsprediction(os.path.join(prediction_map, map))
#     except:
#         print('niet gelukt')
#         continue
# results_df = pd.DataFrame(results)
# results_df.to_csv(os.path.join(prediction_map, "results.csv"), index=False)


# results zijn er al
results_df = pd.read_csv(os.path.join(prediction_map, "results.csv"))
print(results_df.head())

# filtered_df = results_df[(results_df['threshold'] >= 0.3) & (results_df['threshold'] <= 0.8)]
# filtered_df = filtered_df[filtered_df['horizon'].isin(['day5', 'day20'])]

plt.figure(figsize=(10,6))
sns.lineplot(data=results_df, x="positive_threshold", y="r2", hue="horizon", style="negative_threshold", markers=True, dashes=False)
plt.title("r2 per negative en positive")
plt.xlabel("positive")
plt.ylabel("r2")
plt.grid(True)
plt.tight_layout()
plt.show()



fig, ax = plt.subplots(figsize=(10, 4))
ax.axis('off')
table = ax.table(cellText=results_df.values, colLabels=results_df.columns, cellLoc='center', loc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.2)

plt.savefig("tabel_resultaten.png", bbox_inches='tight', dpi=300)