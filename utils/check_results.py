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
from scipy.stats import ks_2samp



# Pad configuratie
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Huidige scriptmap
print(base_path)
data_path = os.path.join(base_path, "data", "testbatch2")
print(data_path)
label_path = os.path.join(data_path, "stock_labels.csv")
print(label_path)


def distribution(cpreds, labels, dpred):
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    ax1, ax2, ax3 = axes.flatten()

    # Find overall min and max values for x and y axes
    all_data = np.concatenate([cpreds, labels, dpred])
    x_min, x_max = np.min(all_data), np.max(all_data)

    # Create histograms and store the return values
    h1 = sns.histplot(cpreds, bins=100, kde=True, color='blue', label='voorspellingen', ax=ax1)
    h2 = sns.histplot(labels, bins=100, kde=True, color='orange', label='Labels', ax=ax2)
    h3 = sns.histplot(dpred, bins=100, kde=True, color='blue', label='voorspellingen', ax=ax3)

    # Find the maximum y value across all plots
    y_max = max([ax.get_ylim()[1] for ax in [ax1, ax2, ax3]])

    # Set titles and labels
    ax1.set_title("Distributie van Correlatie voorspellingen")
    ax2.set_title("Distributie van labels")
    ax3.set_title("Distributie van Dynami voorspellingen")

    # Set same x and y limits for all plots
    for ax in [ax1, ax2, ax3]:
        ax.set_xlabel("Waarde")
        ax.set_ylabel("Frequentie")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(0, y_max)  # Set same y limits
        ax.legend()

    plt.tight_layout()
    plt.show()



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


def check_labelsvsprediction(path, modelname):
    """ Controleer wat er in de eerste nr-aantal pkl-bestanden staat"""

    predictionsdf = pd.read_csv(os.path.join(path, "pred.csv"))
    predictiondates = pd.unique(predictionsdf["dt"].values)
    # predictiondates = predictiondates[0:1] # het aantal dagen aanpassen
    print(f"predictiondates: {predictiondates}")
    predictionsdf = predictionsdf[predictionsdf['dt'].isin(predictiondates)]  # Filter op de eerste 5 dagen
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
    # print(predictionsdf)

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
    d, p = ks_2samp(labels, predictions)
    print(f"KS-D distribution: {d:.4f} (p-value={p:.4g})")

    print(f"MAE: {mae:.6f}")
    print(f"MSE: {mse:.6f}")
    print(f"RMSE: {np.sqrt(mse):.6f}")
    print(f"R2: {r2:.6f}")
    # print(f"BCE: {bce:.6f}")
    print(f"Accuracy op richting: {acc:.2%}")

    # for horizon, name in [(1, 'day1'), (5, 'day5'), (20, 'day20')]:
    #     horizon_df = predictionsdf.groupby("code").head(horizon)
        
    #     preds = np.tanh(np.log(horizon_df["score"].values + 1))
    #     labels = np.tanh(np.log(horizon_df["true_score"].values + 1))

    #     mae, mse, r2 = evaluate_predictions(preds, labels)
    #     acc = direction_accuracy(preds, labels)

    #     results.append({
    #         # "positive_threshold": p,
    #         # "negative_threshold": n,
    #         "name": modelname,
    #         "horizon": name,
    #         "mae": mae,
    #         "mse": mse,
    #         "rmse": np.sqrt(mse),
    #         "r2": r2,
    #         "accuracy": acc
    #     })

    return tllabels, predictions

results = []
prediction_path = os.path.join(data_path, "prediction_corr_TE", "0.4_3")
print(prediction_path)
print(" ==== te small ====")
labels, corrpredictions = check_labelsvsprediction(prediction_path, 'small')

prediction_path = os.path.join(data_path, "prediction_corr_TEbig", "0.4_3")
print(prediction_path)
print(" ==== te big ====")
labels, dynamipredictions = check_labelsvsprediction(prediction_path, 'big')

distribution(corrpredictions, labels, dynamipredictions)

# results_df = pd.DataFrame(results)
# results_df.to_csv(os.path.join(data_path, "results.csv"), index=False)

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
results_df = pd.read_csv(os.path.join(data_path, "results.csv"))
print(results_df.head())

# filtered_df = results_df[(results_df['threshold'] >= 0.3) & (results_df['threshold'] <= 0.8)]
# filtered_df = filtered_df[filtered_df['horizon'].isin(['day5', 'day20'])]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

sns.lineplot(data=results_df, x="positive_threshold", y="rmse", hue="horizon", style="negative_threshold", markers=True, dashes=False, ax=ax1)
ax1.set_title("RMSE per negative and positive")
ax1.set_xlabel("Positive")
ax1.set_ylabel("RMSE")
ax1.grid(True)

sns.lineplot(data=results_df, x="positive_threshold", y="r2", hue="horizon", style="negative_threshold", markers=True, dashes=False, ax=ax2)
ax2.set_title("r2 per negative and positive")
ax2.set_xlabel("Positive")
ax2.set_ylabel("r2")
ax2.grid(True)

plt.tight_layout()
plt.show()
