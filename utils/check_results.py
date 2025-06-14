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

from scipy.stats import wasserstein_distance
from sklearn.metrics import precision_score, recall_score, f1_score, matthews_corrcoef


# Pad configuratie
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Huidige scriptmap
print(base_path)

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



def evaluate_reg_predictions(predictions, labels):
    mae = np.mean(np.abs(predictions - labels))
    mse = np.mean((predictions - labels) ** 2)
    r2 = r2_score(labels, predictions)
    distance = wasserstein_distance(predictions, labels)
    # tpredictions = torch.tensor(predictions, dtype=torch.float32)
    # tlabels = torch.tensor(labels, dtype=torch.float32)
    # print('tlabels: ', tlabels)
    # print('tpredictions: ', tpredictions)
    # print(type(tpredictions), type(tlabels))
    # BCE = nn.BCELoss(reduction='mean')
    # bce = BCE(tpredictions, tlabels)
    return mae, mse, r2, distance#, bce

def evaluate_cla_predictions(predictions, labels):
    tpredictions = torch.tensor(predictions, dtype=torch.float32)
    tlabels = torch.tensor(labels, dtype=torch.float32)
    # print('tlabels: ', tlabels)
    # print('tpredictions: ', tpredictions)
    # print(type(tpredictions), type(tlabels))
    BCE = nn.BCELoss(reduction='mean')
    bce = BCE(tpredictions, tlabels)
    
    # Calculate accuracy
    pred_classes = (tpredictions > 0.5).float()
    accuracy = (pred_classes == tlabels).float().mean().item()

    precision = precision_score(labels, pred_classes)
    recall = recall_score(labels, pred_classes)
    f1 = f1_score(labels, pred_classes)
    Mcorc = matthews_corrcoef(labels, pred_classes)
    
    return bce, accuracy, precision, recall, f1, Mcorc

# def direction_accuracy(predictions, labels, threshold=0.0000000):
#     if threshold == 'mean':
#         thresh_val = np.mean(labels)
#     else:
#         thresh_val = threshold

#     pred_up = predictions > thresh_val
#     label_up = labels > 0
#     # print(pred_up)
#     # print(label_up)
#     print(f"Threshold: {thresh_val}")
#     print("  === positieve ===")
#     print(" positieve labels: ", np.sum(label_up))
#     print(" positieve voorspellingen: ", np.sum(pred_up))
#     print(" aantal positieve voorspellingen die ook positief zijn: ", np.sum(pred_up[label_up]))
#     print("  === negatieve ===")
#     print(" negatieve labels: ", np.sum(~label_up))
#     print(" negatieve voorspellingen: ", np.sum(~pred_up))
#     print(" aantal negatieve voorspellingen die ook negatief zijn: ", np.sum(~pred_up[~label_up]))
          
#     acc = np.mean(pred_up == label_up)
#     return acc


def check_labelsvsprediction(path):
    """ Controleer wat er in de eerste nr-aantal pkl-bestanden staat"""

    predictionsdf = pd.read_csv(os.path.join(path, "pred.csv"))
    predictiondates = pd.unique(predictionsdf["dt"].values)
    # predictiondates = predictiondates[0:1] # het aantal dagen aanpassen
    # print(f"predictiondates: {predictiondates}")
    predictionsdf = predictionsdf[predictionsdf['dt'].isin(predictiondates)]  # Filter op de eerste x dagen
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
    # print(predictionsdf.head(5))

    predictions = torch.tensor(predictionsdf['score'].values, dtype=torch.float32).numpy()
    labels = torch.tensor(predictionsdf['true_score'].values, dtype=torch.float32).numpy()

    if task == 'regression':
        tllabels = np.tanh(np.log(labels+1))
    else:
        tllabels = (labels > 0).astype(float)
    print(len(labels), len(predictions))

    tllabel_stats = f"tanh log Labels - Gemiddelde: {np.mean(tllabels):.4f}, Std: {np.std(tllabels):.4f}, Max: {np.max(tllabels):.4f}, Min: {np.min(tllabels):.4f}"
    pred_stats = f"Voorspellingen - Gemiddelde: {np.mean(predictions):.4f}, Std: {np.std(predictions):.4f}, Max: {np.max(predictions):.4f}, Min: {np.min(predictions):.4f}"
    
    print(tllabel_stats)
    print(pred_stats)

    # # mae, mse , bce = evaluate_predictions(predictions, labels)
    # if task == 'regression':
    #     mae, mse, r2 = evaluate_reg_predictions(predictions, tllabels)
    # if task == 'classification':
    #     bce, acc= evaluate_reg_predictions(predictions, tllabels)
    # d, p = ks_2samp(labels, predictions)
    # print(f"KS-D distribution: {d:.4f} (p-value={p:.4g})")

    # print(f"MAE: {mae:.6f}")
    # print(f"MSE: {mse:.6f}")
    # print(f"RMSE: {np.sqrt(mse):.6f}")
    # print(f"R2: {r2:.6f}")
    # # print(f"BCE: {bce:.6f}")
    # print(f"Accuracy op richting: {acc:.2%}")

    for horizon, name in [(1, 'day1'), (5, 'day5'), (20, 'day20')]:
        horizon_df = predictionsdf.groupby("code").head(horizon)
        
        preds = horizon_df["score"].values
        if task == 'regression':
            labs = np.tanh(np.log(horizon_df["true_score"].values + 1))
        elif task == 'classification':
            labs = (horizon_df["true_score"].values > 0).astype(float)

        # print(preds[0:10])
        # print(labs[0:10])
        if task == 'regression':
            mae, mse, r2, WS = evaluate_reg_predictions(preds, labs)
            rmse = np.sqrt(mse)
            bce, acc, precision, recall, f1, Mcorc = None, None, None, None, None, None
        if task == 'classification':
            bce, acc, precision, recall, f1, Mcorc = evaluate_cla_predictions(preds, labs)
            bce = bce.item()
            mae, mse, r2, rmse, WS = None, None, None, None, None

        results.append({
            # "positive_threshold": p,
            # "negative_threshold": n,
            "input": input,
            "time": times,
            "task": task,
            "horizon": name,
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
            "r2": r2,
            "accuracy": acc,
            "precission": precision,
            "recall": recall,
            "F1": f1,
            "MCC": Mcorc,
            "bce": bce,
            "WS-dist": WS
        })

    return tllabels, predictions


# results = []

data_path = os.path.join(base_path, "data", "CSI300")
print(data_path)

label_path = os.path.join(data_path, "stock_labels.csv")
print(label_path)

# labelsdf = pd.read_csv(label_path, index_col=0)
# print(labelsdf['2024-10-09'].values)
# label_up = labelsdf['2024-10-09'].values > 0
# print(label_up)
# for predictionmap in os.listdir(os.path.join(data_path)):
    
#     if not predictionmap.startswith("prediction"):
#         continue

#     print(f"\npredictionmap: {predictionmap}")
#     parts = predictionmap[len("prediction_"):].split("_")

#     if (parts[0] == "random1") or (parts[0] == "random2") or (parts[0] == "random3"):
#         print("niet geselecteerd: ",predictionmap)
#         print(parts)
#         continue

#     print("wel geselecteerd: ",predictionmap)
#     print(parts)
#     input = ""
#     task = ""
#     times = ""

#     if len(parts) == 2:
#         input = parts[0]
#         task = "regression"
#         times = parts[1]

#     elif len(parts) == 3:
#         input = parts[0]
#         task = "classification"
#         times = parts[2]


#     print(f"Map: {predictionmap} → input: {input}, time: {times}, task: {task}")

#     prediction_path = os.path.join(data_path, predictionmap)
#     print(prediction_path)
#     labels, corrpredictions = check_labelsvsprediction(prediction_path)


#     # distribution(corrpredictions, labels, dynamipredictions)

# results_df = pd.DataFrame(results)
# results_df.to_csv(os.path.join(data_path, "results_alltimes.csv"), index=False)

# CSV inlezen
# df = pd.read_csv(os.path.join(data_path, "results_alltimes.csv"))

# # Drop de task-kolom
# df = df.drop(columns=["task"])

# # Groeperen per unieke combinatie en aggregatie toepassen
# df_combined = df.groupby(["input", "time", "horizon"], as_index=False).agg({
#     "mae": "max",  # max omdat maar één van de twee rijen een waarde heeft
#     "mse": "max",
#     "rmse": "max",
#     "r2": "max",
#     "accuracy": "max",
#     "precission": "max",
#     "recall": "max",
#     "F1": "max",
#     "MCC": "max",
#     "bce": "max",
#     "WS-dist": "max"
# })

# time_order = [-120, -100, -80, -60, -40, -20, 0]
# input_order = ["corr", "DSE"]
# horizon_order = ["day1", "day5", "day20"]

# df_combined["time"] = pd.Categorical(df_combined["time"], categories=time_order, ordered=True)
# df_combined["input"] = pd.Categorical(df_combined["input"], categories=input_order, ordered=True)
# df_combined["horizon"] = pd.Categorical(df_combined["horizon"], categories=horizon_order, ordered=True)

# # df_combined = df_combined.sort_values(by=["time", "input", "horizon"])
# df_combined = df_combined.sort_values(by=["horizon", "input", "time"])
# # Opslaan of printen
# df_combined.to_csv(os.path.join(data_path, "results_alltimes_combined.csv"), index=False)




# # results zijn er al
# results_df = pd.read_csv(os.path.join(data_path, "results.csv"))
# print(results_df.head())

# # filtered_df = results_df[(results_df['threshold'] >= 0.3) & (results_df['threshold'] <= 0.8)]
# # filtered_df = filtered_df[filtered_df['horizon'].isin(['day5', 'day20'])]

# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

# sns.lineplot(data=results_df, x="positive_threshold", y="rmse", hue="horizon", style="negative_threshold", markers=True, dashes=False, ax=ax1)
# ax1.set_title("RMSE per negative and positive")
# ax1.set_xlabel("Positive")
# ax1.set_ylabel("RMSE")
# ax1.grid(True)

# sns.lineplot(data=results_df, x="positive_threshold", y="r2", hue="horizon", style="negative_threshold", markers=True, dashes=False, ax=ax2)
# ax2.set_title("r2 per negative and positive")
# ax2.set_xlabel("Positive")
# ax2.set_ylabel("r2")
# ax2.grid(True)

# plt.tight_layout()
# plt.show()



# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Data inladen
# df = pd.read_csv(os.path.join(data_path, "results_combined.csv"))
# df["horizon"] = pd.Categorical(df["horizon"], categories=["day1", "day5", "day20"], ordered=True)
# # 1. Staafdiagram: Gemiddelde MAE per horizon en encoder
# plt.figure(figsize=(10, 5))
# sns.barplot(data=df, x="horizon", y="rmse", hue="input", ci=None)
# plt.title("Gemiddelde RMSE: GRU vs. TE per horizon")
# plt.ylabel("RMSE (lager = beter)")
# plt.show()

# # 2. Boxplot: Spreiding van R2-scores per model
# plt.figure(figsize=(10, 5))
# sns.boxplot(data=df, x="horizon", y="r2", hue="input")
# plt.title("Spreiding van R²-scores per horizon")
# plt.ylabel("R² (hoger = beter)")
# plt.show()

# # 3. Lijngrafiek: Trend in Accuracy over horizons
# plt.figure(figsize=(10, 5))
# sns.lineplot(data=df, x="horizon", y="accuracy", hue="input", ci=None, marker="o")
# plt.title("Accuracy over verschillende horizons")
# plt.ylabel("Accuracy (hoger = beter)")
# plt.show()

# # 4. Samenvattende tabel (gemiddelden per groep)
# summary_table = df.groupby(["input", "horizon"]).agg({
#     "mae": "mean",
#     "rmse": "mean",
#     "r2": "mean",
#     "accuracy": "mean",
#     "bce": "mean"
# }).round(3)
# print(summary_table)


# import pandas as pd
# import matplotlib.pyplot as plt

# Laad de data
df = pd.read_csv(os.path.join(data_path, "results_alltimes_combined.csv"))
df = df.sort_values(["horizon", "input", "time"])  # Belangrijk: sorteer op tijd!

# time_mapping = {-120: 0, -100: 1, -80: 2, -60: 3, -40: 4, -20: 5, 0: 6}
# df['time'] = df['time'].map(time_mapping)

# Horizons en configuratie
horizons = ['day1', 'day5', 'day20']
fig, axes = plt.subplots(3, 2, figsize=(18, 20))
plt.subplots_adjust(hspace=0.4, wspace=0.3)

for i, horizon in enumerate(horizons):
    subset = df[df['horizon'] == horizon]
    
    # --- Regressie metrics ---
    ax1 = axes[i, 0]
    ax1_r2 = ax1.twinx()
    
    for method, color in zip(['corr', 'DSE'], ['blue', 'red']):
        data = subset[subset['input'] == method].sort_values('time')
        print(data[['time', 'mae']])
        # Explicitly set drawstyle en markeringen
        ax1.plot(data['time'], data['mae'], color=color, linestyle='--', 
                marker='o', label=f'{method} MAE')
        ax1.plot(data['time'], data['rmse'], color=color, linestyle=':', 
                marker='s', label=f'{method} RMSE')
        ax1_r2.plot(data['time'], data['r2'], color=color, linestyle='-', 
                   marker='^', label=f'{method} R2')
    
    # --- Classificatie metrics ---
    ax2 = axes[i, 1]
    ax2_acc_rec = ax2.twinx()
    
    for method, color in zip(['corr', 'DSE'], ['blue', 'red']):
        data = subset[subset['input'] == method].sort_values('time')
        
        ax2.plot(data['time'], data['bce'], color=color, linestyle='-', 
               marker='o', label=f'{method} BCE')
        ax2_acc_rec.plot(data['time'], data['accuracy'], color=color, linestyle='--', 
                        marker='s', label=f'{method} Accuracy')
        ax2_acc_rec.plot(data['time'], data['MCC'], color=color, linestyle=':', 
                        marker='^', label=f'{method} MCC')
    
    ax1.set_title(f'Regressie Metrics ({horizon})', fontsize=12)
    ax1.set_xlabel('Time', fontsize=10)
    ax1.set_ylabel('MAE / RMSE', fontsize=10)
    ax1_r2.set_ylabel('R2', fontsize=10)
    ax1.grid(False)
    
    # Verzamel handvatten voor de legenda
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_r2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=8)

    ax2.set_title(f'Classificatie Metrics ({horizon})', fontsize=12)
    ax2.set_xlabel('Time', fontsize=10)
    ax2.set_ylabel('BCE', fontsize=10)
    ax2_acc_rec.set_ylabel('Accuracy / MCC', fontsize=10)
    ax2.grid(False)
    
    # Legenda voor classificatie
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_acc_rec.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=8)

plt.tight_layout()
plt.show()