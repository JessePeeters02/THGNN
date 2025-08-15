import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score
from scipy.stats import wasserstein_distance, ks_2samp

# region Pad configuratie
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Huidige scriptmap
print(base_path)
""" uncomment de database die je wilt gebruiken"""
# database = "CSI300"
database = "S&P500"
# database = "NASDAQ_batches_5_200", "batch_1"
# database = "NASDAQ_batches_5_200", "batch_2"
# database = "NASDAQ_batches_5_200", "batch_3"
# database = "NASDAQ_batches_5_200", "batch_4"
# database = "NASDAQ_batches_5_200", "batch_5"
# database = "testbatch1"
# database = "testbatch2"
# database = "testbatch_mini"
if isinstance(database, str):
    database = [database]
data_path = os.path.join(base_path, "data", *database)
print(data_path)
# prediction_path = os.path.join(data_path, "model_saved_rolingwindow_test")
# prediction_path = os.path.join(data_path, "model_saved_rolingwindow_corr")
# prediction_path = os.path.join(data_path, "model_saved_rolingwindow_noBeta")
# prediction_path = os.path.join(data_path, "model_saved_rolingwindow_STATIC_t1")
# prediction_path = os.path.join(data_path, "model_saved_rolingwindow_onlycosine")
# prediction_path = os.path.join(data_path, "model_saved_rolingwindow_cosineDSC")
# prediction_path = os.path.join(data_path, "model_saved_rolingwindow_DSE")
# nieuwe paden
# prediction_path = os.path.join(data_path, "model_saved_rolingwindow_corr_t2")
# prediction_path = os.path.join(data_path, "model_saved_rolingwindow_onlycosine_t2")
# prediction_path = os.path.join(data_path, "model_saved_rolingwindow_cosineDSC_t2")
# prediction_path = os.path.join(data_path, "model_saved_rolingwindow_corrDSC_t2")
# prediction_path = os.path.join(data_path, "model_saved_rolingwindow_DSEcorr_t2")
# prediction_path = os.path.join(data_path, "model_saved_rolingwindow_DSE_t11")
# prediction_path = os.path.join(data_path, "model_saved_rolingwindow_STATIC_t2")
prediction_path = os.path.join(data_path, "model_saved_rolingwindow_STATICcorr_t1")
output_path = os.path.join(data_path, "results")
# endregion

# region distributies
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

def plot_distributions(predictions, labels):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Combineer data voor consistente x-as limieten
    combined = np.concatenate([predictions, labels])
    x_min, x_max = np.min(combined), np.max(combined)
    
    # Plot voorspellingen
    sns.histplot(predictions, bins=50, kde=True, color='blue', ax=ax1)
    ax1.set_title("Distributie van voorspellingen")
    ax1.set_xlim(x_min, x_max)
    
    # Plot labels
    sns.histplot(labels, bins=50, kde=True, color='orange', ax=ax2)
    ax2.set_title("Distributie van labels")
    ax2.set_xlim(x_min, x_max)
    
    plt.tight_layout()
    plt.show()
# endregion

# region voorspellingen csv maken
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

def check_labelsvsprediction():
    predictionsdf = pd.read_csv(os.path.join(prediction_path, "pred.csv"))
    predictions = predictionsdf['score'].values
    labels = predictionsdf['label'].values
    labels = np.tanh(np.log(labels + 1))
    predictionsdf["dt"] = pd.to_datetime(predictionsdf["dt"])

    print(len(labels), len(predictions))
    tllabel_stats = f"Labels - Gemiddelde: {np.mean(labels):.4f}, Std: {np.std(labels):.4f}, Max: {np.max(labels):.4f}, Min: {np.min(labels):.4f}"
    pred_stats = f"Voorspellingen - Gemiddelde: {np.mean(predictions):.4f}, Std: {np.std(predictions):.4f}, Max: {np.max(predictions):.4f}, Min: {np.min(predictions):.4f}"
    print("Statistieken:")
    print(tllabel_stats)
    print(pred_stats)

    mae, mse, r2, WS = evaluate_reg_predictions(predictions, labels)
    d, p = ks_2samp(labels, predictions)
    print(f"KS-D distribution: {d:.4f} (p-value={p:.4g})")

    print(f"MAE: {mae:.6f}")
    print(f"MSE: {mse:.6f}")
    print(f"RMSE: {np.sqrt(mse):.6f}")
    print(f"R2: {r2:.6f}")
    # print(f"BCE: {bce:.6f}")
    # print(f"Accuracy op richting: {acc:.2%}")

    results = []
    for day, group in predictionsdf.groupby("dt"):
        preds = group["score"].values
        labs = group["label"].values
        mae, mse, r2, WS = evaluate_reg_predictions(preds, labs)
        d, p = ks_2samp(labs, preds)
        rmse = np.sqrt(mse)

        results.append({
            "dt": day.strftime("%Y-%m-%d"),
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
            "r2": r2,
            "WS-dist": WS,
            "KS-d": d,
            "KS-p": p
        })

    mae, mse, r2, WS = evaluate_reg_predictions(predictions, labels)
    rmse = np.sqrt(mse)
    d, p = ks_2samp(labels, predictions)
    results.append({
        "dt": "OVERALL",
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
        "WS-dist": WS,
        "KS-d": d,
        "KS-p": p
    })
    plot_distributions(predictions, labels)
    result_df = pd.DataFrame(results)
    os.makedirs(output_path, exist_ok=True)
    prediction_type = os.path.basename(prediction_path).replace("model_saved_rolingwindow_", "")
    save_name = os.path.join(output_path, f"results_{prediction_type}.csv")
    result_df.to_csv(save_name, index=False)
    print(f"Dagresultaten opgeslagen naar: {save_name}")

    return labels, predictions
# endregion

check_labelsvsprediction()


