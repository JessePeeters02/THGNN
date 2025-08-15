import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from scipy.stats import ttest_rel, wilcoxon

# region configuratie

base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Huidige scriptmap
print(base_path)

""" uncomment de database die je wilt gebruiken"""
database = "CSI300"
# database = "S&P500"
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

input_path = os.path.join(data_path, "results")

"""select de metrics en modellen die je wilt vergelijken"""
metrics = ["rmse", "mae", "r2"]                  # pas aan: "mae", "mse", "rmse", "r2", "WS-dist", "KS-d", "KS-p"

# models = ["corr", "onlycosine", "STATIC_t1", "cosineDSC_t1", "DSE_t1"]
models = ["corr", "STATICcorr_t1", "corrDSC_t1", "DSEcorr_t1", "DSEcorr_t11", "DSEcorr_t12"]
# models = ["corr_t2", "onlycosine_t2", "STATIC_t2", "cosineDSC_t2", "DSE_t2"]
# models = ["corr_t2", "STATICcorr_t2", "corrDSC_t2", "DSEcorr_t2"]


# models = ["corr", "onlycosine", "STATICcorr_t1", "corrDSC_t1", "DSEcorr_t1", "DSE_t1"]
# models = ["corr", "onlycosine", "DSEcorr_t1", "DSE_t1"]
# models = ["corr_t2", "onlycosine_t2", "DSEcorr_t2", "DSE_t2"]


"""Plot-opties"""
use_seaborn_theme = True                     # zet op False als je pure matplotlib wil
figsize = (12, 6)
save_png = False
output_path = os.path.join(base_path, "plots")
os.makedirs(output_path, exist_ok=True)

# endregion


# region plots
def line_plots(xtick_rotation: int = 60):

    # -- 1) X-as labels afleiden uit het eerste model --
    first = models[0]
    df0 = pd.read_csv(os.path.join(input_path, f"results_{first}.csv"))
    df0["dt"] = df0["dt"].astype(str)

    date_mask0 = df0["dt"].str.upper() != "OVERALL"
    date_labels = df0.loc[date_mask0, "dt"].tolist()      # alle dagen in volgorde
    x_overall   = len(date_labels)                         # index voor OVERALL
    x_labels    = date_labels + ["OVERALL"]

    # offsets voor OVERALL-punten zodat ze niet overlappen
    n = len(models)
    offsets = np.linspace(-0.5, 0.5, n) if n > 1 else [0.0]

    # -- 2) Plot per metric --
    for metric in metrics:
        plt.figure(figsize=(12, 6))

        for i, mo in enumerate(models):
            df = pd.read_csv(os.path.join(input_path, f"results_{mo}.csv"))
            df["dt"] = df["dt"].astype(str)

            # lijn: alle dagen
            date_mask = df["dt"].str.upper() != "OVERALL"
            y_line = df.loc[date_mask, metric].to_numpy()
            x_line = range(len(date_labels))  # aanname: zelfde volgorde/ aantal dagen
            plt.plot(x_line, y_line, marker="o", linewidth=1.8, label=mo)

            # los punt: OVERALL (laatste rij)
            y_overall = y_line.mean()
            # y_overall = df.loc[~date_mask, metric].iloc[0]
            plt.scatter(x_overall + offsets[i], y_overall,
                        marker="D", s=80, edgecolors="black", linewidths=0.6, zorder=5)

        # simpele x-as: labels schuin
        plt.xticks(range(len(x_labels)), x_labels, rotation=xtick_rotation, ha="right")

        # visuele scheiding voor OVERALL
        plt.axvline(x_overall - 0.5, linestyle="--", alpha=0.5)

        plt.xlabel("Date")
        plt.ylabel(metric)
        plt.title(f"{metric} over time per model")
        plt.legend()
        plt.tight_layout()

        if save_png:
            out_dir = os.path.join(input_path, "..", "plots")
            os.makedirs(out_dir, exist_ok=True)
            plt.savefig(os.path.join(out_dir, f"{metric}.png"), dpi=160, bbox_inches="tight")
            plt.close()
        else:
            plt.show()
# endregion

# region plots
def significantieverschil(test_type="wilcoxon", alpha=0.05):
    """
    Voer pairwise significantietesten uit op daggemiddelde metrics van de modellen.

    Parameters
    ----------
    test_type : str
        "ttest"    -> Paired t-test
        "wilcoxon" -> Wilcoxon signed-rank test
    alpha : float
        Significantie-niveau (bv. 0.05)

    Returns
    -------
    results_df : pd.DataFrame
        DataFrame met metric, model_a, model_b, p_value en significant-boolean
    """

    results = []

    # alle data per model inladen
    model_data = {}
    for mo in models:
        df = pd.read_csv(os.path.join(input_path, f"results_{mo}.csv"))
        # enkel de rijen met echte dagen (geen OVERALL)
        df = df[df["dt"].str.upper() != "OVERALL"].reset_index(drop=True)
        model_data[mo] = df

    # pairwise combinaties
    for metric in metrics:
        for m1, m2 in combinations(models, 2):
            y1 = model_data[m1][metric].to_numpy()
            y2 = model_data[m2][metric].to_numpy()

            # Kies test
            if test_type == "ttest":
                stat, pval = ttest_rel(y1, y2)
            elif test_type == "wilcoxon":
                stat, pval = wilcoxon(y1, y2)
            else:
                raise ValueError("test_type moet 'ttest' of 'wilcoxon' zijn")

            results.append({
                "metric": metric,
                "model_a": m1,
                "model_b": m2,
                "p_value": pval,
                "significant": pval < alpha
            })

    results_df = pd.DataFrame(results)
    return results_df
# endregion

# region heatmap significantieverschil
def plot_significance_heatmap(df_sig, metric):
    """Maak een heatmap van p-waarden voor een bepaalde metric."""
    df_metric = df_sig[df_sig["metric"] == metric]
    all_models = sorted(set(df_metric["model_a"]) | set(df_metric["model_b"]))

    # matrix vullen
    mat = pd.DataFrame(np.nan, index=all_models, columns=all_models)
    for _, row in df_metric.iterrows():
        mat.loc[row["model_a"], row["model_b"]] = row["p_value"]
        mat.loc[row["model_b"], row["model_a"]] = row["p_value"]

    plt.figure(figsize=(8, 6))
    sns.heatmap(mat, annot=True, fmt=".3f", cmap="coolwarm_r",
                cbar_kws={"label": "p-value"}, linewidths=0.5)
    plt.title(f"P-waarden (Wilcoxon) voor metric: {metric}")
    plt.show()
# endregion



# region functies en mappen kiezen

line_plots()
df_sig = significantieverschil(test_type="wilcoxon", alpha=0.05)
print(df_sig)
plot_significance_heatmap(df_sig, metric="rmse")
plot_significance_heatmap(df_sig, metric="mae")
plot_significance_heatmap(df_sig, metric="r2")

# endregion















# region oude code
""" oude code van grafieken maken, kan nog handig zijn ter inspiratie
        # distribution(corrpredictions, labels, dynamipredictions)

def nasdaq_batches():
    for batchmap in os.listdir(os.path.join(data_path)):
        if not batchmap.startswith("batch"):
            continue

        print(f"\nbatchmap: {batchmap}")

        for predictionmap in (os.path.join(data_path, batchmap)):
            if not predictionmap.startswith("prediction_"):
                continue
            parts = predictionmap[len("prediction_"):].split("_")

            if (parts[0] == "random1") or (parts[0] == "random2") or (parts[0] == "random3"):
                print("niet geselecteerd: ",batchmap)
                print(parts)
                continue

            print("wel geselecteerd: ",batchmap)
            print(parts)
            input = ""
            task = ""
            times = ""

            if len(parts) == 2:
                input = parts[0]
                task = "regression"
                times = parts[1]

            elif len(parts) == 3:
                input = parts[0]
                task = "classification"
                times = parts[2]


            print(f"Map: {batchmap} → input: {input}, time: {times}, task: {task}")

            prediction_path = os.path.join(data_path, batchmap)
            print(prediction_path)
            labels, corrpredictions = check_labelsvsprediction(prediction_path)



results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(data_path, "results_alltimes.csv"), index=False)

# CSV inlezen
df = pd.read_csv(os.path.join(data_path, "results_alltimes.csv"))

# Drop de task-kolom
df = df.drop(columns=["task"])

# Groeperen per unieke combinatie en aggregatie toepassen
df_combined = df.groupby(["input", "time", "horizon"], as_index=False).agg({
    "mae": "max",  # max omdat maar één van de twee rijen een waarde heeft
    "mse": "max",
    "rmse": "max",
    "r2": "max",
    "accuracy": "max",
    "precission": "max",
    "recall": "max",
    "F1": "max",
    "MCC": "max",
    "bce": "max",
    "WS-dist": "max"
})

time_order = [-120, -100, -80, -60, -40, -20, 0]
input_order = ["corr", "DSE"]
horizon_order = ["day1", "day5", "day20"]

df_combined["time"] = pd.Categorical(df_combined["time"], categories=time_order, ordered=True)
df_combined["input"] = pd.Categorical(df_combined["input"], categories=input_order, ordered=True)
df_combined["horizon"] = pd.Categorical(df_combined["horizon"], categories=horizon_order, ordered=True)

# df_combined = df_combined.sort_values(by=["time", "input", "horizon"])
df_combined = df_combined.sort_values(by=["horizon", "input", "time"])
# Opslaan of printen
df_combined.to_csv(os.path.join(data_path, "results_alltimes_combined.csv"), index=False)




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

# Data inladen
df = pd.read_csv(os.path.join(data_path, "results_combined.csv"))
df["horizon"] = pd.Categorical(df["horizon"], categories=["day1", "day5", "day20"], ordered=True)
# 1. Staafdiagram: Gemiddelde MAE per horizon en encoder
plt.figure(figsize=(10, 5))
sns.barplot(data=df, x="horizon", y="rmse", hue="input", ci=None)
plt.title("Gemiddelde RMSE: GRU vs. TE per horizon")
plt.ylabel("RMSE (lager = beter)")
plt.show()

# 2. Boxplot: Spreiding van R2-scores per model
plt.figure(figsize=(10, 5))
sns.boxplot(data=df, x="horizon", y="r2", hue="input")
plt.title("Spreiding van R²-scores per horizon")
plt.ylabel("R² (hoger = beter)")
plt.show()

# 3. Lijngrafiek: Trend in Accuracy over horizons
plt.figure(figsize=(10, 5))
sns.lineplot(data=df, x="horizon", y="accuracy", hue="input", ci=None, marker="o")
plt.title("Accuracy over verschillende horizons")
plt.ylabel("Accuracy (hoger = beter)")
plt.show()

# 4. Samenvattende tabel (gemiddelden per groep)
summary_table = df.groupby(["input", "horizon"]).agg({
    "mae": "mean",
    "rmse": "mean",
    "r2": "mean",
    "accuracy": "mean",
    "bce": "mean"
}).round(3)
print(summary_table)


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

"""
# endregion