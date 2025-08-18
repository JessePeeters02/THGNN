import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# region configuratie

base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(base_path)
database = "CSI300"
# database = "S&P500"

data_path = os.path.join(base_path, "data", database)
print(data_path)

input_path = os.path.join(data_path, "results")

metrics = ["rmse", "mae", "r2"]                  # pas aan: "mae", "mse", "rmse", "r2", "WS-dist", "KS-d", "KS-p"...

models = ["CS", "SSA-CS", "DSC-CS", "DSE-CS"]
# models = ["PC", "SSA-PC", "DSC-PC", "DSE-PC"]
# models = ["PC", "CS", "DSE-PC", "DSE-CS"]

use_seaborn_theme = True
save_png = True
output_path = os.path.join(base_path, "plots")
os.makedirs(output_path, exist_ok=True)

# endregion


# region plots
def line_plots(xtick_rotation: int = 45):
    plt.rcParams.update({'font.size': 20}) 
    SMALL_SIZE = 20*0.9 
    MEDIUM_SIZE = 24*0.9 
    BIGGER_SIZE = 28*0.9

    plt.rc('font', size=SMALL_SIZE)
    plt.rc('axes', titlesize=BIGGER_SIZE) 
    plt.rc('axes', labelsize=MEDIUM_SIZE)
    plt.rc('xtick', labelsize=SMALL_SIZE)
    plt.rc('ytick', labelsize=SMALL_SIZE)
    plt.rc('legend', fontsize=SMALL_SIZE) 
    plt.rc('figure', titlesize=BIGGER_SIZE)

    # Different markers for each model
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', 'h', '8']

    first = models[0]
    df0 = pd.read_csv(os.path.join(input_path, f"results_{first}.csv"))
    df0["dt"] = df0["dt"].astype(str)

    date_mask0 = df0["dt"].str.upper() != "OVERALL"
    date_labels = df0.loc[date_mask0, "dt"].tolist()
    x_overall = len(date_labels)

    for metric in metrics:
        plt.figure(figsize=(16, 6))

        all_values = []
        overall_values = []
        for mo in models:
            df = pd.read_csv(os.path.join(input_path, f"results_{mo}.csv"))
            df["dt"] = df["dt"].astype(str)
            daily_mask = df["dt"].str.upper() != "OVERALL"
            all_values.extend(df.loc[daily_mask, metric].tolist())
            overall_values.append(df.loc[~daily_mask, metric].iloc[0])

        for i, mo in enumerate(models):
            df = pd.read_csv(os.path.join(input_path, f"results_{mo}.csv"))
            df["dt"] = df["dt"].astype(str)
            
            daily_mask = df["dt"].str.upper() != "OVERALL"
            daily_values = df.loc[daily_mask, metric].to_numpy()
            overall_value = df.loc[~daily_mask, metric].iloc[0]
            
            plt.plot(range(len(date_labels)), daily_values, 
                    marker=markers[i % len(markers)], 
                    linewidth=1.8, label=mo, markersize=5,
                    markeredgewidth=1)
            
            bar_width = 1.2 / len(models)
            bar_pos = x_overall -0.3 + (i * bar_width)
            plt.bar(bar_pos, overall_value, width=bar_width, 
                   alpha=1, color=plt.gca().lines[-1].get_color())

        min_val = min(min(all_values), min(overall_values))
        max_val = max(max(all_values), max(overall_values))
        margin = (max_val - min_val) * 0.05
        
        y_min = 0 if min_val < 0 else min_val - margin
        plt.ylim(y_min, max_val + margin)

        plt.xticks(list(range(len(date_labels))) + [x_overall], 
                  date_labels + ["OVERALL"], 
                  rotation=xtick_rotation, ha="right")
        
        plt.xlabel("Date")
        plt.ylabel(metric)
        plt.title(f"{database[0]}: {metric} over time per model")
        plt.legend()
        plt.grid(False)
        plt.tight_layout()

        if save_png:
            plt.savefig(os.path.join(output_path, f"{database}_{metric}_{models}.png"), dpi=160, bbox_inches=None)
            # plt.show()
            plt.close()
        else:
            plt.show()
# endregion

# region overall plot

def overall_barplots():
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.5 * len(metrics), 6))

    if len(metrics) == 1:
        axes = [axes]

    for i, metric in enumerate(metrics):
        overall_values = []
        for mo in models:
            df = pd.read_csv(os.path.join(input_path, f"results_{mo}.csv"))
            df["dt"] = df["dt"].astype(str)
            daily_mask = df["dt"].str.upper() != "OVERALL"
            overall_value = df.loc[~daily_mask, metric].iloc[0]
            overall_values.append(overall_value)

        x = range(len(models))
        bars = axes[i].bar(x, overall_values, tick_label=models)

        for bar, color in zip(bars, plt.cm.tab10.colors):
            bar.set_color(color)

        axes[i].set_ylabel(metric)
        axes[i].set_title(f"Overall {metric}")

    plt.tight_layout()

    if save_png:
        plt.savefig(os.path.join(output_path, f"{database}_overall_barplots.png"), dpi=160)
        plt.close()
    else:
        plt.show()


# endregion


# region functies en mappen kiezen

line_plots()
# overall_barplots()

# endregion
