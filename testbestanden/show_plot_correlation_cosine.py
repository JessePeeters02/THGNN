import torch
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
import os
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
from fastdtw import fastdtw
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# alle paden relatief aanmaken
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(base_path, "data", "testbatch2")
daily_data_log_path = os.path.join(data_path, "normaliseddailydata_log")
daily_data_path = os.path.join(data_path, "normaliseddailydata")
raw_data_path = os.path.join(data_path, "stockdata")

# Hyperparameters
feature_cols = ['Open', 'High', 'Low', 'Close']#, 'Volume']
window = 20
restrict_last_n_days= 20 # None voor alles of 20 voor 20-day window


def cosine_similarity(vec1, vec2):
    # if vec1.norm() == 0 or vec2.norm() == 0:
    #     return 0
    vec1 = torch.tensor(vec1, dtype=torch.float32)
    vec2 = torch.tensor(vec2, dtype=torch.float32)
    return F.cosine_similarity(vec1.unsqueeze(0), vec2.unsqueeze(0)).item()

def dtw_similarity(series1, series2):
    distance, _ = fastdtw(series1, series2)
    return 1 / (1 + distance)

def pearson_correlation(vec1, vec2):
    return np.corrcoef(vec1, vec2)[0, 1]

def load_all_stocks(stock_data_path):
    all_stock_data = []
    for file in tqdm(os.listdir(stock_data_path), desc="Loading normalised data"):
        if file.endswith('.csv'):
            df = pd.read_csv(os.path.join(stock_data_path, file))
            all_stock_data.append(df[['Date', 'Stock', 'Open', 'High', 'Low', 'Close']])
    all_stock_data = pd.concat(all_stock_data, ignore_index=True)
    print(all_stock_data.head()) # kleine test om te zien of data deftig is ingeladen

        # Enkel laatste X dagen
    if restrict_last_n_days is not None:
        all_dates = sorted(all_stock_data['Date'].unique())
        last_dates = all_dates[-restrict_last_n_days:-(restrict_last_n_days-window)]
        print(last_dates)
        all_stock_data = all_stock_data[all_stock_data['Date'].isin(last_dates)]

    print(all_stock_data.head())  # test of data deftig is

    return all_stock_data

def load_raw_stocks(raw_stock_path):
    raw_files = [f for f in os.listdir(raw_stock_path) if f.endswith('.csv')]
    raw_data = {}
    for file in tqdm(raw_files, desc="Loading raw data for label creation"):
        stock_name = file.split('.')[0]
        df = pd.read_csv(os.path.join(raw_stock_path, file), parse_dates=['Date'])
        if restrict_last_n_days is not None:
            all_dates = sorted(df['Date'].unique())
            last_dates = all_dates[-restrict_last_n_days:-(restrict_last_n_days-window)]
            df = df[df['Date'].isin(last_dates)]
        raw_data[stock_name] = df
    print(last_dates)
    return raw_data


stock_data = load_all_stocks(daily_data_path)
stock_log_data = load_all_stocks(daily_data_log_path)
raw_data = load_raw_stocks(raw_data_path)
all_dates = sorted(stock_data['Date'].unique())
unique_stocks = sorted(stock_data['Stock'].unique())

close_prices_nor = []
close_prices_log = []
close_prices_raw = []
close_prices_raw_norm = []

feature_vectors_all = []
feature_vectors = {feat: [] for feat in feature_cols}

for stock in unique_stocks:
    df_nor = stock_data[stock_data['Stock'] == stock].sort_values('Date')
    close_prices_nor.append(df_nor['Close'].values)
    
    df_log = stock_log_data[stock_log_data['Stock'] == stock].sort_values('Date')
    close_prices_log.append(df_log['Close'].values)

    if stock in raw_data:
        df_raw = raw_data[stock].sort_values('Date')
        close_prices_raw.append(df_raw['Close'].values)
    else:
        print(f"Warning: {stock} not found in raw_data dict.")
        continue

    for feat in feature_cols:
        feature_vectors[feat].append(df_nor[feat].values)
    feature_vectors_all.append(np.concatenate([df_nor[feat].values for feat in feature_cols]))

for stock in unique_stocks:
    if stock in raw_data:
        df_raw_nor = raw_data[stock].sort_values('Date')
        close_prices_raw_norm.append(scaler.fit_transform(df_raw_nor['Close'].values.reshape(-1, 1)).flatten())
    else:
        print(f"Warning: {stock} not found in raw_data dict.")
        continue

close_prices_nor = np.array(close_prices_nor)
close_prices_log = np.array(close_prices_log)
close_prices_raw = np.array(close_prices_raw)
close_prices_raw_norm = np.array(close_prices_raw_norm)

# Sanity check
print(f"Genormaliseerde shape: {close_prices_nor.shape}")
print(f"log return shape: {close_prices_log.shape}")
print(f"Ruwe shape: {close_prices_raw.shape}")
print(f"Genormaliseerde shape zonder rolling window: {close_prices_raw_norm.shape}")

# Verzamel cosine similarities en correlaties
cos_raws, cor_raws = [], []
# cos_nors, cor_nors = [], []
# cos_logs, cor_logs = [], []
# dtw_logs = []
# cos_raw_norms, cor_raw_norms = [], []

for i in range(close_prices_raw.shape[0]):
    for j in range(i+1, close_prices_raw.shape[0]):
        vec_raw_i, vec_raw_j = close_prices_raw[i], close_prices_raw[j]
#         vec_nor_i, vec_nor_j = close_prices_nor[i], close_prices_nor[j]
#         vec_log_i, vec_log_j = close_prices_log[i], close_prices_log[j]
        # vec_raw_norm_i, vec_raw_norm_j = close_prices_raw_norm[i], close_prices_raw_norm[j]
        
#         if (len(vec_raw_i) != len(vec_raw_j)) or (len(vec_nor_i) != len(vec_nor_j)) or (len(vec_raw_norm_i) != len(vec_raw_norm_j)) or (len(vec_log_i) != len(vec_log_j)):
#             print("dees is zware error")
#             continue
        
#         # Bereken alle metrics
#         cos_raw = cosine_similarity(vec_raw_i, vec_raw_j)
        cos_raw = sklearn_cosine_similarity(vec_raw_i.reshape(1, -1), vec_raw_j.reshape(1, -1))[0,0]
        cor_raw = pearson_correlation(vec_raw_i, vec_raw_j)
#         # cos_nor = cosine_similarity(vec_nor_i, vec_nor_j)
#         cos_nor = sklearn_cosine_similarity(vec_nor_i.reshape(1, -1), vec_nor_j.reshape(1, -1))[0,0]
#         cor_nor = pearson_correlation(vec_nor_i, vec_nor_j)
#         # cos_log = cosine_similarity(vec_log_i, vec_log_j)
#         cos_log = sklearn_cosine_similarity(vec_log_i.reshape(1, -1), vec_log_j.reshape(1, -1))[0,0]
#         cor_log = pearson_correlation(vec_log_i, vec_log_j)
#         # cos_raw_norm = cosine_similarity(vec_raw_norm_i, vec_raw_norm_j)
        # cos_raw_nor = sklearn_cosine_similarity(vec_raw_norm_i.reshape(1, -1), vec_raw_norm_j.reshape(1, -1))[0,0]
#         cor_raw_nor = pearson_correlation(vec_raw_norm_i, vec_raw_norm_j)
#         dtw_log = dtw_similarity(vec_log_i, vec_log_j)
#         if any(np.isnan(x) for x in [cos_raw, cor_raw, cos_nor, cor_nor, cos_raw_nor, cor_raw_nor, cos_log, cor_log, dtw_log]):
#             print("zware error dit hier mag niet!")
#             continue

        cos_raws.append(cos_raw)
        cor_raws.append(cor_raw)
#         cos_nors.append(cos_nor)
#         cor_nors.append(cor_nor)
#         cos_logs.append(cos_log)
#         cor_logs.append(cor_log)
#         cos_raw_norms.append(cos_raw_nor)
#         cor_raw_norms.append(cor_raw_nor)
#         dtw_logs.append(dtw_log)
        correlations = []
        for feat in feature_cols:
            vec_i = feature_vectors[feat][i]
            vec_j = feature_vectors[feat][j]
            if len(vec_i) != len(vec_j):
                continue  # of raise error
            correlations.append(pearson_correlation(vec_i, vec_j))
            
        if correlations:  # check voor lege lijst
            mean_correlation = np.mean(correlations)
            cor_raws.append(mean_correlation)

cos_all_features = []
cosine_per_feature = {feat: [] for feat in feature_cols}

for i in tqdm(range(len(unique_stocks)), desc='calculating everything'):
    for j in range(i + 1, len(unique_stocks)):
        # print(f" i: j: {i,j}, {len(feature_vectors_all)}")
        v_all_i = feature_vectors_all[i]
        v_all_j = feature_vectors_all[j]

        cos_all = sklearn_cosine_similarity(v_all_i.reshape(1, -1), v_all_j.reshape(1, -1))[0, 0]

        cos_all_features.append(cos_all)

        for feat in feature_cols:
            vec_feat_i, vec_feat_j = feature_vectors[feat][i], feature_vectors[feat][j]
            if len(vec_feat_i) != len(vec_feat_j):
                continue
            cos_feat = sklearn_cosine_similarity(vec_feat_i.reshape(1, -1), vec_feat_j.reshape(1, -1))[0, 0]
            cosine_per_feature[feat].append(cos_feat)


# Plotten
fig, axes = plt.subplots(1, 2, figsize=(14, 12))
plots = [
    (cor_raws, cos_all_features, 'Correlation (raw)', 'Cosine (4 features)'),
    # (cor_logs, cor_raws, 'correlatie (logreturns)', 'Correlation (raw)'),
    (cos_raws, cor_raws, 'Cosine (raw)', 'Correlation (raw)'),
    # (cos_raws, cor_nors, 'Cosine (raw)', 'Correlation (normalized)'),
    # (cos_nors, cor_raws, 'Cosine (normalized)', 'Correlation (raw)'),
    # (cos_nors, cor_nors, 'Cosine (normalized)', 'Correlation (normalized)'),
    # (cos_logs, cor_raws, 'Cosine (logreturns)', 'Correlation (raw)'),
    # (cos_logs, cor_nors, 'Cosine (logreturns)', 'Correlation (normalized)'),
    # (dtw_logs, cor_raws, 'dtw (logreturns)', 'Correlation (raw)'),
    # (dtw_logs, cor_nors, 'dtw (logreturns)', 'Correlation (normalized)'),
    # (cos_raw_norms, cor_raws, 'Cosine (raw normalized)', 'Correlation (raw)'),
    # (cos_raw_norms, cor_nors, 'Cosine (raw normalized)', 'Correlation (normalized)'),
]
# for feat in feature_cols:
#     plots.append((
#         cos_all_features, 
#         cosine_per_feature[feat], 
#         "Cosine (Open+High+Low+Close)",
#         f"Cosine ({feat})"
#     ))
# for feat in feature_cols:
#     plots.append((
#         cor_raws, 
#         cosine_per_feature[feat], 
#         "Correlation (raw)",
#         f"Cosine ({feat})"     
#     ))
for ax, (x, y, xlabel, ylabel) in zip(axes, plots):
    ax.scatter(x, y, alpha=0.6)
    x_fit = np.array(x).reshape(-1, 1)
    y_fit = np.array(y)
    model = LinearRegression().fit(x_fit, y_fit)
    x_line = np.linspace(min(x), max(x), 100).reshape(-1, 1)
    y_line = model.predict(x_line)
    ax.plot(x_line, y_line, color='orange', linewidth=2, label='Least Squares Fit')
    # Thresholds
    ax.axvline(x=0.5, color='red', linestyle='--', label='Cosine Threshold 0.5')
    ax.axhline(y=0.5, color='green', linestyle='--', label='Correlation Threshold 0.5')

    # Aslabels & titel
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f'{ylabel} vs {xlabel}')

    # Dynamische x-lim op basis van cosine type
    if 'Cosine (raw)' in xlabel:
        ax.set_xlim(0.9, 1)
    else:
        ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)

    ax.grid(True)
    ax.legend()
    
    print(f"\n{ylabel} vs {xlabel}:")
    r_squared = model.score(x_fit, y_fit)
    print(f"R²: {r_squared:.4f}")
    corr, p_value = pearsonr(x, y)
    print(f"Pearson r: {corr:.4f}, p-value: {p_value:.4e}")

plt.tight_layout()
plt.show()