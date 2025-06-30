import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
from trainer.trainer import *
from data_loader import *
from model.Thgnn import *
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns

# Pad configuratie
base_path = os.path.dirname(os.path.abspath(__file__))
# print(base_path)
data_path = os.path.join(base_path, "data", "testbatch2")
# print(data_path)
labels_path = os.path.join(data_path, "stock_labels.csv")
labelsdf = pd.read_csv(labels_path, index_col=0)
print(labelsdf.head())
percentiles=[0.0001, 0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999, 0.9999]
print("\nLabel distribution:")
labels = labelsdf.values.flatten()
loglabels = np.log(labels+1)
tanhlabels = np.tanh(labels)
tanhloglabels = np.tanh(loglabels)

labels_desc = pd.Series(labels).describe(percentiles=percentiles).round(5)
loglabels_desc = pd.Series(loglabels).describe(percentiles=percentiles).round(5)
tanhlabels_desc = pd.Series(tanhlabels).describe(percentiles=percentiles).round(5)
tanhloglabels_desc = pd.Series(tanhloglabels).describe(percentiles=percentiles).round(5)
summary_df = pd.DataFrame({
    'labels': labels_desc,
    'loglabels': loglabels_desc,
    'tanhlabels': tanhlabels_desc,
    'tanhloglabels': tanhloglabels_desc
})
print(summary_df)

threshold = 0.1

labels_filtered = labels[(labels >= -threshold) & (labels <= threshold)]
loglabels_filtered = loglabels[(loglabels >= -threshold) & (loglabels <= threshold)]
tanhlabels_filtered = tanhlabels[(tanhlabels >= -threshold) & (tanhlabels <= threshold)]
tanhloglabels_filtered = tanhloglabels[(tanhloglabels >= -threshold) & (tanhloglabels <= threshold)]

fig, axes = plt.subplots(2, 2, figsize=(15, 6))

ax1, ax2, ax3, ax4 = axes.flatten()

sns.histplot(labels_filtered, bins=100, kde=True, color='blue', label='Originele labels', ax=ax1)
ax1.set_xlim(-threshold, threshold)
ax1.set_title("Distributie van labels")
ax1.set_xlabel("Waarde")
ax1.set_ylabel("Frequentie")
ax1.legend()

sns.histplot(loglabels_filtered, bins=100, kde=True, color='orange', label='Log-returns', ax=ax2)
ax2.set_xlim(-threshold, threshold)
ax2.set_title("Distributie van log-returns")
ax2.set_xlabel("Waarde")
ax2.set_ylabel("Frequentie")
ax2.legend()

sns.histplot(tanhlabels_filtered, bins=100, kde=True, color='blue', label='Originele tanh labels', ax=ax3)
ax3.set_xlim(-threshold, threshold)
ax3.set_title("Distributie van labels")
ax3.set_xlabel("Waarde")
ax3.set_ylabel("Frequentie")
ax3.legend()

sns.histplot(tanhloglabels_filtered, bins=100, kde=True, color='orange', label='tanh Log-returns', ax=ax4)
ax4.set_xlim(-threshold, threshold)
ax4.set_title("Distributie van log-returns")
ax4.set_xlabel("Waarde")
ax4.set_ylabel("Frequentie")
ax4.legend()

plt.tight_layout()
plt.show()
