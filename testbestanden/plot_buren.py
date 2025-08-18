import os
import pandas as pd
import matplotlib.pyplot as plt

# CSV inlezen
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
database = "CSI300"
# database = "S&P500"
aantal_buren = 486 if database == "S&P500" else 217
data_path = os.path.join(base_path, "data", database)
print(data_path)
input_path = os.path.join(data_path, "relation_DSEcorr_t1", "edge_evaluation_log.csv")
input_path2 = os.path.join(data_path, "relation_corrDSC_t1", "edge_evaluation_log.csv")
df = pd.read_csv(input_path, parse_dates=["date"])
df2 = pd.read_csv(input_path2, parse_dates=["date"])
df = df.sort_values("date")

print(f"\n--- Average number of relationships {database} ---")
print(f"Average cos_pos: {df['cos_pos'].mean()/2/aantal_buren:.2f}")
print(f"Average ssa_pos: {df['ssa_pos'].mean()/2/aantal_buren:.2f}")
print(f"Average dsc_pos: {df2['pred_pos'].mean()/2/aantal_buren:.2f}")
print(f"Average pred_pos: {df['pred_pos'].mean()/2/aantal_buren:.2f}")
print(f"Average cos_neg: {df['cos_neg'].mean()/2/aantal_buren:.2f}")
print(f"Average ssa_neg: {df['ssa_neg'].mean()/2/aantal_buren:.2f}")
print(f"Average dsc_neg: {df2['pred_neg'].mean()/2/aantal_buren:.2f}")
print(f"Average pred_neg: {df['pred_neg'].mean()/2/aantal_buren:.2f}")

plt.rcParams.update({'font.size': 20})
bigger = 0.9
SMALL_SIZE = 20*bigger
MEDIUM_SIZE = 24*bigger
BIGGER_SIZE = 28*bigger
plt.rc('font', size=SMALL_SIZE)
plt.rc('axes', titlesize=BIGGER_SIZE)
plt.rc('axes', labelsize=MEDIUM_SIZE)
plt.rc('xtick', labelsize=SMALL_SIZE)
plt.rc('ytick', labelsize=SMALL_SIZE)
plt.rc('legend', fontsize=SMALL_SIZE)
plt.rc('figure', titlesize=BIGGER_SIZE)

figsize = (16, 4)
plt.figure(figsize=figsize)
plt.plot(df["date"], df["cos_pos"]/2/aantal_buren, label="PC")
plt.plot(df["date"], df["ssa_pos"]/2/aantal_buren, label="SSA-PC")
plt.plot(df2["date"], df2["pred_pos"]/2/aantal_buren, label="DSC-PC")
plt.plot(df["date"], df["pred_pos"]/2/aantal_buren, label="DSE-PC")
plt.xlabel("Date")
plt.ylabel("Amount")
plt.legend()
plt.ylim(bottom=0)
plt.ylim(top=aantal_buren)
plt.tight_layout()
plt.show()

plt.figure(figsize=figsize)
plt.plot(df["date"], df["cos_neg"]/2/aantal_buren, label="PC")
plt.plot(df["date"], df["ssa_neg"]/2/aantal_buren, label="SSA-PC")
plt.plot(df2["date"], df2["pred_neg"]/2/aantal_buren, label="DSC-PC")
plt.plot(df["date"], df["pred_neg"]/2/aantal_buren, label="DSE-PC")
plt.xlabel("Date")
plt.ylabel("Amount")
plt.legend()
plt.ylim(bottom=0)
plt.ylim(top=aantal_buren)
plt.tight_layout()
plt.show()