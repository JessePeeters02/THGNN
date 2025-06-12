import pandas as pd
import matplotlib.pyplot as plt
import os

# Pad configuratie
base_path = os.path.dirname(os.path.abspath(__file__))

df = pd.read_csv(os.path.join(base_path, 'snapshot_log_csi300.csv'))
print(df)

# Optional: if the first column contains dates
df['date'] = pd.to_datetime(df['date'])
df = df[df['date']>='2024-07-01']  # adjusted date and correct filtering
# Create line plot
plt.figure(figsize=(6, 3))  # Set figure size to 6x3 inches
plt.plot(df['date'], df['pos_edges'], label='Positive neighbors')
plt.plot(df['date'], df['neg_edges'], label='Negative neighbors')
# Labels and title
plt.xlabel('Date')
plt.ylabel('Number of neighbors')
# plt.title('Number of neighbors over time')
plt.xticks(rotation=45)  # Rotate dates if they overlap
plt.legend()
plt.tight_layout()
# Show the plot
plt.show()
