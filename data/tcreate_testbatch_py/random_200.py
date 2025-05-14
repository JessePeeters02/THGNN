import os
import pandas as pd
import numpy as np
from tqdm import tqdm

TOP_N = 1000
NUM_BATCHES = 5
BATCH_SIZE = 200
SEGMENTS = 4
STOCKS_PER_SEGMENT_PER_BATCH = BATCH_SIZE // SEGMENTS

base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
stock_data_path = os.path.join(os.path.dirname(base_path), "portfolio_construction", "data", "NASDAQ", "NASDAQ_data_wTO")
output_dir = os.path.join(base_path, "data", f"NASDAQ_batches_{NUM_BATCHES}_{BATCH_SIZE}")
os.makedirs(output_dir, exist_ok=True)

def compute_stock_scores(stock_data_path):
    stock_files = [f for f in os.listdir(stock_data_path) if f.endswith('.csv')]
    stock_scores = []
    skipped = 0

    for stock_file in tqdm(stock_files, desc="Scores van stocks berekenen"):
        stock_name = stock_file.split('.')[0]
        df = pd.read_csv(os.path.join(stock_data_path, stock_file))

        # Vereiste kolommen check
        if not {'Close', 'Volume', 'Turnover'}.issubset(df.columns):
            continue

        # Filter op geldige waarden
        if ((df['Volume'] <= 0).any() or 
            (df['Turnover'] <= 0).any() or 
            (df['Close'] <= 0).any()):
            skipped += 1
            continue

        # Bereken gewicht (bedrijfsgrootte-score)
        score = (df['Close'] * df['Volume'] / df['Turnover']).mean()
        stock_scores.append((stock_name, score))

    print(f"  {skipped} aandelen overgeslagen door ongeldige data.")
    print(f"  {len(stock_scores)} aandelen succesvol")
    return stock_scores

def select_top_n_stocks(stock_scores, n=TOP_N):
    df = pd.DataFrame(stock_scores, columns=['Stock', 'Score'])
    df = df.sort_values(by='Score', ascending=False).reset_index(drop=True)
    print(df.head(10))
    return df.iloc[:n]

def split_into_segments(df_top_n, num_segments=SEGMENTS):
    segment_size = len(df_top_n) // num_segments
    segments = []
    for i in range(num_segments):
        start = i * segment_size
        end = (i + 1) * segment_size if i < num_segments - 1 else len(df_top_n)
        segment = df_top_n.iloc[start:end].copy()
        print(f"    segment {i+1}: {len(segment)} stocks")
        segments.append(segment)
    print(f" Verdeling in {num_segments} segmenten van elk ±{segment_size} aandelen.")
    return segments

def create_balanced_batches(segments, num_batches=NUM_BATCHES):
    batches = [[] for _ in range(num_batches)]
    used_stocks = set()

    print(f"Bouwen van {num_batches} gebalanceerde batches...")
    for seg_index, segment in enumerate(segments):
        stocks = segment['Stock'].sample(frac=1, random_state=42 + seg_index).tolist()  # shuffle per segment
        assert len(stocks) >= num_batches * STOCKS_PER_SEGMENT_PER_BATCH, "Niet genoeg stocks in segment!"

        for i in range(num_batches):
            batch_stocks = stocks[i*STOCKS_PER_SEGMENT_PER_BATCH : (i+1)*STOCKS_PER_SEGMENT_PER_BATCH]
            batches[i].extend(batch_stocks)
            used_stocks.update(batch_stocks)

    # Final shuffle per batch
    for batch in batches:
        np.random.shuffle(batch)
        print(f"  Batch {batches.index(batch)+1}: {len(batch)} stocks")

    print(f"Elke batch bevat 200 aandelen uit alle grootteklassen.")
    return batches

def save_batches(batches, output_dir):
    for i, batch in enumerate(batches):
        df = pd.DataFrame({'Stock': batch})
        batch_dir = os.path.join(output_dir, f"batch_{i+1}")
        os.makedirs(batch_dir, exist_ok=True)
        batch_path = os.path.join(batch_dir, f"batch.csv")
        df.to_csv(batch_path, index=False)
        print(f"💾 Batch {i+1} opgeslagen: {batch_path}")

if __name__ == "__main__":
    scores = compute_stock_scores(stock_data_path)
    df_top1000 = select_top_n_stocks(scores, n=TOP_N)
    segments = split_into_segments(df_top1000, num_segments=SEGMENTS)
    batches = create_balanced_batches(segments, num_batches=NUM_BATCHES)
    save_batches(batches, output_dir)
