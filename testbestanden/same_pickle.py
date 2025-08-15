import os
import time
import pickle
import numpy as np
import torch
from tqdm import tqdm

# Tolerantie voor floats (alleen absolute)
TOL = 1e-9
MAX_SHOW = 20  # max aantal verschillen om te tonen

def to_float_tensor(x) -> torch.Tensor:
    """Converteer labels naar torch.float32 tensor (CPU) voor een eerlijke vergelijking."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().to(dtype=torch.float32, copy=False)
    arr = np.asarray(x)
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32, copy=False)
    return torch.from_numpy(arr)

def load_labels_tensor(pkl_path: str) -> torch.Tensor:
    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, dict) or "labels" not in obj:
        raise RuntimeError(f"'labels' ontbreekt in {pkl_path}")
    return to_float_tensor(obj["labels"])

def compare_label_dirs(dir_a: str, dir_b: str, tol: float = TOL):
    files_a = sorted([f for f in os.listdir(dir_a) if f.endswith(".pkl")])
    files_b = sorted([f for f in os.listdir(dir_b) if f.endswith(".pkl")])

    set_a, set_b = set(files_a), set(files_b)
    only_a = sorted(set_a - set_b)
    only_b = sorted(set_b - set_a)
    common = sorted(set_a & set_b)

    print(f"[INFO] {dir_a}: {len(files_a)} files | {dir_b}: {len(files_b)} files")
    if only_a:
        print(f"[WARN] Alleen in {dir_a}: {len(only_a)} (eerste 10): {only_a[:10]}")
    if only_b:
        print(f"[WARN] Alleen in {dir_b}: {len(only_b)} (eerste 10): {only_b[:10]}")

    equal_cnt = 0
    mismatch_cnt = 0
    examples = []

    for fname in tqdm(common, desc="Vergelijken (labels)"):
        pa = os.path.join(dir_a, fname)
        pb = os.path.join(dir_b, fname)

        la = load_labels_tensor(pa)
        lb = load_labels_tensor(pb)

        if la.shape != lb.shape:
            mismatch_cnt += 1
            examples.append((fname, f"shape {tuple(la.shape)} vs {tuple(lb.shape)}"))
            continue

        diff = torch.abs(la - lb)
        # vergelijk op absolute tolerantie
        n_over = int((diff > tol).sum().item())
        if n_over == 0:
            equal_cnt += 1
        else:
            mismatch_cnt += 1
            maxdiff = float(diff.max().item())
            examples.append((fname, f"max|Δ|={maxdiff:.3e}, n>|tol|={n_over}"))

    print("\n=== SAMENVATTING ===")
    print(f"Gemeenschappelijke files: {len(common)}")
    print(f"Labels identiek (≤ {tol}): {equal_cnt}")
    print(f"Labels NIET identiek: {mismatch_cnt}")

    if examples:
        print(f"\nVoorbeelden (max {MAX_SHOW}):")
        for fn, msg in examples[:MAX_SHOW]:
            print(f"  - {fn}: {msg}")

def print_all_labels_from_dir(pkl_dir: str):
    files = sorted([f for f in os.listdir(pkl_dir) if f.endswith(".pkl")])
    for fname in files:
        path = os.path.join(pkl_dir, fname)
        with open(path, "rb") as f:
            obj = pickle.load(f)
        labels = obj.get("mask", None)
        print(f"{fname}:")
        print(labels)
        print("-" * 40)



if __name__ == "__main__":
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_sp = os.path.join(base_path, "data", "CSI300")

    dir_new = os.path.join(data_sp, "data_train_predict_STATICcorr_t1")        # jouw herlabelde set
    dir_good = os.path.join(data_sp, "data_train_predict_onlycosine")  # referentie

    start = time.time()
    compare_label_dirs(dir_new, dir_good, tol=TOL)
    print(f"\nKlaar in {time.time()-start:.1f}s")





def print_all_labels_from_dir(pkl_dir: str):
    files = sorted([f for f in os.listdir(pkl_dir) if f.endswith(".pkl")])
    for fname in files:
        path = os.path.join(pkl_dir, fname)
        with open(path, "rb") as f:
            obj = pickle.load(f)
        labels = obj.get("labels", None)
        print(f"{fname}:")
        print(labels)
        print("-" * 40)