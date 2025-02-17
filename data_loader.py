import os
import sys
from torch.utils import data
import pickle

base_path = os.path.dirname(os.path.abspath(__file__))  # Huidige scriptmap
data_path = os.path.join(base_path, "data", "data_train_predict")
os.makedirs(data_path, exist_ok=True)
print(f"Pad naar de datasetmap: {data_path}")

class AllGraphDataSampler(data.Dataset):
    print(f"Pad naar de datasetmap: {data_path}")

    # def __init__(self, base_dir, gname_list=None, data_start=None, data_middle=None, data_end=None, mode="train"):
    #     print("Laden van dataset...")
    #     for bestand in os.listdir(data_path):
    #         print("Bestand gevonden:", bestand)
    #         with open(os.path.join(data_path, bestand), "rb") as f:
    #             try:
    #                 data = pickle.load(f)
    #                 print(f"Bestand {bestand} geladen met type: {type(data)}")
    #             except Exception as e:
    #                 print(f"Fout bij laden van {bestand}: {e}")
    #     self.data_dir = os.path.join(base_dir)
    #     self.mode = mode
    #     self.data_start = data_start
    #     self.data_middle = data_middle
    #     self.data_end = data_end
    #     if gname_list is None:
    #         self.gnames_all = os.listdir(self.data_dir)
    #         self.gnames_all.sort()

    #     print(f"Gnames_all range: {len(self.gnames_all)}")
    #     print(f"Train data range: {self.data_start} - {self.data_middle}")
    #     print(f"Val data range: {self.data_middle} - {self.data_end}")
    #     if self.data_start >= len(self.gnames_all):
    #         print(f"Fout: data_start ({self.data_start}) is groter dan het aantal grafen ({len(self.gnames_all)}).")
    #     if self.data_middle > len(self.gnames_all):
    #         print(f"Fout: data_middle ({self.data_middle}) is groter dan het aantal grafen ({len(self.gnames_all)}).")
    #     if self.data_end > len(self.gnames_all):
    #         print(f"Fout: data_end ({self.data_end}) is groter dan het aantal grafen ({len(self.gnames_all)}).")


    #     if mode == "train":
    #         self.gnames_all = self.gnames_all[self.data_start:self.data_middle]
    #     elif mode == "val":
    #         self.gnames_all = self.gnames_all[self.data_middle:self.data_end]
    #     self.data_all = self.load_state()

    def __init__(self, base_dir, gname_list=None, data_start=None, data_middle=None, data_end=None, mode="train"):
        print("Laden van dataset...")
        self.data_dir = os.path.join(base_dir)
        self.mode = mode
        self.data_start = data_start
        self.data_middle = data_middle
        self.data_end = data_end
        
        # Controle of er bestanden zijn in de map
        self.gnames_all = os.listdir(self.data_dir)
        print("Bestanden in de data map:", self.gnames_all)
        if len(self.gnames_all) == 0:
            print("Waarschuwing: Geen bestanden in de data map.")
        
        # Controle van de indices
        if self.data_start >= len(self.gnames_all):
            print(f"Fout: data_start ({self.data_start}) is groter dan het aantal grafen ({len(self.gnames_all)}).")
        if self.data_middle > len(self.gnames_all):
            print(f"Fout: data_middle ({self.data_middle}) is groter dan het aantal grafen ({len(self.gnames_all)}).")
        if self.data_end > len(self.gnames_all):
            print(f"Fout: data_end ({self.data_end}) is groter dan het aantal grafen ({len(self.gnames_all)}).")
        
        if gname_list is None:
            self.gnames_all.sort()
        
        if mode == "train":
            self.gnames_all = self.gnames_all[self.data_start:self.data_middle]
        elif mode == "val":
            self.gnames_all = self.gnames_all[self.data_middle:self.data_end]
        print(f"Gnames all after split: {self.gnames_all}")  # Controleer na splitsen
        self.data_all = self.load_state()


    def __len__(self):
        return len(self.data_all)

    # def load_state(self):
    #     data_all = []
    #     length = len(self.gnames_all)
    #     for i in range(length):
    #         sys.stdout.flush()
    #         sys.stdout.write('{} data loading: {:.2f}%{}'.format(self.mode, i*100/length, '\r'))
    #         data_all.append(pickle.load(open(os.path.join(self.data_dir, self.gnames_all[i]), "rb")))
    #     print('{} data loaded!'.format(self.mode))
    #     return data_all

    def load_state(self):
        data_all = []
        length = len(self.gnames_all)
        for i in range(length):
            sys.stdout.flush()
            sys.stdout.write('{} data loading: {:.2f}%{}'.format(self.mode, i*100/length, '\r'))
            try:
                data = pickle.load(open(os.path.join(self.data_dir, self.gnames_all[i]), "rb"))
                data_all.append(data)
            except Exception as e:
                print(f"Fout bij laden van {self.gnames_all[i]}: {e}")
        print('{} data loaded!'.format(self.mode))
        return data_all

    def __getitem__(self, idx):
        print(f"Sample ophalen op index {idx}")
        sample = self.data[idx]  # Hier kan de fout zitten
        print(f"Sample geladen: {sample}")
        return self.data_all[idx]
