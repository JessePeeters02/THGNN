from trainer.trainer import *
from data_loader import *
from model.Thgnn import *
# from model.Thgnn_new import *
import warnings
import torch
import os
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import pandas as pd
from pandas.core.frame import DataFrame
from tqdm import tqdm
import numpy as np

warnings.filterwarnings("ignore")
t_float = torch.float64
torch.multiprocessing.set_sharing_strategy('file_system')

# base_path = os.path.dirname(os.path.abspath(__file__))  # Huidige scriptmap
# print(f"base_path: {base_path}")
# data_path = os.path.join(base_path, "data", "S&P500")
# print(f"data_path: {data_path}")
# data_train_predict_path = os.path.join(data_path, "data_train_predict_corr") #gpu_wvt, oldway_0.6, gpu_wvt
# print(f"data_train_predict_path: {data_train_predict_path}")
# daily_stock_path = os.path.join(data_path, "daily_stock_corr") #gpu_wvt, oldway, gpu_wvt
# print(f"daily_stock_path: {daily_stock_path}")
# save_path = os.path.join(data_path, "model_saved_corr_bin")
# os.makedirs(save_path, exist_ok=True)
# prediction_path = os.path.join(data_path, "prediction_corr_bin")
# os.makedirs(prediction_path, exist_ok=True)

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(device)

class Args:
    def __init__(self, gpu=0, subtask="regression"): #regression or classification_binare, also switch: trainer.py 31/32 and thgnn.py 128/129
        # device
        self.gpu = str(0)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # data settings
        # adj_threshold = 0.4
        # self.adj_str = str(int(100*adj_threshold))
        self.pos_adj_dir = "pos_adj" #+ self.adj_str
        self.neg_adj_dir = "neg_adj" #+ self.adj_str
        self.feat_dir = "features"
        self.label_dir = "labels"
        self.mask_dir = "mask"
        self.data_start = data_start
        self.data_middle = data_middle
        self.data_end = data_end
        self.pre_data = pre_data
        # epoch settings
        self.max_epochs = 20
        self.epochs_eval = 10
        # learning rate settings
        self.lr = 0.001
        self.gamma = 0.3
        # model settings
        self.hidden_dim = 128
        self.num_heads = 8
        self.out_features = 32
        self.model_name = "StockHeteGAT"
        self.loss_fcn = mse_loss
        # save model settings
        #self.save_path = os.path.join(os.path.abspath('.'), "/home/THGNN-main/data/model_saved/")
        self.save_path = save_path
        self.load_path = self.save_path
        self.save_name = self.model_name + "_hidden_" + str(self.hidden_dim) + "_head_" + str(self.num_heads) + \
                         "_outfeat_" + str(self.out_features)
        self.epochs_save_by = self.max_epochs
        self.sub_task = subtask
        eval("self.{}".format(self.sub_task))()

    def regression(self):
        self.save_name = self.save_name + "_reg_rank_"
        self.loss_fcn = mse_loss
        self.label_dir = self.label_dir + "_regression"
        self.mask_dir = self.mask_dir + "_regression"

    def regression_binary(self):
        self.save_name = self.save_name + "_reg_binary_"
        self.loss_fcn = mse_loss
        self.label_dir = self.label_dir + "_twoclass"
        self.mask_dir = self.mask_dir + "_twoclass"

    def classification_binary(self):
        self.save_name = self.save_name + "_clas_binary_"
        self.loss_fcn = bce_loss
        self.label_dir = self.label_dir + "_twoclass"
        self.mask_dir = self.mask_dir + "_twoclass"

    def classification_tertiary(self):
        self.save_name = self.save_name + "_clas_tertiary_"
        self.loss_fcn = bce_loss
        self.label_dir = self.label_dir + "_threeclass"
        self.mask_dir = self.mask_dir + "_threeclass"


def fun_train_predict(data_start, data_middle, data_end, pre_data):
    args = Args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    #dataset = AllGraphDataSampler(base_dir="/home/THGNN-main/data/data_train_predict/", data_start=data_start,
    #                              data_middle=data_middle, data_end=data_end)
    #val_dataset = AllGraphDataSampler(base_dir="/home/THGNN-main/data/data_train_predict/", mode="val", data_start=data_start,
    #                                  data_middle=data_middle, data_end=data_end)
    dataset = AllGraphDataSampler(base_dir=data_train_predict_path, mode="train", data_start=data_start,
                              data_middle=data_middle, data_end=data_end)
    # print(f"Aantal samples in dataset: {len(dataset)}")
    val_dataset = AllGraphDataSampler(base_dir=data_train_predict_path, mode="val", data_start=data_start,
                                  data_middle=data_middle, data_end=data_end)
    predict_dataset = AllGraphDataSampler(base_dir=data_train_predict_path, mode="val",
                                       data_start=data_end, data_middle=data_end, data_end=data_end+1)
    
    dataset_loader = DataLoader(dataset, pin_memory=False, collate_fn=lambda x: x)
    val_dataset_loader = DataLoader(val_dataset, pin_memory=False)
    predict_dataset_loader = DataLoader(predict_dataset, pin_memory=False)

    model = eval(args.model_name)(hidden_dim=args.hidden_dim, num_heads=args.num_heads,
                                  out_features=args.out_features).to(args.device)

    """ train """
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    cold_scheduler = StepLR(optimizer=optimizer, step_size=5000, gamma=0.9, last_epoch=-1)
    default_scheduler = cold_scheduler

    print('start training')
    best_val_loss = float('inf')
    best_model_state = None

    for epoch in range(args.max_epochs):
        train_loss = train_epoch(epoch=epoch, args=args, model=model, dataset_train=dataset_loader,
                                 optimizer=optimizer, scheduler=default_scheduler, loss_fcn=args.loss_fcn)
        if (epoch+1) % args.epochs_eval == 0:
            val_loss, _ = eval_epoch(args=args, model=model, dataset_eval=val_dataset_loader, loss_fcn=args.loss_fcn)
            print('Epoch: {}/{}, train loss: {:.6f}, val loss: {:.6f}'.format(epoch + 1, args.max_epochs, train_loss, val_loss))
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict()
                print("save model!")
                state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch + 1}
                torch.save(state, os.path.join(args.save_path, pre_data + "_epoch_" + str(epoch + 1) + ".dat"))
        else:
            print('Epoch: {}/{}, train loss: {:.6f}'.format(epoch + 1, args.max_epochs, train_loss))


    # predict
    checkpoint = torch.load(os.path.join(args.load_path, pre_data + "_epoch_" + str(epoch + 1) + ".dat"))
    model.load_state_dict(checkpoint['model'])
    model.eval()

    # prepare prediction input: laatste dag van de val-set (data_end - 1)
    predict_input_dataset = AllGraphDataSampler(base_dir=data_train_predict_path, mode="val",
                                                data_start=data_end - 1, data_middle=data_end - 1, data_end=data_end)

    predict_input_loader = DataLoader(predict_input_dataset, pin_memory=False)

    df_score = pd.DataFrame()
    df_weights = pd.DataFrame()

    for i, tmp_data in enumerate(predict_input_dataset):
        pos_adj, neg_adj, features, labels, mask = extract_data(tmp_data, args.device)

        with torch.no_grad():
            logits, weights = model(features, pos_adj, neg_adj, requires_weight=True)

        result = logits.data.cpu().numpy().tolist()
        result_new = [r[0] for r in result]

        # We koppelen deze scores aan de testdaglabels (die we apart ophalen, zonder hun features te gebruiken)
        test_label_dataset = AllGraphDataSampler(base_dir=data_train_predict_path, mode="val",
                                                 data_start=data_end, data_middle=data_end, data_end=data_end + 1)

        df = pd.read_csv(os.path.join(daily_stock_path, tmp_data[data_end]), dtype=object)
        df['score'] = pd.DataFrame({'score': result_new})
        df_score = pd.concat([df_score, df])

        # attention statistics, net zoals in jouw oorspronkelijke versie
        pos_weights = weights["pos_attn_weights"].cpu().numpy()
        neg_weights = weights["neg_attn_weights"].cpu().numpy()
        sem_weights = weights["sem_attn_weights"].cpu().numpy()

        df_weights = pd.concat([df_weights, pd.DataFrame({
            "sample_id": i,
            "pos_weight_mean": np.mean(pos_weights),
            "neg_weight_mean": np.mean(neg_weights),
            "beta_self_mean": np.mean(sem_weights[:, 0]),
            "beta_pos_mean": np.mean(sem_weights[:, 1]),
            "beta_neg_mean": np.mean(sem_weights[:, 2]),
        }, index=[0])])

    # totaalmean toevoegen
    total_means = df_weights.mean(numeric_only=True).to_dict()
    total_means["sample_id"] = "TOTAAL"
    df_weights = pd.concat([df_weights, pd.DataFrame([total_means])], ignore_index=True)

    df_score.to_csv(os.path.join(prediction_path, "pred.csv"))
    df_weights.to_csv(os.path.join(prediction_path, "attention_weights.csv"))
    
if __name__ == "__main__":

    base_path = os.path.dirname(os.path.abspath(__file__))  # Huidige scriptmap
    print(f"base_path: {base_path}")
    data_path = os.path.join(base_path, "data", "testbatch_mini")
    print(f"data_path: {data_path}")

    data_train_predict_path = os.path.join(data_path, f"data_train_predict_mini") #gpu_wvt, oldway_0.6, gpu_wvt
    print(f"data_train_predict_path: {data_train_predict_path}")
    daily_stock_path = os.path.join(data_path, f"daily_stock_mini") #gpu_wvt, oldway, gpu_wvt
    print(f"daily_stock_path: {daily_stock_path}")
    save_path = os.path.join(data_path, f"model_saved_rolingwindow_test")
    os.makedirs(save_path, exist_ok=True)
    prediction_path = os.path.join(data_path, f"model_saved_rolingwindow_test")
    os.makedirs(prediction_path, exist_ok=True)
    print(prediction_path)

    total_data_points = len(os.listdir(data_train_predict_path))
    print(f"Total data points: {total_data_points}")

    val_len = 10
    window_len = 10
    rolling_start = total_data_points - window_len - 1  # Laat genoeg ruimte over voor testdagen
    rolling_end = total_data_points - 2                 # Laatste dag waarop je kan voorspellen

    for T in range(rolling_start, rolling_end + 1):
        # Rolling setup per predictiedag T
        train_start = 0
        train_end = T - val_len - 1
        val_start = T - val_len
        val_end = T - 1
        predict_day = T

        data_start = train_start
        data_middle = val_start
        data_end = val_end + 1  # data_end is exclusive, dus +1 om val-set af te sluiten

        pre_data = f"rolling_T{T}"

        print(f"\n==== Rolling predictiedag: T={T} ====")
        print(f"Train: {train_start} - {train_end}")
        print(f"Val:   {val_start} - {val_end}")
        print(f"Test:  {predict_day}")
        print(f"Data start: {data_start}, middle: {data_middle}, end: {data_end}, pre_data: {pre_data}")

        fun_train_predict(data_start, data_middle, data_end, pre_data)