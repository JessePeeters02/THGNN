from trainer.trainer import *
from data_loader import *
from model.Thgnn import *
# from model.Thgnn_no_beta import *
# from model.Thgnn_no_alpha import *
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
import time

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Selected device: {device}")

# if torch.cuda.is_available():
#     device = torch.device("cuda")
#     print(device)

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
        # self.loss_fcn = mae_loss
        # save model settings
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
        # self.loss_fcn = mae_loss
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
    best_epoch = 0

    for epoch in range(args.max_epochs):
        train_loss = train_epoch(epoch=epoch, args=args, model=model, dataset_train=dataset_loader,
                                 optimizer=optimizer, scheduler=default_scheduler, loss_fcn=args.loss_fcn)
        if ((epoch + 1) % args.epochs_eval == 0) and (epoch + 1 >= 9):
            val_loss, _ = eval_epoch(args=args, model=model, dataset_eval=val_dataset_loader, loss_fcn=args.loss_fcn)
            print('Epoch: {}/{}, train loss: {:.6f}, val loss: {:.6f}'.format(epoch + 1, args.max_epochs, train_loss, val_loss))
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict()
                best_epoch = epoch + 1
                print("save model!")
                state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch + 1}
                torch.save(state, os.path.join(args.save_path, pre_data + "_epoch_" + str(epoch + 1) + ".dat"))
        else:
            print('Epoch: {}/{}, train loss: {:.6f}'.format(epoch + 1, args.max_epochs, train_loss))


    # predict
    checkpoint = torch.load(os.path.join(args.load_path, pre_data + "_epoch_" + str(best_epoch) + ".dat"), map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    df_score = pd.DataFrame()
    df_weights = pd.DataFrame()

    for i, tmp_data in enumerate(predict_dataset):
        pos_adj, neg_adj, features, labels, mask = extract_data(tmp_data, args.device)

        with torch.no_grad():
            logits, weights = model(features, pos_adj, neg_adj, requires_weight=True)

        result = logits.data.cpu().numpy().tolist()
        result_new = [r[0] for r in result]

        # Pak de juiste dag uit de daily_stock_path directory, corresponderend met data_end
        daily_file = tmp_data["date"].replace(".pkl", ".csv")
        df = pd.read_csv(os.path.join(daily_stock_path, daily_file), dtype=object)
        df['score'] = pd.DataFrame({'score': result_new})
        df['label'] = labels.cpu().numpy()
        df_score = pd.concat([df_score, df])

        # attention statistics, net zoals in jouw oorspronkelijke versie
        pos_weights = weights["pos_attn_weights"].cpu().numpy()
        neg_weights = weights["neg_attn_weights"].cpu().numpy()
        sem_weights = weights["sem_attn_weights"].cpu().numpy()

        df_weights = pd.concat([df_weights, pd.DataFrame({
            "sample_id": i+1,
            "dt": tmp_data["date"].replace(".pkl", ""),
            "pos_weight_mean": np.mean(pos_weights),
            "neg_weight_mean": np.mean(neg_weights),
            "beta_self_mean": np.mean(sem_weights[:, 0]),
            "beta_pos_mean": np.mean(sem_weights[:, 1]),
            "beta_neg_mean": np.mean(sem_weights[:, 2]),
        }, index=[0])])

    # totaalmean toevoegen
    if len(df_weights) > 1:
        total_means = df_weights.mean(numeric_only=True).to_dict()
        total_means["sample_id"] = "TOTAAL"
        total_means["dt"] = "/"
        df_weights = pd.concat([df_weights, pd.DataFrame([total_means])], ignore_index=True)

    df_score.to_csv(
    os.path.join(prediction_path, "pred.csv"),
    mode='a',
    index=False,
    header=not os.path.exists(os.path.join(prediction_path, "pred.csv"))
    )

    df_weights.to_csv(
    os.path.join(prediction_path, "attention_weights.csv"),
    mode='a',
    index=False,
    header=not os.path.exists(os.path.join(prediction_path, "attention_weights.csv"))
    )


def CSI300_cosineDSC():
    print(f"Start CSI300_cosineDSC: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    start_time = time.time()
    global base_path, data_path, data_train_predict_path, daily_stock_path, save_path, prediction_path, data_start, data_middle, data_end, pre_data
    base_path = os.path.dirname(os.path.abspath(__file__))  # Huidige scriptmap
    print(f"base_path: {base_path}")
    data_path = os.path.join(base_path, "data", "CSI300")
    print(f"data_path: {data_path}")
    data_train_predict_path = os.path.join(data_path, f"data_train_predict_cosineDSC") #gpu_wvt, oldway_0.6, gpu_wvt
    print(f"data_train_predict_path: {data_train_predict_path}")
    daily_stock_path = os.path.join(data_path, f"daily_stock_cosineDSC") #gpu_wvt, oldway, gpu_wvt
    print(f"daily_stock_path: {daily_stock_path}")
    save_path = os.path.join(data_path, f"model_saved_rolingwindow_cosineDSC")
    os.makedirs(save_path, exist_ok=True)
    prediction_path = save_path
    total_data_points = len(os.listdir(data_train_predict_path))
    print(f"Total data points: {total_data_points}")
    val_len = 10
    window_len = 8
    rolling_start = total_data_points - window_len  # Laat genoeg ruimte over voor testdagen #inclusief
    rolling_end = total_data_points                 # Laatste dag waarop je kan voorspellen #exclusief
    for T in range(rolling_start, rolling_end):
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
    end_time = time.time()
    minutes_taken = round((end_time - start_time) / 60, 1)
    print(f"Done CSI300_cosineDSC. Time taken: {minutes_taken} minutes") 

def CSI300_full():
    print(f"Start CSI300_full: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    start_time = time.time()
    global base_path, data_path, data_train_predict_path, daily_stock_path, save_path, prediction_path, data_start, data_middle, data_end, pre_data
    base_path = os.path.dirname(os.path.abspath(__file__))  # Huidige scriptmap
    print(f"base_path: {base_path}")
    data_path = os.path.join(base_path, "data", "CSI300")
    print(f"data_path: {data_path}")
    data_train_predict_path = os.path.join(data_path, f"data_train_predict_DSE") #gpu_wvt, oldway_0.6, gpu_wvt
    print(f"data_train_predict_path: {data_train_predict_path}")
    daily_stock_path = os.path.join(data_path, f"daily_stock_DSE") #gpu_wvt, oldway, gpu_wvt
    print(f"daily_stock_path: {daily_stock_path}")
    save_path = os.path.join(data_path, f"model_saved_rolingwindow_DSE")
    os.makedirs(save_path, exist_ok=True)
    prediction_path = save_path
    total_data_points = len(os.listdir(data_train_predict_path))
    print(f"Total data points: {total_data_points}")
    val_len = 10
    window_len = 120
    rolling_start = total_data_points - window_len  # Laat genoeg ruimte over voor testdagen #inclusief
    rolling_end = total_data_points                 # Laatste dag waarop je kan voorspellen #exclusief
    for T in range(rolling_start, rolling_end):
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
    end_time = time.time()
    minutes_taken = round((end_time - start_time) / 60, 1)
    print(f"Done CSI300_full. Time taken: {minutes_taken} minutes") 

def CSI300_STATIC():
    print(f"Start CSI300_STATIC: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    start_time = time.time()
    global base_path, data_path, data_train_predict_path, daily_stock_path, save_path, prediction_path, data_start, data_middle, data_end, pre_data
    base_path = os.path.dirname(os.path.abspath(__file__))  # Huidige scriptmap
    print(f"base_path: {base_path}")
    data_path = os.path.join(base_path, "data", "CSI300")
    print(f"data_path: {data_path}")
    data_train_predict_path = os.path.join(data_path, f"data_train_predict_STATIC") #gpu_wvt, oldway_0.6, gpu_wvt
    print(f"data_train_predict_path: {data_train_predict_path}")
    daily_stock_path = os.path.join(data_path, f"daily_stock_STATIC") #gpu_wvt, oldway, gpu_wvt
    print(f"daily_stock_path: {daily_stock_path}")
    save_path = os.path.join(data_path, f"model_saved_rolingwindow_STATIC")
    os.makedirs(save_path, exist_ok=True)
    prediction_path = save_path
    total_data_points = len(os.listdir(data_train_predict_path))
    print(f"Total data points: {total_data_points}")
    val_len = 10
    window_len = 20
    rolling_start = total_data_points - window_len  # Laat genoeg ruimte over voor testdagen #inclusief
    rolling_end = total_data_points                 # Laatste dag waarop je kan voorspellen #exclusief
    for T in range(rolling_start, rolling_end):
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
    end_time = time.time()
    minutes_taken = round((end_time - start_time) / 60, 1)
    print(f"Done CSI300_STATIC. Time taken: {minutes_taken} minutes")

def CSI300_full_gericht():
    print(f"Start CSI300_full_gericht: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    start_time = time.time()
    global base_path, data_path, data_train_predict_path, daily_stock_path, save_path, prediction_path, data_start, data_middle, data_end, pre_data
    base_path = os.path.dirname(os.path.abspath(__file__))  # Huidige scriptmap
    print(f"base_path: {base_path}")
    data_path = os.path.join(base_path, "data", "CSI300")
    print(f"data_path: {data_path}")
    data_train_predict_path = os.path.join(data_path, f"data_train_predict_DSE_gericht") #gpu_wvt, oldway_0.6, gpu_wvt
    print(f"data_train_predict_path: {data_train_predict_path}")
    daily_stock_path = os.path.join(data_path, f"daily_stock_DSE_gericht") #gpu_wvt, oldway, gpu_wvt
    print(f"daily_stock_path: {daily_stock_path}")
    save_path = os.path.join(data_path, f"model_saved_rolingwindow_DSE_gericht")
    os.makedirs(save_path, exist_ok=True)
    prediction_path = save_path
    total_data_points = len(os.listdir(data_train_predict_path))
    print(f"Total data points: {total_data_points}")
    val_len = 10
    window_len = 20
    rolling_start = total_data_points - window_len  # Laat genoeg ruimte over voor testdagen #inclusief
    rolling_end = total_data_points                 # Laatste dag waarop je kan voorspellen #exclusief
    for T in range(rolling_start, rolling_end):
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
    end_time = time.time()
    minutes_taken = round((end_time - start_time) / 60, 1)
    print(f"Done CSI300_full_gericht. Time taken: {minutes_taken} minutes")

def CSI300_full_ct():
    print(f"Start CSI300_full_ct: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    start_time = time.time()
    global base_path, data_path, data_train_predict_path, daily_stock_path, save_path, prediction_path, data_start, data_middle, data_end, pre_data
    base_path = os.path.dirname(os.path.abspath(__file__))  # Huidige scriptmap
    print(f"base_path: {base_path}")
    data_path = os.path.join(base_path, "data", "CSI300")
    print(f"data_path: {data_path}")
    data_train_predict_path = os.path.join(data_path, f"data_train_predict_DSE_ct") #gpu_wvt, oldway_0.6, gpu_wvt
    print(f"data_train_predict_path: {data_train_predict_path}")
    daily_stock_path = os.path.join(data_path, f"daily_stock_DSE_ct") #gpu_wvt, oldway, gpu_wvt
    print(f"daily_stock_path: {daily_stock_path}")
    save_path = os.path.join(data_path, f"model_saved_rolingwindow_DSE_ct")
    os.makedirs(save_path, exist_ok=True)
    prediction_path = save_path
    total_data_points = len(os.listdir(data_train_predict_path))
    print(f"Total data points: {total_data_points}")
    val_len = 10
    window_len = 20
    rolling_start = total_data_points - window_len  # Laat genoeg ruimte over voor testdagen #inclusief
    rolling_end = total_data_points                 # Laatste dag waarop je kan voorspellen #exclusief
    for T in range(rolling_start, rolling_end):
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
    end_time = time.time()
    minutes_taken = round((end_time - start_time) / 60, 1)
    print(f"Done CSI300_full_ct. Time taken: {minutes_taken} minutes")

def SP500_corr_log():
    print(f"Start SP500_corr_log: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    start_time = time.time()
    global base_path, data_path, data_train_predict_path, daily_stock_path, save_path, prediction_path, data_start, data_middle, data_end, pre_data
    base_path = os.path.dirname(os.path.abspath(__file__))  # Huidige scriptmap
    print(f"base_path: {base_path}")
    data_path = os.path.join(base_path, "data", "S&P500")
    print(f"data_path: {data_path}")
    data_train_predict_path = os.path.join(data_path, f"data_train_predict_corr_log") #gpu_wvt, oldway_0.6, gpu_wvt
    print(f"data_train_predict_path: {data_train_predict_path}")
    daily_stock_path = os.path.join(data_path, f"daily_stock_corr_log") #gpu_wvt, oldway, gpu_wvt
    print(f"daily_stock_path: {daily_stock_path}")
    save_path = os.path.join(data_path, f"model_saved_rolingwindow_corr_log")
    os.makedirs(save_path, exist_ok=True)
    prediction_path = save_path
    total_data_points = len(os.listdir(data_train_predict_path))
    print(f"Total data points: {total_data_points}")
    val_len = 10
    window_len = 20
    rolling_start = total_data_points - window_len  # Laat genoeg ruimte over voor testdagen #inclusief
    rolling_end = total_data_points                 # Laatste dag waarop je kan voorspellen #exclusief
    for T in range(rolling_start, rolling_end):
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
    end_time = time.time()
    minutes_taken = round((end_time - start_time) / 60, 1)
    print(f"Done SP500_corr. Time taken: {minutes_taken} minutes") 

def SP500_cosineDSC():
    print(f"Start SP500_cosineDSC: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    start_time = time.time()
    global base_path, data_path, data_train_predict_path, daily_stock_path, save_path, prediction_path, data_start, data_middle, data_end, pre_data
    base_path = os.path.dirname(os.path.abspath(__file__))  # Huidige scriptmap
    print(f"base_path: {base_path}")
    data_path = os.path.join(base_path, "data", "S&P500")
    print(f"data_path: {data_path}")
    data_train_predict_path = os.path.join(data_path, f"data_train_predict_cosineDSC") #gpu_wvt, oldway_0.6, gpu_wvt
    print(f"data_train_predict_path: {data_train_predict_path}")
    daily_stock_path = os.path.join(data_path, f"daily_stock_cosineDSC") #gpu_wvt, oldway, gpu_wvt
    print(f"daily_stock_path: {daily_stock_path}")
    save_path = os.path.join(data_path, f"model_saved_rolingwindow_cosineDSC")
    os.makedirs(save_path, exist_ok=True)
    prediction_path = save_path
    total_data_points = len(os.listdir(data_train_predict_path))
    print(f"Total data points: {total_data_points}")
    val_len = 10
    window_len = 20
    rolling_start = total_data_points - window_len  # Laat genoeg ruimte over voor testdagen #inclusief
    rolling_end = total_data_points                 # Laatste dag waarop je kan voorspellen #exclusief
    for T in range(rolling_start, rolling_end):
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
    end_time = time.time()
    minutes_taken = round((end_time - start_time) / 60, 1)
    print(f"Done SP500_cosineDSC. Time taken: {minutes_taken} minutes")

def SP500_full():
    print(f"Start SP500_full: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    start_time = time.time()
    global base_path, data_path, data_train_predict_path, daily_stock_path, save_path, prediction_path, data_start, data_middle, data_end, pre_data
    base_path = os.path.dirname(os.path.abspath(__file__))  # Huidige scriptmap
    print(f"base_path: {base_path}")
    data_path = os.path.join(base_path, "data", "S&P500")
    print(f"data_path: {data_path}")
    data_train_predict_path = os.path.join(data_path, f"data_train_predict_DSE") #gpu_wvt, oldway_0.6, gpu_wvt
    print(f"data_train_predict_path: {data_train_predict_path}")
    daily_stock_path = os.path.join(data_path, f"daily_stock_DSE") #gpu_wvt, oldway, gpu_wvt
    print(f"daily_stock_path: {daily_stock_path}")
    save_path = os.path.join(data_path, f"model_saved_rolingwindow_DSE1")
    os.makedirs(save_path, exist_ok=True)
    prediction_path = save_path
    total_data_points = len(os.listdir(data_train_predict_path))
    print(f"Total data points: {total_data_points}")
    val_len = 10
    window_len = 20
    rolling_start = total_data_points - window_len  # Laat genoeg ruimte over voor testdagen #inclusief
    rolling_end = total_data_points                 # Laatste dag waarop je kan voorspellen #exclusief
    for T in range(rolling_start, rolling_end):
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
    end_time = time.time()
    minutes_taken = round((end_time - start_time) / 60, 1)
    print(f"Done SP500_full. Time taken: {minutes_taken} minutes")

def SP500_STATIC():
    print(f"Start SP500_STATIC: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    start_time = time.time()
    global base_path, data_path, data_train_predict_path, daily_stock_path, save_path, prediction_path, data_start, data_middle, data_end, pre_data
    base_path = os.path.dirname(os.path.abspath(__file__))  # Huidige scriptmap
    print(f"base_path: {base_path}")
    data_path = os.path.join(base_path, "data", "S&P500")
    print(f"data_path: {data_path}")
    data_train_predict_path = os.path.join(data_path, f"data_train_predict_STATIC") #gpu_wvt, oldway_0.6, gpu_wvt
    print(f"data_train_predict_path: {data_train_predict_path}")
    daily_stock_path = os.path.join(data_path, f"daily_stock_STATIC") #gpu_wvt, oldway, gpu_wvt
    print(f"daily_stock_path: {daily_stock_path}")
    save_path = os.path.join(data_path, f"model_saved_rolingwindow_STATIC")
    os.makedirs(save_path, exist_ok=True)
    prediction_path = save_path
    total_data_points = len(os.listdir(data_train_predict_path))
    print(f"Total data points: {total_data_points}")
    val_len = 10
    window_len = 10
    rolling_start = total_data_points - window_len  # Laat genoeg ruimte over voor testdagen #inclusief
    rolling_end = total_data_points                 # Laatste dag waarop je kan voorspellen #exclusief
    for T in range(rolling_start, rolling_end):
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
    end_time = time.time()
    minutes_taken = round((end_time - start_time) / 60, 1)
    print(f"Done SP500_STATIC. Time taken: {minutes_taken} minutes")

def testbatch_mini_corr():
    print(f"Start testbatch_mini_corr: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    start_time = time.time()
    global base_path, data_path, data_train_predict_path, daily_stock_path, save_path, prediction_path, data_start, data_middle, data_end, pre_data
    base_path = os.path.dirname(os.path.abspath(__file__))  # Huidige scriptmap
    print(f"base_path: {base_path}")
    data_path = os.path.join(base_path, "data", "testbatch_mini")
    print(f"data_path: {data_path}")
    data_train_predict_path = os.path.join(data_path, f"data_train_predict_corr") #gpu_wvt, oldway_0.6, gpu_wvt
    print(f"data_train_predict_path: {data_train_predict_path}")
    daily_stock_path = os.path.join(data_path, f"daily_stock_corr") #gpu_wvt, oldway, gpu_wvt
    print(f"daily_stock_path: {daily_stock_path}")
    save_path = os.path.join(data_path, f"model_saved_rolingwindow_corr")
    os.makedirs(save_path, exist_ok=True)
    prediction_path = save_path
    total_data_points = len(os.listdir(data_train_predict_path))
    print(f"Total data points: {total_data_points}")
    val_len = 10
    window_len = 20
    rolling_start = total_data_points - window_len  # Laat genoeg ruimte over voor testdagen #inclusief
    rolling_end = total_data_points                 # Laatste dag waarop je kan voorspellen #exclusief
    for T in range(rolling_start, rolling_end):
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
    end_time = time.time()
    minutes_taken = round((end_time - start_time) / 60, 1)
    print(f"Done testbatch_mini_corr. Time taken: {minutes_taken} minutes")

def testbatch_mini_onlycosine():
    print(f"Start testbatch_mini_onlycosine: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    start_time = time.time()
    global base_path, data_path, data_train_predict_path, daily_stock_path, save_path, prediction_path, data_start, data_middle, data_end, pre_data
    base_path = os.path.dirname(os.path.abspath(__file__))  # Huidige scriptmap
    print(f"base_path: {base_path}")
    data_path = os.path.join(base_path, "data", "testbatch_mini")
    print(f"data_path: {data_path}")
    data_train_predict_path = os.path.join(data_path, f"data_train_predict_onlycosine") #gpu_wvt, oldway_0.6, gpu_wvt
    print(f"data_train_predict_path: {data_train_predict_path}")
    daily_stock_path = os.path.join(data_path, f"daily_stock_onlycosine") #gpu_wvt, oldway, gpu_wvt
    print(f"daily_stock_path: {daily_stock_path}")
    save_path = os.path.join(data_path, f"model_saved_rolingwindow_onlycosine")
    os.makedirs(save_path, exist_ok=True)
    prediction_path = save_path
    total_data_points = len(os.listdir(data_train_predict_path))
    print(f"Total data points: {total_data_points}")
    val_len = 10
    window_len = 20
    rolling_start = total_data_points - window_len  # Laat genoeg ruimte over voor testdagen #inclusief
    rolling_end = total_data_points                 # Laatste dag waarop je kan voorspellen #exclusief
    for T in range(rolling_start, rolling_end):
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
    end_time = time.time()
    minutes_taken = round((end_time - start_time) / 60, 1)
    print(f"Done testbatch_mini_onlycosine. Time taken: {minutes_taken} minutes")

def testbatch_mini_cosineDSC():
    print(f"Start testbatch_mini_cosineDSC: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    start_time = time.time()
    global base_path, data_path, data_train_predict_path, daily_stock_path, save_path, prediction_path, data_start, data_middle, data_end, pre_data
    base_path = os.path.dirname(os.path.abspath(__file__))  # Huidige scriptmap
    print(f"base_path: {base_path}")
    data_path = os.path.join(base_path, "data", "testbatch_mini")
    print(f"data_path: {data_path}")
    data_train_predict_path = os.path.join(data_path, f"data_train_predict_cosineDSC") #gpu_wvt, oldway_0.6, gpu_wvt
    print(f"data_train_predict_path: {data_train_predict_path}")
    daily_stock_path = os.path.join(data_path, f"daily_stock_cosineDSC") #gpu_wvt, oldway, gpu_wvt
    print(f"daily_stock_path: {daily_stock_path}")
    save_path = os.path.join(data_path, f"model_saved_rolingwindow_cosineDSC")
    os.makedirs(save_path, exist_ok=True)
    prediction_path = save_path
    total_data_points = len(os.listdir(data_train_predict_path))
    print(f"Total data points: {total_data_points}")
    val_len = 10
    window_len = 20
    rolling_start = total_data_points - window_len  # Laat genoeg ruimte over voor testdagen #inclusief
    rolling_end = total_data_points                 # Laatste dag waarop je kan voorspellen #exclusief
    for T in range(rolling_start, rolling_end):
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
    end_time = time.time()
    minutes_taken = round((end_time - start_time) / 60, 1)
    print(f"Done testbatch_mini_cosineDSC. Time taken: {minutes_taken} minutes")

def testbatch_mini_full():
    print(f"Start testbatch_mini_full: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    start_time = time.time()
    global base_path, data_path, data_train_predict_path, daily_stock_path, save_path, prediction_path, data_start, data_middle, data_end, pre_data
    base_path = os.path.dirname(os.path.abspath(__file__))  # Huidige scriptmap
    print(f"base_path: {base_path}")
    data_path = os.path.join(base_path, "data", "testbatch_mini")
    print(f"data_path: {data_path}")
    data_train_predict_path = os.path.join(data_path, f"data_train_predict_DSE") #gpu_wvt, oldway_0.6, gpu_wvt
    print(f"data_train_predict_path: {data_train_predict_path}")
    daily_stock_path = os.path.join(data_path, f"daily_stock_DSE") #gpu_wvt, oldway, gpu_wvt
    print(f"daily_stock_path: {daily_stock_path}")
    save_path = os.path.join(data_path, f"model_saved_rolingwindow_DSE")
    os.makedirs(save_path, exist_ok=True)
    prediction_path = save_path
    total_data_points = len(os.listdir(data_train_predict_path))
    print(f"Total data points: {total_data_points}")
    val_len = 10
    window_len = 20
    rolling_start = total_data_points - window_len  # Laat genoeg ruimte over voor testdagen #inclusief
    rolling_end = total_data_points                 # Laatste dag waarop je kan voorspellen #exclusief
    for T in range(rolling_start, rolling_end):
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
    end_time = time.time()
    minutes_taken = round((end_time - start_time) / 60, 1)
    print(f"Done testbatch_mini_full. Time taken: {minutes_taken} minutes")

def nasdaq5batches_corr():
    print(f"Start nasdaq5batches_corr: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    start_time = time.time()
    global base_path, data_path, data_train_predict_path, daily_stock_path, save_path, prediction_path, data_start, data_middle, data_end, pre_data
    base_path = os.path.dirname(os.path.abspath(__file__))  # Huidige scriptmap
    print(f"base_path: {base_path}")
    for i in range(5):
        data_path = os.path.join(base_path, "data", "NASDAQ_batches_5_200")
        data_path = os.path.join(data_path, f"batch_{i+1}")
        print(f"data_path: {data_path}")
        data_train_predict_path = os.path.join(data_path, f"data_train_predict_corr") #gpu_wvt, oldway_0.6, gpu_wvt
        print(f"data_train_predict_path: {data_train_predict_path}")
        daily_stock_path = os.path.join(data_path, f"daily_stock_corr") #gpu_wvt, oldway, gpu_wvt
        print(f"daily_stock_path: {daily_stock_path}")
        save_path = os.path.join(data_path, f"model_saved_rolingwindow_corr")
        os.makedirs(save_path, exist_ok=True)
        prediction_path = save_path
        total_data_points = len(os.listdir(data_train_predict_path))
        print(f"Total data points: {total_data_points}")
        val_len = 10
        window_len = 20
        rolling_start = total_data_points - window_len  # Laat genoeg ruimte over voor testdagen #inclusief
        rolling_end = total_data_points                 # Laatste dag waarop je kan voorspellen #exclusief
        for T in range(rolling_start, rolling_end):
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
    end_time = time.time()
    minutes_taken = round((end_time - start_time) / 60, 1)
    print(f"Done nasdaq5batches_corr. Time taken: {minutes_taken} minutes") 

def nasdaq5batches_onlycosine():
    print(f"Start nasdaq5batches_onlycosine: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    start_time = time.time()
    global base_path, data_path, data_train_predict_path, daily_stock_path, save_path, prediction_path, data_start, data_middle, data_end, pre_data
    base_path = os.path.dirname(os.path.abspath(__file__))  # Huidige scriptmap
    print(f"base_path: {base_path}")
    for i in range(5):
        data_path = os.path.join(base_path, "data", "NASDAQ_batches_5_200")
        data_path = os.path.join(data_path, f"batch_{i+1}")
        print(f"data_path: {data_path}")
        data_train_predict_path = os.path.join(data_path, f"data_train_predict_onlycosine") #gpu_wvt, oldway_0.6, gpu_wvt
        print(f"data_train_predict_path: {data_train_predict_path}")
        daily_stock_path = os.path.join(data_path, f"daily_stock_onlycosine") #gpu_wvt, oldway, gpu_wvt
        print(f"daily_stock_path: {daily_stock_path}")
        save_path = os.path.join(data_path, f"model_saved_rolingwindow_onlycosine")
        os.makedirs(save_path, exist_ok=True)
        prediction_path = save_path
        total_data_points = len(os.listdir(data_train_predict_path))
        print(f"Total data points: {total_data_points}")
        val_len = 10
        window_len = 20
        rolling_start = total_data_points - window_len  # Laat genoeg ruimte over voor testdagen #inclusief
        rolling_end = total_data_points                 # Laatste dag waarop je kan voorspellen #exclusief
        for T in range(rolling_start, rolling_end):
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
    end_time = time.time()
    minutes_taken = round((end_time - start_time) / 60, 1)
    print(f"Done nasdaq5batches_onlycosine. Time taken: {minutes_taken} minutes") 

def nasdaq5batches_cosineDSC():
    print(f"Start nasdaq5batches_cosineDSC: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    start_time = time.time()
    global base_path, data_path, data_train_predict_path, daily_stock_path, save_path, prediction_path, data_start, data_middle, data_end, pre_data
    base_path = os.path.dirname(os.path.abspath(__file__))  # Huidige scriptmap
    print(f"base_path: {base_path}")
    for i in range(5):
        data_path = os.path.join(base_path, "data", "NASDAQ_batches_5_200")
        data_path = os.path.join(data_path, f"batch_{i+1}")
        print(f"data_path: {data_path}")
        data_train_predict_path = os.path.join(data_path, f"data_train_predict_cosineDSC") #gpu_wvt, oldway_0.6, gpu_wvt
        print(f"data_train_predict_path: {data_train_predict_path}")
        daily_stock_path = os.path.join(data_path, f"daily_stock_cosineDSC") #gpu_wvt, oldway, gpu_wvt
        print(f"daily_stock_path: {daily_stock_path}")
        save_path = os.path.join(data_path, f"model_saved_rolingwindow_cosineDSC")
        os.makedirs(save_path, exist_ok=True)
        prediction_path = save_path
        total_data_points = len(os.listdir(data_train_predict_path))
        print(f"Total data points: {total_data_points}")
        val_len = 10
        window_len = 20
        rolling_start = total_data_points - window_len  # Laat genoeg ruimte over voor testdagen #inclusief
        rolling_end = total_data_points                 # Laatste dag waarop je kan voorspellen #exclusief
        for T in range(rolling_start, rolling_end):
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
    end_time = time.time()
    minutes_taken = round((end_time - start_time) / 60, 1)
    print(f"Done nasdaq5batches_cosineDSC. Time taken: {minutes_taken} minutes") 

def nasdaq5batches_full():
    print(f"Start nasdaq5batches_full: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    start_time = time.time()
    global base_path, data_path, data_train_predict_path, daily_stock_path, save_path, prediction_path, data_start, data_middle, data_end, pre_data
    base_path = os.path.dirname(os.path.abspath(__file__))  # Huidige scriptmap
    print(f"base_path: {base_path}")
    for i in range(5):
        data_path = os.path.join(base_path, "data", "NASDAQ_batches_5_200")
        data_path = os.path.join(data_path, f"batch_{i+1}")
        print(f"data_path: {data_path}")
        data_train_predict_path = os.path.join(data_path, f"data_train_predict_DSE") #gpu_wvt, oldway_0.6, gpu_wvt
        print(f"data_train_predict_path: {data_train_predict_path}")
        daily_stock_path = os.path.join(data_path, f"daily_stock_DSE") #gpu_wvt, oldway, gpu_wvt
        print(f"daily_stock_path: {daily_stock_path}")
        save_path = os.path.join(data_path, f"model_saved_rolingwindow_DSE")
        os.makedirs(save_path, exist_ok=True)
        prediction_path = save_path
        total_data_points = len(os.listdir(data_train_predict_path))
        print(f"Total data points: {total_data_points}")
        val_len = 10
        window_len = 20
        rolling_start = total_data_points - window_len  # Laat genoeg ruimte over voor testdagen #inclusief
        rolling_end = total_data_points                 # Laatste dag waarop je kan voorspellen #exclusief
        for T in range(rolling_start, rolling_end):
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
    end_time = time.time()
    minutes_taken = round((end_time - start_time) / 60, 1)
    print(f"Done nasdaq5batches_full. Time taken: {minutes_taken} minutes") 


# nog te runnen:
# testbatch_mini_corr()
# testbatch_mini_onlycosine()
# testbatch_mini_cosineDSC()
# testbatch_mini_full()
# SP500_STATIC()


# bezig met runnen:
# CSI300_full_t1()
# CSI300_full_corr_t1()
# CSI300_full_gericht()
# CSI300_full()
# CSI300_full_ct()
# nasdaq5batches_full()
# nasdaq5batches_corr()
# CSI300_onlycosine()
# CSI300_cosineDSC()
# CSI300_corr()
# CSI300_STATIC()
# SP500_onlycosine()
# SP500_cosineDSC()
# nasdaq5batches_onlycosine()
# nasdaq5batches_cosineDSC()
# SP500_full()
# SP500_corr()
# CSI300_onlycosine()
# succesvol gerund:


# # al de functies klaar om te runnen
# CSI300_corr()
# CSI300_onlycosine()
# CSI300_cosineDSC()
# CSI300_full()
# SP500_corr()
# SP500_corr_log()
# SP500_onlycosine()
# SP500_cosineDSC()
# SP500_full()
# testbatch_mini_corr()
# testbatch_mini_onlycosine()
# testbatch_mini_cosineDSC()
# testbatch_mini_full()
# nasdaq5batches_corr()
# nasdaq5batches_onlycosine()
# nasdaq5batches_cosineDSC()
# nasdaq5batches_full()
# SP500_STATIC()
# CSI300_STATIC()




# region rerun (t1)

def CSI300_onlycosine():
    print(f"Start CSI300_onlycosine: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    start_time = time.time()
    global base_path, data_path, data_train_predict_path, daily_stock_path, save_path, prediction_path, data_start, data_middle, data_end, pre_data
    base_path = os.path.dirname(os.path.abspath(__file__))  # Huidige scriptmap
    print(f"base_path: {base_path}")
    data_path = os.path.join(base_path, "data", "CSI300")
    print(f"data_path: {data_path}")
    data_train_predict_path = os.path.join(data_path, f"data_train_predict_onlycosine") #gpu_wvt, oldway_0.6, gpu_wvt
    print(f"data_train_predict_path: {data_train_predict_path}")
    daily_stock_path = os.path.join(data_path, f"daily_stock_onlycosine") #gpu_wvt, oldway, gpu_wvt
    print(f"daily_stock_path: {daily_stock_path}")
    save_path = os.path.join(data_path, f"model_saved_rolingwindow_onlycosine")
    os.makedirs(save_path, exist_ok=True)
    prediction_path = save_path
    total_data_points = len(os.listdir(data_train_predict_path))
    print(f"Total data points: {total_data_points}")
    val_len = 10
    window_len = 20
    rolling_start = total_data_points - window_len  # Laat genoeg ruimte over voor testdagen #inclusief
    rolling_end = total_data_points                 # Laatste dag waarop je kan voorspellen #exclusief
    for T in range(rolling_start, rolling_end):
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
    end_time = time.time()
    minutes_taken = round((end_time - start_time) / 60, 1)
    print(f"Done CSI300_onlycosine. Time taken: {minutes_taken} minutes") 

def CSI300_corr():
    print(f"Start CSI300_corr: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    start_time = time.time()
    global base_path, data_path, data_train_predict_path, daily_stock_path, save_path, prediction_path, data_start, data_middle, data_end, pre_data
    base_path = os.path.dirname(os.path.abspath(__file__))  # Huidige scriptmap
    print(f"base_path: {base_path}")
    data_path = os.path.join(base_path, "data", "CSI300")
    print(f"data_path: {data_path}")
    data_train_predict_path = os.path.join(data_path, f"data_train_predict_corr") #gpu_wvt, oldway_0.6, gpu_wvt
    print(f"data_train_predict_path: {data_train_predict_path}")
    daily_stock_path = os.path.join(data_path, f"daily_stock_corr") #gpu_wvt, oldway, gpu_wvt
    print(f"daily_stock_path: {daily_stock_path}")
    save_path = os.path.join(data_path, f"model_saved_rolingwindow_corr")
    os.makedirs(save_path, exist_ok=True)
    prediction_path = save_path
    total_data_points = len(os.listdir(data_train_predict_path))
    print(f"Total data points: {total_data_points}")
    val_len = 10
    window_len = 20
    rolling_start = total_data_points - window_len  # Laat genoeg ruimte over voor testdagen #inclusief
    rolling_end = total_data_points                 # Laatste dag waarop je kan voorspellen #exclusief
    for T in range(rolling_start, rolling_end):
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
    end_time = time.time()
    minutes_taken = round((end_time - start_time) / 60, 1)
    print(f"Done CSI300_corr. Time taken: {minutes_taken} minutes") 

def CSI300_cosineDSC_t1():
    print(f"Start CSI300_cosineDSC_t1: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    start_time = time.time()
    global base_path, data_path, data_train_predict_path, daily_stock_path, save_path, prediction_path, data_start, data_middle, data_end, pre_data
    base_path = os.path.dirname(os.path.abspath(__file__))  # Huidige scriptmap
    print(f"base_path: {base_path}")
    data_path = os.path.join(base_path, "data", "CSI300")
    print(f"data_path: {data_path}")
    data_train_predict_path = os.path.join(data_path, f"data_train_predict_cosineDSC_t1") #gpu_wvt, oldway_0.6, gpu_wvt
    print(f"data_train_predict_path: {data_train_predict_path}")
    daily_stock_path = os.path.join(data_path, f"daily_stock_cosineDSC_t1") #gpu_wvt, oldway, gpu_wvt
    print(f"daily_stock_path: {daily_stock_path}")
    save_path = os.path.join(data_path, f"model_saved_rolingwindow_cosineDSC_t1")
    os.makedirs(save_path, exist_ok=True)
    prediction_path = save_path
    total_data_points = len(os.listdir(data_train_predict_path))
    print(f"Total data points: {total_data_points}")
    val_len = 10
    window_len = 20
    rolling_start = total_data_points - window_len  # Laat genoeg ruimte over voor testdagen #inclusief
    rolling_end = total_data_points                 # Laatste dag waarop je kan voorspellen #exclusief
    for T in range(rolling_start, rolling_end):
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
    end_time = time.time()
    minutes_taken = round((end_time - start_time) / 60, 1)
    print(f"Done CSI300_cosineDSC_t1. Time taken: {minutes_taken} minutes") 

def CSI300_corrDSC_t1():
    print(f"Start CSI300_corrDSC_t1: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    start_time = time.time()
    global base_path, data_path, data_train_predict_path, daily_stock_path, save_path, prediction_path, data_start, data_middle, data_end, pre_data
    base_path = os.path.dirname(os.path.abspath(__file__))  # Huidige scriptmap
    print(f"base_path: {base_path}")
    data_path = os.path.join(base_path, "data", "CSI300")
    print(f"data_path: {data_path}")
    data_train_predict_path = os.path.join(data_path, f"data_train_predict_corrDSC_t1") #gpu_wvt, oldway_0.6, gpu_wvt
    print(f"data_train_predict_path: {data_train_predict_path}")
    daily_stock_path = os.path.join(data_path, f"daily_stock_corrDSC_t1") #gpu_wvt, oldway, gpu_wvt
    print(f"daily_stock_path: {daily_stock_path}")
    save_path = os.path.join(data_path, f"model_saved_rolingwindow_corrDSC_t1")
    os.makedirs(save_path, exist_ok=True)
    prediction_path = save_path
    total_data_points = len(os.listdir(data_train_predict_path))
    print(f"Total data points: {total_data_points}")
    val_len = 10
    window_len = 20
    rolling_start = total_data_points - window_len  # Laat genoeg ruimte over voor testdagen #inclusief
    rolling_end = total_data_points                 # Laatste dag waarop je kan voorspellen #exclusief
    for T in range(rolling_start, rolling_end):
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
    end_time = time.time()
    minutes_taken = round((end_time - start_time) / 60, 1)
    print(f"Done CSI300_corrDSC_t1. Time taken: {minutes_taken} minutes")

def CSI300_STATIC_t1():
    print(f"Start CSI300_STATIC_t1: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    start_time = time.time()
    global base_path, data_path, data_train_predict_path, daily_stock_path, save_path, prediction_path, data_start, data_middle, data_end, pre_data
    base_path = os.path.dirname(os.path.abspath(__file__))  # Huidige scriptmap
    print(f"base_path: {base_path}")
    data_path = os.path.join(base_path, "data", "CSI300")
    print(f"data_path: {data_path}")
    data_train_predict_path = os.path.join(data_path, f"data_train_predict_STATIC_t1") #gpu_wvt, oldway_0.6, gpu_wvt
    print(f"data_train_predict_path: {data_train_predict_path}")
    daily_stock_path = os.path.join(data_path, f"daily_stock_STATIC_t1") #gpu_wvt, oldway, gpu_wvt
    print(f"daily_stock_path: {daily_stock_path}")
    save_path = os.path.join(data_path, f"model_saved_rolingwindow_STATIC_t1")
    os.makedirs(save_path, exist_ok=True)
    prediction_path = save_path
    total_data_points = len(os.listdir(data_train_predict_path))
    print(f"Total data points: {total_data_points}")
    val_len = 10
    window_len = 20
    rolling_start = total_data_points - window_len  # Laat genoeg ruimte over voor testdagen #inclusief
    rolling_end = total_data_points                 # Laatste dag waarop je kan voorspellen #exclusief
    for T in range(rolling_start, rolling_end):
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
    end_time = time.time()
    minutes_taken = round((end_time - start_time) / 60, 1)
    print(f"Done CSI300_STATIC_t1. Time taken: {minutes_taken} minutes")

def CSI300_STATICcorr_t1():
    print(f"Start CSI300_STATICcorr_t1: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    start_time = time.time()
    global base_path, data_path, data_train_predict_path, daily_stock_path, save_path, prediction_path, data_start, data_middle, data_end, pre_data
    base_path = os.path.dirname(os.path.abspath(__file__))  # Huidige scriptmap
    print(f"base_path: {base_path}")
    data_path = os.path.join(base_path, "data", "CSI300")
    print(f"data_path: {data_path}")
    data_train_predict_path = os.path.join(data_path, f"data_train_predict_STATICcorr_t1") #gpu_wvt, oldway_0.6, gpu_wvt
    print(f"data_train_predict_path: {data_train_predict_path}")
    daily_stock_path = os.path.join(data_path, f"daily_stock_STATICcorr_t1") #gpu_wvt, oldway, gpu_wvt
    print(f"daily_stock_path: {daily_stock_path}")
    save_path = os.path.join(data_path, f"model_saved_rolingwindow_STATICcorr_t1")
    os.makedirs(save_path, exist_ok=True)
    prediction_path = save_path
    total_data_points = len(os.listdir(data_train_predict_path))
    print(f"Total data points: {total_data_points}")
    val_len = 10
    window_len = 20
    rolling_start = total_data_points - window_len  # Laat genoeg ruimte over voor testdagen #inclusief
    rolling_end = total_data_points                 # Laatste dag waarop je kan voorspellen #exclusief
    for T in range(rolling_start, rolling_end):
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
    end_time = time.time()
    minutes_taken = round((end_time - start_time) / 60, 1)
    print(f"Done CSI300_STATICcorr_t1. Time taken: {minutes_taken} minutes")

def CSI300_full_t1():
    print(f"Start CSI300_full_t1: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    start_time = time.time()
    global base_path, data_path, data_train_predict_path, daily_stock_path, save_path, prediction_path, data_start, data_middle, data_end, pre_data
    base_path = os.path.dirname(os.path.abspath(__file__))  # Huidige scriptmap
    print(f"base_path: {base_path}")
    data_path = os.path.join(base_path, "data", "CSI300")
    print(f"data_path: {data_path}")
    data_train_predict_path = os.path.join(data_path, f"data_train_predict_DSE_t1") #gpu_wvt, oldway_0.6, gpu_wvt
    print(f"data_train_predict_path: {data_train_predict_path}")
    daily_stock_path = os.path.join(data_path, f"daily_stock_DSE_t1") #gpu_wvt, oldway, gpu_wvt
    print(f"daily_stock_path: {daily_stock_path}")
    save_path = os.path.join(data_path, f"model_saved_rolingwindow_DSE_t1")
    os.makedirs(save_path, exist_ok=True)
    prediction_path = save_path
    total_data_points = len(os.listdir(data_train_predict_path))
    print(f"Total data points: {total_data_points}")
    val_len = 10
    window_len = 20
    rolling_start = total_data_points - window_len  # Laat genoeg ruimte over voor testdagen #inclusief
    rolling_end = total_data_points                 # Laatste dag waarop je kan voorspellen #exclusief
    for T in range(rolling_start, rolling_end):
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
    end_time = time.time()
    minutes_taken = round((end_time - start_time) / 60, 1)
    print(f"Done CSI300_full_t1. Time taken: {minutes_taken} minutes")

def CSI300_full_corr_t1():
    print(f"Start CSI300_full_corr_t1: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    start_time = time.time()
    global base_path, data_path, data_train_predict_path, daily_stock_path, save_path, prediction_path, data_start, data_middle, data_end, pre_data
    base_path = os.path.dirname(os.path.abspath(__file__))  # Huidige scriptmap
    print(f"base_path: {base_path}")
    data_path = os.path.join(base_path, "data", "CSI300")
    print(f"data_path: {data_path}")
    data_train_predict_path = os.path.join(data_path, f"data_train_predict_DSEcorr_t1") #gpu_wvt, oldway_0.6, gpu_wvt
    print(f"data_train_predict_path: {data_train_predict_path}")
    daily_stock_path = os.path.join(data_path, f"daily_stock_DSEcorr_t1") #gpu_wvt, oldway, gpu_wvt
    print(f"daily_stock_path: {daily_stock_path}")
    save_path = os.path.join(data_path, f"model_saved_rolingwindow_DSEcorr_t1")
    os.makedirs(save_path, exist_ok=True)
    prediction_path = save_path
    total_data_points = len(os.listdir(data_train_predict_path))
    print(f"Total data points: {total_data_points}")
    val_len = 10
    window_len = 20
    rolling_start = total_data_points - window_len  # Laat genoeg ruimte over voor testdagen #inclusief
    rolling_end = total_data_points                 # Laatste dag waarop je kan voorspellen #exclusief
    for T in range(rolling_start, rolling_end):
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
    end_time = time.time()
    minutes_taken = round((end_time - start_time) / 60, 1)
    print(f"Done CSI300_full_corr_t1. Time taken: {minutes_taken} minutes")

def SP500_onlycosine():
    print(f"Start SP500_onlycosine: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    start_time = time.time()
    global base_path, data_path, data_train_predict_path, daily_stock_path, save_path, prediction_path, data_start, data_middle, data_end, pre_data
    base_path = os.path.dirname(os.path.abspath(__file__))  # Huidige scriptmap
    print(f"base_path: {base_path}")
    data_path = os.path.join(base_path, "data", "S&P500")
    print(f"data_path: {data_path}")
    data_train_predict_path = os.path.join(data_path, f"data_train_predict_onlycosine") #gpu_wvt, oldway_0.6, gpu_wvt
    print(f"data_train_predict_path: {data_train_predict_path}")
    daily_stock_path = os.path.join(data_path, f"daily_stock_onlycosine") #gpu_wvt, oldway, gpu_wvt
    print(f"daily_stock_path: {daily_stock_path}")
    save_path = os.path.join(data_path, f"model_saved_rolingwindow_onlycosine")
    os.makedirs(save_path, exist_ok=True)
    prediction_path = save_path
    total_data_points = len(os.listdir(data_train_predict_path))
    print(f"Total data points: {total_data_points}")
    val_len = 10
    window_len = 20
    rolling_start = total_data_points - window_len  # Laat genoeg ruimte over voor testdagen #inclusief
    rolling_end = total_data_points                 # Laatste dag waarop je kan voorspellen #exclusief
    for T in range(rolling_start, rolling_end):
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
    end_time = time.time()
    minutes_taken = round((end_time - start_time) / 60, 1)
    print(f"Done SP500_onlycosine. Time taken: {minutes_taken} minutes") 

def SP500_corr():
    print(f"Start SP500_corr: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    start_time = time.time()
    global base_path, data_path, data_train_predict_path, daily_stock_path, save_path, prediction_path, data_start, data_middle, data_end, pre_data
    base_path = os.path.dirname(os.path.abspath(__file__))  # Huidige scriptmap
    print(f"base_path: {base_path}")
    data_path = os.path.join(base_path, "data", "S&P500")
    print(f"data_path: {data_path}")
    data_train_predict_path = os.path.join(data_path, f"data_train_predict_corr") #gpu_wvt, oldway_0.6, gpu_wvt
    print(f"data_train_predict_path: {data_train_predict_path}")
    daily_stock_path = os.path.join(data_path, f"daily_stock_corr") #gpu_wvt, oldway, gpu_wvt
    print(f"daily_stock_path: {daily_stock_path}")
    save_path = os.path.join(data_path, f"model_saved_rolingwindow_corr")
    os.makedirs(save_path, exist_ok=True)
    prediction_path = save_path
    total_data_points = len(os.listdir(data_train_predict_path))
    print(f"Total data points: {total_data_points}")
    val_len = 10
    window_len = 20
    rolling_start = total_data_points - window_len  # Laat genoeg ruimte over voor testdagen #inclusief
    rolling_end = total_data_points                 # Laatste dag waarop je kan voorspellen #exclusief
    for T in range(rolling_start, rolling_end):
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
    end_time = time.time()
    minutes_taken = round((end_time - start_time) / 60, 1)
    print(f"Done SP500_corr. Time taken: {minutes_taken} minutes") 

def SP500_cosineDSC_t1():
    print(f"Start SP500_cosineDSC_t1: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    start_time = time.time()
    global base_path, data_path, data_train_predict_path, daily_stock_path, save_path, prediction_path, data_start, data_middle, data_end, pre_data
    base_path = os.path.dirname(os.path.abspath(__file__))  # Huidige scriptmap
    print(f"base_path: {base_path}")
    data_path = os.path.join(base_path, "data", "S&P500")
    print(f"data_path: {data_path}")
    data_train_predict_path = os.path.join(data_path, f"data_train_predict_cosineDSC_t1") #gpu_wvt, oldway_0.6, gpu_wvt
    print(f"data_train_predict_path: {data_train_predict_path}")
    daily_stock_path = os.path.join(data_path, f"daily_stock_cosineDSC_t1") #gpu_wvt, oldway, gpu_wvt
    print(f"daily_stock_path: {daily_stock_path}")
    save_path = os.path.join(data_path, f"model_saved_rolingwindow_cosineDSC_t1")
    os.makedirs(save_path, exist_ok=True)
    prediction_path = save_path
    total_data_points = len(os.listdir(data_train_predict_path))
    print(f"Total data points: {total_data_points}")
    val_len = 10
    window_len = 6
    rolling_start = total_data_points - window_len  # Laat genoeg ruimte over voor testdagen #inclusief
    rolling_end = total_data_points                 # Laatste dag waarop je kan voorspellen #exclusief
    for T in range(rolling_start, rolling_end):
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
    end_time = time.time()
    minutes_taken = round((end_time - start_time) / 60, 1)
    print(f"Done SP500_cosineDSC_t1. Time taken: {minutes_taken} minutes") 

def SP500_corrDSC_t1():
    print(f"Start SP500_corrDSC_t1: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    start_time = time.time()
    global base_path, data_path, data_train_predict_path, daily_stock_path, save_path, prediction_path, data_start, data_middle, data_end, pre_data
    base_path = os.path.dirname(os.path.abspath(__file__))  # Huidige scriptmap
    print(f"base_path: {base_path}")
    data_path = os.path.join(base_path, "data", "S&P500")
    print(f"data_path: {data_path}")
    data_train_predict_path = os.path.join(data_path, f"data_train_predict_corrDSC_t1") #gpu_wvt, oldway_0.6, gpu_wvt
    print(f"data_train_predict_path: {data_train_predict_path}")
    daily_stock_path = os.path.join(data_path, f"daily_stock_corrDSC_t1") #gpu_wvt, oldway, gpu_wvt
    print(f"daily_stock_path: {daily_stock_path}")
    save_path = os.path.join(data_path, f"model_saved_rolingwindow_corrDSC_t1")
    os.makedirs(save_path, exist_ok=True)
    prediction_path = save_path
    total_data_points = len(os.listdir(data_train_predict_path))
    print(f"Total data points: {total_data_points}")
    val_len = 10
    window_len = 20
    rolling_start = total_data_points - window_len  # Laat genoeg ruimte over voor testdagen #inclusief
    rolling_end = total_data_points                 # Laatste dag waarop je kan voorspellen #exclusief
    for T in range(rolling_start, rolling_end):
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
    end_time = time.time()
    minutes_taken = round((end_time - start_time) / 60, 1)
    print(f"Done SP500_corrDSC_t1. Time taken: {minutes_taken} minutes")

def SP500_STATIC_t1():
    print(f"Start SP500_STATIC_t1: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    start_time = time.time()
    global base_path, data_path, data_train_predict_path, daily_stock_path, save_path, prediction_path, data_start, data_middle, data_end, pre_data
    base_path = os.path.dirname(os.path.abspath(__file__))  # Huidige scriptmap
    print(f"base_path: {base_path}")
    data_path = os.path.join(base_path, "data", "S&P500")
    print(f"data_path: {data_path}")
    data_train_predict_path = os.path.join(data_path, f"data_train_predict_STATIC_t1") #gpu_wvt, oldway_0.6, gpu_wvt
    print(f"data_train_predict_path: {data_train_predict_path}")
    daily_stock_path = os.path.join(data_path, f"daily_stock_STATIC_t1") #gpu_wvt, oldway, gpu_wvt
    print(f"daily_stock_path: {daily_stock_path}")
    save_path = os.path.join(data_path, f"model_saved_rolingwindow_STATIC_t1")
    os.makedirs(save_path, exist_ok=True)
    prediction_path = save_path
    total_data_points = len(os.listdir(data_train_predict_path))
    print(f"Total data points: {total_data_points}")
    val_len = 10
    window_len = 20
    rolling_start = total_data_points - window_len  # Laat genoeg ruimte over voor testdagen #inclusief
    rolling_end = total_data_points                 # Laatste dag waarop je kan voorspellen #exclusief
    for T in range(rolling_start, rolling_end):
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
    end_time = time.time()
    minutes_taken = round((end_time - start_time) / 60, 1)
    print(f"Done SP500_STATIC_t1. Time taken: {minutes_taken} minutes")

def SP500_STATICcorr_t1():
    print(f"Start SP500_STATICcorr_t1: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    start_time = time.time()
    global base_path, data_path, data_train_predict_path, daily_stock_path, save_path, prediction_path, data_start, data_middle, data_end, pre_data
    base_path = os.path.dirname(os.path.abspath(__file__))  # Huidige scriptmap
    print(f"base_path: {base_path}")
    data_path = os.path.join(base_path, "data", "S&P500")
    print(f"data_path: {data_path}")
    data_train_predict_path = os.path.join(data_path, f"data_train_predict_STATICcorr_t1") #gpu_wvt, oldway_0.6, gpu_wvt
    print(f"data_train_predict_path: {data_train_predict_path}")
    daily_stock_path = os.path.join(data_path, f"daily_stock_STATICcorr_t1") #gpu_wvt, oldway, gpu_wvt
    print(f"daily_stock_path: {daily_stock_path}")
    save_path = os.path.join(data_path, f"model_saved_rolingwindow_STATICcorr_t1")
    os.makedirs(save_path, exist_ok=True)
    prediction_path = save_path
    total_data_points = len(os.listdir(data_train_predict_path))
    print(f"Total data points: {total_data_points}")
    val_len = 10
    window_len = 20
    rolling_start = total_data_points - window_len  # Laat genoeg ruimte over voor testdagen #inclusief
    rolling_end = total_data_points                 # Laatste dag waarop je kan voorspellen #exclusief
    for T in range(rolling_start, rolling_end):
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
    end_time = time.time()
    minutes_taken = round((end_time - start_time) / 60, 1)
    print(f"Done SP500_STATICcorr_t1. Time taken: {minutes_taken} minutes")

def SP500_full_t1():
    print(f"Start SP500_full_t1: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    start_time = time.time()
    global base_path, data_path, data_train_predict_path, daily_stock_path, save_path, prediction_path, data_start, data_middle, data_end, pre_data
    base_path = os.path.dirname(os.path.abspath(__file__))  # Huidige scriptmap
    print(f"base_path: {base_path}")
    data_path = os.path.join(base_path, "data", "S&P500")
    print(f"data_path: {data_path}")
    data_train_predict_path = os.path.join(data_path, f"data_train_predict_DSE_t1") #gpu_wvt, oldway_0.6, gpu_wvt
    print(f"data_train_predict_path: {data_train_predict_path}")
    daily_stock_path = os.path.join(data_path, f"daily_stock_DSE_t1") #gpu_wvt, oldway, gpu_wvt
    print(f"daily_stock_path: {daily_stock_path}")
    save_path = os.path.join(data_path, f"model_saved_rolingwindow_DSE_t1")
    os.makedirs(save_path, exist_ok=True)
    prediction_path = save_path
    total_data_points = len(os.listdir(data_train_predict_path))
    print(f"Total data points: {total_data_points}")
    val_len = 10
    window_len = 20
    rolling_start = total_data_points - window_len  # Laat genoeg ruimte over voor testdagen #inclusief
    rolling_end = total_data_points                 # Laatste dag waarop je kan voorspellen #exclusief
    for T in range(rolling_start, rolling_end):
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
    end_time = time.time()
    minutes_taken = round((end_time - start_time) / 60, 1)
    print(f"Done SP500_full_t1. Time taken: {minutes_taken} minutes")

def SP500_full_corr_t1():
    print(f"Start SP500_full_corr_t1: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    start_time = time.time()
    global base_path, data_path, data_train_predict_path, daily_stock_path, save_path, prediction_path, data_start, data_middle, data_end, pre_data
    base_path = os.path.dirname(os.path.abspath(__file__))  # Huidige scriptmap
    print(f"base_path: {base_path}")
    data_path = os.path.join(base_path, "data", "S&P500")
    print(f"data_path: {data_path}")
    data_train_predict_path = os.path.join(data_path, f"data_train_predict_DSEcorr_t1") #gpu_wvt, oldway_0.6, gpu_wvt
    print(f"data_train_predict_path: {data_train_predict_path}")
    daily_stock_path = os.path.join(data_path, f"daily_stock_DSEcorr_t1") #gpu_wvt, oldway, gpu_wvt
    print(f"daily_stock_path: {daily_stock_path}")
    save_path = os.path.join(data_path, f"model_saved_rolingwindow_DSEcorr_t1")
    os.makedirs(save_path, exist_ok=True)
    prediction_path = save_path
    total_data_points = len(os.listdir(data_train_predict_path))
    print(f"Total data points: {total_data_points}")
    val_len = 10
    window_len = 20
    rolling_start = total_data_points - window_len  # Laat genoeg ruimte over voor testdagen #inclusief
    rolling_end = total_data_points                 # Laatste dag waarop je kan voorspellen #exclusief
    for T in range(rolling_start, rolling_end):
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
    end_time = time.time()
    minutes_taken = round((end_time - start_time) / 60, 1)
    print(f"Done SP500_full_corr_t1. Time taken: {minutes_taken} minutes")


# # everything
# CSI300_onlycosine()
# CSI300_corr()
# CSI300_cosineDSC_t1()
# CSI300_corrDSC_t1()
# CSI300_STATIC_t1
# CSI300_STATICcorr_t1
# CSI300_full_t1()
# CSI300_full_corr_t1()
# SP500_onlycosine()
# SP500_corr()
# SP500_cosineDSC_t1()
# SP500_corrDSC_t1()
# SP500_STATIC_t1
# SP500_STATICcorr_t1
# SP500_full_t1()
# SP500_full_corr_t1()


# # already done
# CSI300_onlycosine()
# CSI300_corr()
# SP500_onlycosine()
# SP500_corr()
# CSI300_full_t1()
# CSI300_full_corr_t1()

# # nog te doen
# SP500_full_t1()

# SP500_cosineDSC_t1()
SP500_STATIC_t1()
# CSI300_cosineDSC_t1()
# CSI300_corrDSC_t1()
# CSI300_STATIC_t1
# CSI300_STATICcorr_t1

# SP500_corrDSC_t1()

# SP500_STATICcorr_t1
# SP500_full_corr_t1()