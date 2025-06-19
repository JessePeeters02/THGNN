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
# data_path = os.path.join(base_path, "data", "testbatch2")
# print(f"data_path: {data_path}")
# data_train_predict_map = os.path.join(data_path, "data_train_predict_corr") #gpu_wvt, oldway_0.6, gpu_wvt
# print(f"data_train_predict_path: {data_train_predict_map}")
# daily_stock_map = os.path.join(data_path, "daily_stock_corr") #gpu_wvt, oldway, gpu_wvt
# print(f"daily_stock_path: {daily_stock_map}")
# save_map = os.path.join(data_path, "model_saved_corr_TEbig")
# os.makedirs(save_map, exist_ok=True)
# prediction_map = os.path.join(data_path, "prediction_corr_TEbig")
# os.makedirs(prediction_map, exist_ok=True)

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(device)

class Args:
    def __init__(self, gpu=0, subtask="regression"): #regression or classification_binare, also switch: trainer.py 31/32 and thgnn.py 128/129
        # device
        self.gpu = str(1)
        self.device = 'cuda'
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
        self.batch_size = 1
        self.loss_fcn = mse_loss
        # save model settings
        #self.save_path = os.path.join(os.path.abspath('.'), "/home/THGNN-main/data/model_saved/")
        self.save_path = save_path
        self.load_path = self.save_path
        self.save_name = self.model_name + "_hidden_" + str(self.hidden_dim) + "_head_" + str(self.num_heads) + \
                         "_outfeat_" + str(self.out_features) + "_batchsize_" + str(self.batch_size)
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
    dataset = AllGraphDataSampler(base_dir=data_train_predict_path, data_start=data_start,
                              data_middle=data_middle, data_end=data_end)
    # print(f"Aantal samples in dataset: {len(dataset)}")
    val_dataset = AllGraphDataSampler(base_dir=data_train_predict_path, mode="val", data_start=data_start,
                                  data_middle=data_middle, data_end=data_end)
    dataset_loader = DataLoader(dataset, batch_size=args.batch_size, pin_memory=False, collate_fn=lambda x: x)
    val_dataset_loader = DataLoader(val_dataset, batch_size=1, pin_memory=False)
    model = eval(args.model_name)(hidden_dim=args.hidden_dim, num_heads=args.num_heads,
                                  out_features=args.out_features).to(args.device)

    # train
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    cold_scheduler = StepLR(optimizer=optimizer, step_size=5000, gamma=0.9, last_epoch=-1)
    default_scheduler = cold_scheduler
    print('start training')
    for epoch in range(args.max_epochs):
        train_loss = train_epoch(epoch=epoch, args=args, model=model, dataset_train=dataset_loader,
                                 optimizer=optimizer, scheduler=default_scheduler, loss_fcn=args.loss_fcn)
        if (epoch+1) % args.epochs_eval == 0:
            eval_loss, _ = eval_epoch(args=args, model=model, dataset_eval=val_dataset_loader, loss_fcn=args.loss_fcn)
            print('Epoch: {}/{}, train loss: {:.6f}, val loss: {:.6f}'.format(epoch + 1, args.max_epochs, train_loss, eval_loss))
        else:
            print('Epoch: {}/{}, train loss: {:.6f}'.format(epoch + 1, args.max_epochs, train_loss))
        if (epoch + 1) % args.epochs_save_by == 0:
            print("save model!")
            state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch + 1}
            torch.save(state, os.path.join(args.save_path, pre_data + "_epoch_" + str(epoch + 1) + ".dat"))

    # predict
    epoch = 19
    checkpoint = torch.load(os.path.join(args.load_path, pre_data + "_epoch_" + str(epoch + 1) + ".dat"))
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    #new
    df_weights = pd.DataFrame()

    data_files = daily_stock_path
    data_code = sorted(os.listdir(data_files))
    data_code_last = data_code[data_middle:data_end]
    df_score=pd.DataFrame()
    for i in tqdm(range(len(val_dataset))):
        file_path = os.path.join(daily_stock_path, data_code_last[i])
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, dtype=object)
        else:
            print(f"File {file_path} not found!")
        df = pd.read_csv(os.path.join(daily_stock_path, data_code_last[i]), dtype=object)
        tmp_data = val_dataset[i]
        pos_adj, neg_adj, features, labels, mask = extract_data(tmp_data, args.device)
        model.eval()
        with torch.no_grad():
            logits, weights = model(features, pos_adj, neg_adj, requires_weight=True)
        result = logits.data.cpu().numpy().tolist()
        result_new = []
        for j in range(len(result)):
            result_new.append(result[j][0])
        res = {"score": result_new}
        res = DataFrame(res)
        df['score'] = res
        df_score=pd.concat([df_score,df])
        # Sla gedetailleerde gewichten per sample op
        pos_weights = weights["pos_attn_weights"].cpu().numpy()  # Shape: [num_heads, num_nodes, num_nodes]
        neg_weights = weights["neg_attn_weights"].cpu().numpy()
        sem_weights = weights["sem_attn_weights"].cpu().numpy()  # Shape: [num_nodes, 3] (beta_self, beta_pos, beta_neg)

        # Bereken statistieken voor POSITIEVE edges
        pos_weights_flat = pos_weights.flatten()  # Maak 1D voor statistieken
        pos_stats = {
            "pos_weight_mean": np.mean(pos_weights_flat),
            "pos_weight_std": np.std(pos_weights_flat),
            "pos_weight_max": np.max(pos_weights_flat),
            "pos_weight_min": np.min(pos_weights_flat),
            "pos_weight_median": np.median(pos_weights_flat),
        }

        # Hetzelfde voor NEGATIEVE edges
        neg_weights_flat = neg_weights.flatten()
        neg_stats = {
            "neg_weight_mean": np.mean(neg_weights_flat),
            "neg_weight_std": np.std(neg_weights_flat),
            "neg_weight_max": np.max(neg_weights_flat),
            "neg_weight_min": np.min(neg_weights_flat),
            "neg_weight_median": np.median(neg_weights_flat),
        }

        # Beta-statistieken (self/pos/neg)
        beta_stats = {
            "beta_self_mean": np.mean(sem_weights[:, 0]),
            "beta_self_std": np.std(sem_weights[:, 0]),
            "beta_pos_mean": np.mean(sem_weights[:, 1]),
            "beta_pos_std": np.std(sem_weights[:, 1]),
            "beta_neg_mean": np.mean(sem_weights[:, 2]),
            "beta_neg_std": np.std(sem_weights[:, 2]),
        }

        # Voeg alles samen in een DataFrame
        df_weights = pd.concat([df_weights, pd.DataFrame({
            **pos_stats,
            **neg_stats,
            **beta_stats,
            # Optioneel: Sample-ID voor tracking
            "sample_id": [i]  
        })])

    final_means = df_weights.mean(numeric_only=True).to_dict()  # Bereken gemiddelde van alle numerieke kolommen
    final_means["sample_id"] = "TOTAAL"  # Markeer als totaalrij

    # Voeg toe aan df_weights
    df_weights = pd.concat([
        df_weights,
        pd.DataFrame([final_means])  # Voeg als nieuwe rij toe
    ], ignore_index=True)

        #df.to_csv('prediction/' + data_code_last[i], encoding='utf-8-sig', index=False)
    df_score.to_csv(os.path.join(prediction_path, "pred.csv"))
    df_weights.to_csv(os.path.join(prediction_path, "attention_weights.csv"))
    print(df_score)
    
if __name__ == "__main__":

    base_path = os.path.dirname(os.path.abspath(__file__))  # Huidige scriptmap
    print(f"base_path: {base_path}")
    data_path = os.path.join(base_path, "data", "CSI300")
    print(f"data_path: {data_path}")

    for i in [1,2,3]:

        data_train_predict_path = os.path.join(data_path, f"data_train_predict_random{i}") #gpu_wvt, oldway_0.6, gpu_wvt
        print(f"data_train_predict_path: {data_train_predict_path}")
        daily_stock_path = os.path.join(data_path, f"daily_stock_random{i}") #gpu_wvt, oldway, gpu_wvt
        print(f"daily_stock_path: {daily_stock_path}")
        save_path = os.path.join(data_path, f"model_saved_random_{i}_-120")
        os.makedirs(save_path, exist_ok=True)
        prediction_path = os.path.join(data_path, f"prediction_random_{i}_-120")
        os.makedirs(prediction_path, exist_ok=True)
        print(prediction_path)

        total_data_points = len(os.listdir(data_train_predict_path))
        print(f"Total data points: {total_data_points}")
        data_start = 0
        data_middle = total_data_points-20 - 120
        data_end = total_data_points -120
        pre_data = '2025-03-07'
        fun_train_predict(data_start, data_middle, data_end, pre_data)

            # data_train_predict_path = os.path.join(data_path, f"data_train_predict_csi300") #gpu_wvt, oldway_0.6, gpu_wvt
            # print(f"data_train_predict_path: {data_train_predict_path}")
            # daily_stock_path = os.path.join(data_path, f"daily_stock_csi300") #gpu_wvt, oldway, gpu_wvt
            # print(f"daily_stock_path: {daily_stock_path}")
            # save_path = os.path.join(data_path, f"model_saved_DSE_{i}")
            # os.makedirs(save_path, exist_ok=True)
            # prediction_path = os.path.join(data_path, f"prediction_DSE_{i}")
            # os.makedirs(prediction_path, exist_ok=True)
            # print(prediction_path)

            # total_data_points = len(os.listdir(data_train_predict_path))
            # print(f"Total data points: {total_data_points}")
            # data_start = 0
            # data_middle = total_data_points-20 -120
            # data_end = total_data_points -120
            # pre_data = '2025-03-07'
            # fun_train_predict(data_start, data_middle, data_end, pre_data)


    # for batchmap in os.listdir(data_path):
    #     print("batchmap: ", batchmap)

    #     # if (batchmap == 'batch_1') or (batchmap == 'batch_2') or (batchmap == 'batch_3'):
    #     #     print('al gebeurd')
    #     #     continue

    #     data_train_predict_path = os.path.join(data_path, batchmap, f"data_train_predict_corr") #gpu_wvt, oldway_0.6, gpu_wvt
    #     print(f"data_train_predict_path: {data_train_predict_path}")
    #     daily_stock_path = os.path.join(data_path, batchmap, f"daily_stock_corr") #gpu_wvt, oldway, gpu_wvt
    #     print(f"daily_stock_path: {daily_stock_path}")
    #     save_path = os.path.join(data_path, batchmap, f"model_saved_corr")
    #     os.makedirs(save_path, exist_ok=True)
    #     prediction_path = os.path.join(data_path, batchmap, f"prediction_corr")
    #     os.makedirs(prediction_path, exist_ok=True)
    #     print(prediction_path)

    #     total_data_points = len(os.listdir(data_train_predict_path))
    #     print(f"Total data points: {total_data_points}")
    #     data_start = 0
    #     data_middle = total_data_points-20
    #     data_end = total_data_points
    #     pre_data = '2025-03-07'
    #     fun_train_predict(data_start, data_middle, data_end, pre_data)

    #     data_train_predict_path = os.path.join(data_path, batchmap, f"data_train_predict_DSE") #gpu_wvt, oldway_0.6, gpu_wvt
    #     print(f"data_train_predict_path: {data_train_predict_path}")
    #     daily_stock_path = os.path.join(data_path, batchmap, f"daily_stock_DSE") #gpu_wvt, oldway, gpu_wvt
    #     print(f"daily_stock_path: {daily_stock_path}")
    #     save_path = os.path.join(data_path, batchmap, f"model_saved_DSE")
    #     os.makedirs(save_path, exist_ok=True)
    #     prediction_path = os.path.join(data_path, batchmap, f"prediction_DSE")
    #     os.makedirs(prediction_path, exist_ok=True)
    #     print(prediction_path)

    #     total_data_points = len(os.listdir(data_train_predict_path))
    #     print(f"Total data points: {total_data_points}")
    #     data_start = 0
    #     data_middle = total_data_points-20
    #     data_end = total_data_points
    #     pre_data = '2025-03-07'
    #     fun_train_predict(data_start, data_middle, data_end, pre_data)

    #     for j in [1, 2, 3]:

    #         data_train_predict_path = os.path.join(data_path, batchmap, f"data_train_predict_random{j}") #gpu_wvt, oldway_0.6, gpu_wvt
    #         print(f"data_train_predict_path: {data_train_predict_path}")
    #         daily_stock_path = os.path.join(data_path, batchmap, f"daily_stock_random{j}") #gpu_wvt, oldway, gpu_wvt
    #         print(f"daily_stock_path: {daily_stock_path}")
    #         save_path = os.path.join(data_path, batchmap, f"model_saved_random{j}")
    #         os.makedirs(save_path, exist_ok=True)
    #         prediction_path = os.path.join(data_path, batchmap, f"prediction_random{j}")
    #         os.makedirs(prediction_path, exist_ok=True)
    #         print(prediction_path)

    #         total_data_points = len(os.listdir(data_train_predict_path))
    #         print(f"Total data points: {total_data_points}")
    #         data_start = 0
    #         data_middle = total_data_points-20
    #         data_end = total_data_points
    #         pre_data = '2025-03-07'
    #         fun_train_predict(data_start, data_middle, data_end, pre_data)

            # data_train_predict_path = os.path.join(data_path, batchmap, "data_train_predict_SP") #gpu_wvt, oldway_0.6, gpu_wvt
            # print(f"data_train_predict_path: {data_train_predict_path}")
            # daily_stock_path = os.path.join(data_path, batchmap, "daily_stock_SP") #gpu_wvt, oldway, gpu_wvt
            # print(f"daily_stock_path: {daily_stock_path}")
            # save_path = os.path.join(data_path, batchmap, "model_saved_DSE")
            # os.makedirs(save_path, exist_ok=True)
            # prediction_path = os.path.join(data_path, batchmap, "prediction_DSE")
            # os.makedirs(prediction_path, exist_ok=True)
            # print(prediction_path)

            # total_data_points = len(os.listdir(data_train_predict_path))
            # print(f"Total data points: {total_data_points}")
            # data_start = 0
            # data_middle = total_data_points-20
            # data_end = total_data_points
            # pre_data = '2025-03-07'
            # fun_train_predict(data_start, data_middle, data_end, pre_data)



    # for map in os.listdir(data_train_predict_map):
    #     sthp, sthn = map.split("_")
    #     sthp = float(sthp)
    #     sthn = float(sthn)
    #     print(sthp, sthn)

    #     data_train_predict_path = os.path.join(data_train_predict_map, f"{sthp}_{sthn}") #gpu_wvt, oldway_0.6, gpu_wvt
    #     print(f"data_train_predict_path: {data_train_predict_path}")
    #     daily_stock_path = os.path.join(daily_stock_map, f"{sthp}_{sthn}") #gpu_wvt, oldway, gpu_wvt
    #     print(f"daily_stock_path: {daily_stock_path}")
    #     save_path = os.path.join(save_map, f"{sthp}_{sthn}")
    #     os.makedirs(save_path, exist_ok=True)
    #     prediction_path = os.path.join(prediction_map, f"{sthp}_{sthn}")
    #     os.makedirs(prediction_path, exist_ok=True)

    #     predict_file = os.path.join(prediction_path, "pred.csv")
    #     if os.path.exists(predict_file):
    #         print(f"Model {sthp}_{sthn} al getraind (bestand bestaat). Overslaan.")
    #         continue

    #     total_data_points = len(os.listdir(data_train_predict_path))
    #     print(f"Total data points: {total_data_points}")
    #     data_start = 0
    #     data_middle = total_data_points-20
    #     data_end = total_data_points
    #     pre_data = '2025-03-06'
    #     fun_train_predict(data_start, data_middle, data_end, pre_data)