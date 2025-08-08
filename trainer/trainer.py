import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np


def mse_loss(logits, targets):
    mse = nn.MSELoss()
    loss = mse(logits.squeeze(), targets)
    # print(f"mse loss: {loss}")
    return loss

def mae_loss(logits, targets):
    mae = nn.L1Loss()
    return mae(logits.squeeze(), targets)

def bce_loss(logits, targets):
    bce = nn.BCELoss()
    loss = bce(logits.squeeze(), targets)
    return loss


def evaluate(model, features, adj_pos, adj_neg, labels, mask, loss_func):
    model.eval()
    with torch.no_grad():
        logits,*_ = model(features, adj_pos, adj_neg)

    loss = loss_func(logits,labels)
    return loss, logits


def extract_data(data_dict, device):
    pos_adj = data_dict['pos_adj'].to(device).squeeze()
    neg_adj = data_dict['neg_adj'].to(device).squeeze()
    features = data_dict['features'].to(device).squeeze()
    mask1 = torch.ones(features.shape[2], dtype=torch.bool)
    mask1[5] = False
    mask1[4] = False
    features = features[:, :, mask1]
    labels = data_dict['labels'].to(device).squeeze()

    # print(f'features mean: {features.mean(dim=(0,1))}, features std: {features.std(dim=(0,1))}')
    # print(f"features std: {features.std().item()}, labels std: {labels.std().item()}")
    # print(f"features min: {features.min().item()}, labels min: {labels.min().item()}")
    # print(f"features max: {features.max().item()}, labels max: {labels.max().item()}")

    # voor log data
    # labels = torch.log(labels+1)/0.025
    # labels = torch.tanh(torch.log(labels +1)/0.025)


    # voor niet log data
    features = features/2
    features = torch.clip(features, -5, 5)
    # print(f"features shape: {features.shape}")
    # feature_norm = nn.LayerNorm(features.size()[1:]).to(device)
    # features = feature_norm(features)
    # labels = torch.log(labels+1)/0.025
    labels = torch.tanh(torch.log(labels +1)/0.025)
    # print(f'features mean: {features.mean(dim=(0,1))}, features std: {features.std(dim=(0,1))}')
    # print(f"features std: {features.std().item()}, labels std: {labels.std().item()}")
    # print(f"features min: {features.min().item()}, labels min: {labels.min().item()}")
    # print(f"features max: {features.max().item()}, labels max: {labels.max().item()}")
    # labels = (labels2 - labels2.mean()) / (labels2.std() + 1e-6)  # Normalize labels
    # labels = (data_dict['labels'].to(device).squeeze() > 0).float()
    # labels = features.mean(dim=(1, 2))
    # labels = torch.tanh(0.5*features.mean(dim=(1, 2)))
    # print(f"labels shape: {labels.shape}, features shape: {features.shape}")
    # print(f"features: min={features.min().item():.4f}, max={features.max().item():.4f}, mean={features.mean().item():.4f}, std={features.std().item():.4f}")
    # print(f"labels: min={labels.min().item():.4f}, max={labels.max().item():.4f}, mean={labels.mean().item():.4f}, std={labels.std().item():.4f}")

    mask = data_dict['mask']
    # print(mask)
    # print(pos_adj, neg_adj)
    # Check distribution of features and labels
    # print("Features mean:", features.mean().item(), "std:", features.std().item())
    # print("Labels mean:", labels.mean().item(), "std:", labels.std().item())
    # If you want to see histograms, you could use matplotlib (optional)
    # plt.hist(features.detach().cpu().numpy().flatten(), bins=50)
    # plt.title("Features distribution")
    # plt.show()
    # plt.hist(labels1.detach().cpu().numpy().flatten(), bins=50)
    # plt.title("Labels1 distribution")
    # plt.show()
    # plt.hist(labels2.detach().cpu().numpy().flatten(), bins=50)
    # plt.title("Labels2 distribution")
    # plt.show()
    # plt.hist(labels.detach().cpu().numpy().flatten(), bins=50)
    # plt.title("Labels distribution")
    # plt.show()

    return pos_adj, neg_adj, features, labels, mask


def train_epoch(epoch, args, model, dataset_train, optimizer, scheduler, loss_fcn):
    model.train()
    loss_return = 0
    dag = 1
    # loss_list = []
    # loss_list2 = []
    aantal_keer_berekend = 0
    for batch_data in tqdm(dataset_train):
        for batch_idx, data in enumerate(batch_data):
            model.zero_grad()
            pos_adj, neg_adj, features, labels, mask = extract_data(data, args.device)
            logits,*_ = model(features, pos_adj, neg_adj)
            print("corr pred-label:", np.corrcoef(logits.detach().cpu().numpy().flatten(), labels.detach().cpu().numpy().flatten())[0,1])
            loss = loss_fcn(logits[mask], labels[mask])
            # print(f"loss: {loss}")
            loss.backward()
            # for name, param in model.named_parameters():
            #     if param.grad is not None:
            #         print(f"{name}: grad mean={param.grad.mean().item():.4e}, std={param.grad.std().item():.4e}, min={param.grad.min().item():.4e}, max={param.grad.max().item():.4e}")
            #     else:
            #         print(f"{name}: grad is None")
            optimizer.step()
            scheduler.step()
            if batch_idx == 0:
                # if loss.detach().cpu().item() > 0.1:
                #     print("loss: ", loss.detach().cpu().item(), "   dag: ", dag)
                # loss_list2.append(loss.detach().cpu().item())
                # loss_list.append((dag, loss.detach().cpu().item()))
                aantal_keer_berekend += 1
                loss_return += loss.detach().cpu().item()
                dag += 1
                # print(f" loss data: {loss.data}")
    # print(f"Epoch {epoch}\nloss_return: {loss_return}\nlen loss: {aantal_keer_berekend}, {len(dataset_train)}")#\nloss list min en max: {min(loss_list2)};{max(loss_list2)}\nloss list: {len(loss_list)}, {loss_list}")
    return loss_return/len(dataset_train)


def eval_epoch(args, model, dataset_eval, loss_fcn):
    loss = 0.
    logits = None
    for batch_idx, data in enumerate(dataset_eval):
        pos_adj, neg_adj, features, labels, mask = extract_data(data, args.device)
        loss, logits = evaluate(model, features, pos_adj, neg_adj, labels, mask, loss_func=loss_fcn)
        break
    return loss, logits