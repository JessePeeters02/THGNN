import torch
import torch.nn as nn
from tqdm import tqdm

def mse_loss(logits, targets):
    mse = nn.MSELoss()
    loss = mse(logits.squeeze(), targets)
    return loss


def bce_loss(logits, targets):
    bce = nn.BCELoss()
    loss = bce(logits.squeeze(), targets)
    return loss


def evaluate(model, features, adj_pos, adj_neg, labels_norm, mask, loss_func=nn.L1Loss()):
    model.eval()
    with torch.no_grad():
        logits = model(features, adj_pos, adj_neg)
    loss = loss_func(logits[mask], labels_norm[mask])
    return loss, logits


def extract_data(data_dict, device, label_mean=None, label_std=None):
    pos_adj = data_dict['pos_adj'].to(device).squeeze()
    neg_adj = data_dict['neg_adj'].to(device).squeeze()
    features = data_dict['features'].to(device).squeeze()
    labels = data_dict['labels'].to(device).squeeze()
    mask = data_dict['mask']
    
    if label_mean is not None and label_std is not None:  # Normaliseer alleen als mean/std gegeven zijn
        labels = (labels - label_mean) / label_std

    return pos_adj, neg_adj, features, labels, mask


def train_epoch(epoch, args, model, dataset_train, optimizer, scheduler, loss_fcn, label_mean, label_std):
    model.train()
    loss_return = 0
    for batch_data in tqdm(dataset_train):
        for batch_idx, data in enumerate(batch_data):
            optimizer.zero_grad()
            pos_adj, neg_adj, features, labels, mask = extract_data(data, args.device, label_mean, label_std)
            logits = model(features, pos_adj, neg_adj)
            loss = loss_fcn(logits[mask], labels[mask])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            # print(f"Gradient norm: {total_norm:.4f}")
            optimizer.step()
            if batch_idx == 0:
                loss_return += loss.data
    return loss_return/len(dataset_train)


def eval_epoch(args, model, dataset_eval, loss_fcn, label_mean=None, label_std=None):
    loss = 0.
    logits = None
    for batch_idx, data in enumerate(dataset_eval):
        pos_adj, neg_adj, features, labels, mask = extract_data(data, args.device, label_mean, label_std)
        loss, logits = evaluate(model, features, pos_adj, neg_adj, labels, mask, loss_func=loss_fcn)
        break
    return loss, logits