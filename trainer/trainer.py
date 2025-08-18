import torch
import torch.nn as nn
from tqdm import tqdm


def mse_loss(logits, targets):
    mse = nn.MSELoss()
    loss = mse(logits.squeeze(), targets)
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
    to_dev = lambda t: t.to(device, non_blocking=True)
    pos_adj = to_dev(data_dict['pos_adj']).squeeze()
    neg_adj = to_dev(data_dict['neg_adj']).squeeze()
    features = to_dev(data_dict['features']).squeeze()
    labels = to_dev(data_dict['labels']).squeeze()
    labels = torch.tanh(torch.log(labels +1))
    mask = data_dict['mask']

    return pos_adj, neg_adj, features, labels, mask


def train_epoch(epoch, args, model, dataset_train, optimizer, scheduler, loss_fcn):
    model.train()
    loss_return = 0
    dag = 1
    aantal_keer_berekend = 0
    for batch_data in tqdm(dataset_train):
        for batch_idx, data in enumerate(batch_data):
            model.zero_grad()
            pos_adj, neg_adj, features, labels, mask = extract_data(data, args.device)
            logits,*_ = model(features, pos_adj, neg_adj)
            # print("corr pred-label:", np.corrcoef(logits.detach().cpu().numpy().flatten(), labels.detach().cpu().numpy().flatten())[0,1])
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