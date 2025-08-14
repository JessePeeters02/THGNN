from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn as nn
import torch
import math

class GraphAttnMultiHead(Module):
    def __init__(self, in_features, out_features, negative_slope=0.2, num_heads=4, bias=True, residual=False):
        super(GraphAttnMultiHead, self).__init__()
        self.num_heads = num_heads
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, num_heads * out_features))
        self.weight_u = Parameter(torch.FloatTensor(num_heads, out_features, 1))
        self.weight_v = Parameter(torch.FloatTensor(num_heads, out_features, 1))
        self.leaky_relu = nn.LeakyReLU(negative_slope=negative_slope)
        self.residual = residual
        if self.residual:
            self.project = nn.Linear(in_features, num_heads*out_features)
        else:
            self.project = None
        if bias:
            self.bias = Parameter(torch.FloatTensor(1, num_heads * out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(-1))
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
        self.weight.data.uniform_(-stdv, stdv)
        # self.weight.data.uniform_(-1, 1)
        stdv = 1. / math.sqrt(self.weight_u.size(-1))
        self.weight_u.data.uniform_(-stdv, stdv)
        self.weight_v.data.uniform_(-stdv, stdv)

    def forward(self, inputs, adj_mat, requires_weight=False):
        support = torch.mm(inputs, self.weight)
        support = support.reshape(-1, self.num_heads, self.out_features).permute(1, 0, 2)
        f_1 = torch.matmul(support, self.weight_u).reshape(self.num_heads, 1, -1)
        f_2 = torch.matmul(support, self.weight_v).reshape(self.num_heads, -1, 1)
        logits = f_1 + f_2
        weight = self.leaky_relu(logits)

        # Kies snelste pad o.b.v. dichtheid
        with torch.no_grad():
            # adj_mat kan [N,N] of [H,N,N] zijn; we nemen mean over laatste 2 dims
            if adj_mat.dim() == 2:
                density = (adj_mat != 0).float().mean()
            else:
                density = (adj_mat != 0).float().mean(dim=(-2, -1)).mean()
        use_sparse = adj_mat.is_sparse and float(density.item()) < 0.15

        if use_sparse:
            masked_weight = torch.mul(weight, adj_mat).to_sparse()
            attn_weights = torch.sparse.softmax(masked_weight, dim=2).to_dense()
        else:
            # Dense: mask wegvullen met zeer negatieve waarde en softmax over dim=2
            mask = (adj_mat > 0)
            very_neg = torch.finfo(weight.dtype).min
            logits_masked = weight.masked_fill(~mask, very_neg)
            attn_weights = torch.softmax(logits_masked, dim=2)

        support = torch.matmul(attn_weights, support)
        support = support.permute(1, 0, 2).reshape(-1, self.num_heads * self.out_features)
        if self.bias is not None:
            support = support + self.bias
        if self.residual:
            support = support + self.project(inputs)
        if requires_weight:
            return support, attn_weights
        else:
            return support, None


class PairNorm(nn.Module):
    def __init__(self, mode='PN', scale=1):
        assert mode in ['None', 'PN', 'PN-SI', 'PN-SCS']
        super(PairNorm, self).__init__()
        self.mode = mode
        self.scale = scale

    def forward(self, x):
        if self.mode == 'None':
            return x
        col_mean = x.mean(dim=0)
        if self.mode == 'PN':
            x = x - col_mean
            rownorm_mean = (1e-6 + x.pow(2).sum(dim=1).mean()).sqrt()
            x = self.scale * x / rownorm_mean
        if self.mode == 'PN-SI':
            x = x - col_mean
            rownorm_individual = (1e-6 + x.pow(2).sum(dim=1, keepdim=True)).sqrt()
            x = self.scale * x / rownorm_individual
        if self.mode == 'PN-SCS':
            rownorm_individual = (1e-6 + x.pow(2).sum(dim=1, keepdim=True)).sqrt()
            x = self.scale * x / rownorm_individual - col_mean
        return x


class GraphAttnSemIndividual(Module):
    def __init__(self, in_features, hidden_size=128, act=nn.Tanh()):
        super(GraphAttnSemIndividual, self).__init__()
        self.project = nn.Sequential(nn.Linear(in_features, hidden_size),
                                     act,
                                     nn.Linear(hidden_size, 1, bias=False))

    def forward(self, inputs, requires_weight=False):
        # inputs: [N, 3, D]
        N = inputs.size(0)
        device = inputs.device
        dtype = inputs.dtype
        # Forceer betas naar 1/3
        beta = torch.full((N, 3, 1), 1/3, device=device, dtype=dtype)
        combined = (beta * inputs).sum(1)  # [N, D]
        if requires_weight:
            return combined, beta.squeeze(-1)  # squeeze voor [N,3]
        else:
            return combined, None


class StockHeteGAT(nn.Module):
    def __init__(self, in_features=6, out_features=8, num_heads=8, hidden_dim=64, num_layers=1):
        super(StockHeteGAT, self).__init__()
        self.encoding = nn.GRU(
            input_size=in_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=0.1
        )
        self.pos_gat = GraphAttnMultiHead(
            in_features=hidden_dim,
            out_features=out_features,
            num_heads=num_heads,
            residual=False
        )
        self.neg_gat = GraphAttnMultiHead(
            in_features=hidden_dim,
            out_features=out_features,
            num_heads=num_heads,
            residual=False
        )
        self.mlp_self = nn.Linear(hidden_dim, hidden_dim)
        self.mlp_pos = nn.Linear(out_features*num_heads, hidden_dim)
        self.mlp_neg = nn.Linear(out_features*num_heads, hidden_dim)
        self.pn = PairNorm(mode='PN-SI')
        # self.pn = PairNorm(mode='PN')
        # self.pn = PairNorm(mode='None')
        self.sem_gat = GraphAttnSemIndividual(in_features=hidden_dim,
                                              hidden_size=hidden_dim,
                                              act=nn.Tanh())
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Tanh()  
            # nn.Sigmoid()
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.02)

    def forward(self, inputs, pos_adj, neg_adj, requires_weight=True):
        _, support = self.encoding(inputs)
        support = support.squeeze()
        # print("GRU support mean:", support.mean().item(), "std:", support.std().item(), "min:", support.min().item(), "max:", support.max().item())
        pos_support, pos_attn_weights = self.pos_gat(support, pos_adj, requires_weight)
        # print("Pos GAT support mean:", pos_support.mean().item(), "std:", pos_support.std().item())
        neg_support, neg_attn_weights = self.neg_gat(support, neg_adj, requires_weight)
        # print("Neg GAT support mean:", neg_support.mean().item(), "std:", neg_support.std().item())
        support = self.mlp_self(support)
        pos_support = self.mlp_pos(pos_support)
        neg_support = self.mlp_neg(neg_support)
        all_embedding = torch.stack((support, pos_support, neg_support), dim=1)
        all_embedding, sem_attn_weights = self.sem_gat(all_embedding, requires_weight)
        # all_embedding = support  # of pos_support, of neg_support
        # sem_attn_weights = None
        # print("Sem beta std across dimensions:", sem_attn_weights.std(dim=1).mean().item())
        all_embedding = self.pn(all_embedding)
        # print("Support mean:", all_embedding.mean().item(), "std:", all_embedding.std().item())
        preds = self.predictor(all_embedding)
        # Print distributie van de voorspellingen
        # print(f"Predictions: min={preds.min().item():.4f}, max={preds.max().item():.4f}, mean={preds.mean().item():.4f}, std={preds.std().item():.4f}")
        if requires_weight:
            return (
                preds,
                {
                    "pos_attn_weights": pos_attn_weights,
                    "neg_attn_weights": neg_attn_weights,
                    "sem_attn_weights": sem_attn_weights,
                },
            )
        else:
            return preds