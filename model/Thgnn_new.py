from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn as nn
import torch
import math
import torch.nn.functional as F

class GraphAttnMultiHead(Module):
    def __init__(self, in_features, out_features, negative_slope=0.2, num_heads=4, bias=True, residual=True):
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
        masked_weight = torch.mul(weight, adj_mat).to_sparse()
        attn_weights = torch.sparse.softmax(masked_weight, dim=2).to_dense()
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
        w = self.project(inputs)
        beta = torch.softmax(w, dim=1)
        if requires_weight:
            return (beta * inputs).sum(1), beta
        else:
            return (beta * inputs).sum(1), None
        

class TransformerEncoder(nn.Module):
    def __init__(self, d_model, d_ff, n_heads, n_layers, dropout):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, d_ff, n_heads, dropout)
            for _ in range(n_layers)
        ])
        
    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, n_heads, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self attention
        attn_out, _ = self.self_attn(x, x, x, attn_mask=mask)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)
        
        # Feed forward
        ff_out = self.linear2(self.dropout(F.relu(self.linear1(x))))
        x = x + self.dropout(ff_out)
        x = self.norm2(x)
        return x
    

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class StockHeteGAT(nn.Module):
    def __init__(self, in_features=4, out_features=8, num_heads=8, hidden_dim=64, num_layers=1):
        super(StockHeteGAT, self).__init__()
        # Basis input projectie
        self.input_proj = nn.Linear(in_features, hidden_dim)
        
        # Transformer encoder
        self.pos_encoder = PositionalEncoding(hidden_dim)
        self.transformer = TransformerEncoder(
            d_model=hidden_dim,
            d_ff=hidden_dim*4,
            n_heads=num_heads//2,
            n_layers=num_layers
        )
        self.pos_gat = GraphAttnMultiHead(
            in_features=hidden_dim,
            out_features=out_features,
            num_heads=num_heads
        )
        self.neg_gat = GraphAttnMultiHead(
            in_features=hidden_dim,
            out_features=out_features,
            num_heads=num_heads
        )
        self.mlp_self = nn.Linear(hidden_dim, hidden_dim)
        self.mlp_pos = nn.Linear(out_features*num_heads, hidden_dim)
        self.mlp_neg = nn.Linear(out_features*num_heads, hidden_dim)
        self.pn = PairNorm(mode='PN-SI')
        self.sem_gat = GraphAttnSemIndividual(in_features=hidden_dim,
                                              hidden_size=hidden_dim,
                                              act=nn.Tanh())
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Tanh()#,
            #nn.Sigmoid()
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.02)

    def forward(self, inputs, pos_adj, neg_adj, requires_weight=False):
        print("\n=== Features Debug ===")
        print("Shape:", inputs.shape)  # Verwacht [num_nodes, num_features]
        print("Stats - Mean: {:.4f}, Std: {:.4f}, Min: {:.4f}, Max: {:.4f}".format(
            inputs.mean(), inputs.std(), inputs.min(), inputs.max()))
        print("Voorbeeld (eerste node):", inputs[0, :5])  # Eerste 5 features van node 0
        # Input shape transformeren
        inputs = self.input_norm(inputs.float())
        print("Normed Shape:", inputs.shape)  # Verwacht [num_nodes, window, num_features]
        batch_size, seq_len, num_features = inputs.shape
        
        # Eerst door input layer
        x = inputs.reshape(-1, num_features)  # [200*20, 4]
        x = self.feature_proj(x).reshape(batch_size, seq_len, -1)  # [200, 20, hidden_dim]
        
        # Transformer verwacht [seq_len, batch_size, features]
        x = x.transpose(0, 1)  # [20, 200, hidden_dim]
        x = self.pos_encoder(x)
        x = self.transformer(x)        
        support = x[-1]
        pos_support, pos_attn_weights = self.pos_gat(support, pos_adj, requires_weight)
        neg_support, neg_attn_weights = self.neg_gat(support, neg_adj, requires_weight)
        support = self.mlp_self(support)
        pos_support = self.mlp_pos(pos_support)
        neg_support = self.mlp_neg(neg_support)
        all_embedding = torch.stack((support, pos_support, neg_support), dim=1)
        all_embedding, sem_attn_weights = self.sem_gat(all_embedding, requires_weight)
        all_embedding = self.pn(all_embedding)
        if requires_weight:
            return self.predictor(all_embedding), (pos_attn_weights, neg_attn_weights, sem_attn_weights)
        else:
            output = self.predictor(all_embedding)
            print(f"Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
            return output
