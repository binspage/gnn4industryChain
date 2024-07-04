import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, ChebConv, GCNConv, SAGEConv


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class GCNNet(torch.nn.Module):
    def __init__(self, args):
        super(GCNNet, self).__init__()

        if args.ca:
            self.beta_q0 = torch.nn.Parameter(torch.zeros([1]))
            self.beta_q1 = torch.nn.Linear(9, args.model_dim, bias=False)
            self.beta_q2 = torch.nn.Linear(9, args.model_dim, bias=False)

            self.w_k = torch.nn.Linear(4, args.model_dim, bias=False)
            self.w_v = torch.nn.Linear(4, args.model_dim, bias=False)
            self.bn1 = nn.BatchNorm1d(4)

        self.args = args
        self.bn = nn.BatchNorm1d(4)

        self.conv1 = GCNConv(4, 64)
        self.conv2 = GCNConv(64, 128)
        self.conv3 = GCNConv(128, 64)
        self.conv4 = GCNConv(64, 32)

        self.conv_bn1 = nn.BatchNorm1d(64)
        self.conv_bn2 = nn.BatchNorm1d(128)
        self.conv_bn3 = nn.BatchNorm1d(64)
        self.conv_bn4 = nn.BatchNorm1d(32)

        self.out = nn.Linear(32, 1)

        self.apply(init_weights)

    def forward(self, data, attn=False):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x_ESG = x[:, 4:13]
        x_esg = x[:, 13:22]
        x_features = x[:, 0:4]

        if self.args.ca:
            x_q1 = self.beta_q1(x_esg)
            x_q2 = self.beta_q2(torch.mul(x_ESG, x_esg))
            x_q = self.beta_q0 + x_q1 + x_q2
            x_features = self.bn(x_features)

            x_k = self.w_k(x_features)
            x_v = self.w_v(x_features)

            attn_weight = torch.mm(x_q, x_k.T) / math.sqrt(x_q.size(-1))
            attn_weight = F.softmax(attn_weight, dim=1)
            x = attn_weight @ x_v
            x = self.bn1(x)
        else:
            x = self.bn(x_features)

        x = self.conv1(x, edge_index, edge_attr)
        x = torch.relu(self.conv_bn1(x))
        x = self.conv2(x, edge_index, edge_attr)
        x = torch.relu(self.conv_bn2(x))
        x = self.conv3(x, edge_index, edge_attr)
        x = torch.relu(self.conv_bn3(x))
        x = self.conv4(x, edge_index, edge_attr)
        x = torch.relu(self.conv_bn4(x))

        out = self.out(x)
        out = out.squeeze(-1)

        return out


class GATNet(torch.nn.Module):
    def __init__(self, args):
        super(GATNet, self).__init__()

        if args.ca:
            self.beta_q0 = torch.nn.Parameter(torch.zeros([1]))
            self.beta_q1 = torch.nn.Linear(9, args.model_dim, bias=False)
            self.beta_q2 = torch.nn.Linear(9, args.model_dim, bias=False)

            self.w_k = torch.nn.Linear(4, args.model_dim, bias=False)
            self.w_v = torch.nn.Linear(4, args.model_dim, bias=False)
            self.bn1 = nn.BatchNorm1d(4)

        self.args = args
        self.bn = nn.BatchNorm1d(4)

        self.GAT1 = GATConv(4, 64//args.num_heads, heads=args.num_heads, concat=True)
        self.GAT_bn1 = nn.BatchNorm1d(64)
        self.GAT2 = GATConv(64, 128//args.num_heads, heads=args.num_heads, concat=True)
        self.GAT_bn2 = nn.BatchNorm1d(128)
        self.GAT3 = GATConv(128, 64//args.num_heads, heads=args.num_heads, concat=True)
        self.GAT_bn3 = nn.BatchNorm1d(64)
        self.GAT4 = GATConv(64, 32//args.num_heads, heads=args.num_heads, concat=True)
        self.GAT_bn4 = nn.BatchNorm1d(32)
        self.out = nn.Linear(32, 1)

    def forward(self, data, attn=False):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x_ESG = x[:, 4:13]
        x_esg = x[:, 13:22]
        x_features = x[:, 0:4]

        if self.args.ca:
            x_q1 = self.beta_q1(x_esg)
            x_q2 = self.beta_q2(torch.mul(x_ESG, x_esg))
            x_q = self.beta_q0 + x_q1 + x_q2
            x_features = self.bn(x_features)

            x_k = self.w_k(x_features)
            x_v = self.w_v(x_features)

            attn_weight = torch.mm(x_q, x_k.T) / math.sqrt(x_q.size(-1))
            attn_weight = F.softmax(attn_weight, dim=1)
            x = attn_weight @ x_v
            x = self.bn1(x) + x_features
        else:
            x = self.bn(x_features)

        x = self.GAT1(x, edge_index)
        x = self.GAT_bn1(x)
        x = torch.relu(x)

        x = self.GAT2(x, edge_index)
        x = self.GAT_bn2(x)
        x = torch.relu(x)

        x = self.GAT3(x, edge_index)
        x = self.GAT_bn3(x)
        x = torch.relu(x)

        x = self.GAT4(x, edge_index)
        x = self.GAT_bn4(x)
        x = torch.relu(x)

        out = self.out(x)
        out = out.squeeze(-1)

        return out


class ChebNet(torch.nn.Module):
    def __init__(self, args):
        super(ChebNet, self).__init__()
        if args.ca:
            self.beta_q0 = torch.nn.Parameter(torch.zeros([1]))
            self.beta_q1 = torch.nn.Linear(9, args.model_dim, bias=False)
            self.beta_q2 = torch.nn.Linear(9, args.model_dim, bias=False)

            self.w_k = torch.nn.Linear(4, args.model_dim, bias=False)
            self.w_v = torch.nn.Linear(4, args.model_dim, bias=False)
            self.bn1 = nn.BatchNorm1d(4)
        self.args = args
        self.bn = nn.BatchNorm1d(4)

        self.cheb1 = ChebConv(4, 64, K=3)
        self.cheb2 = ChebConv(64, 128, K=3)
        self.cheb3 = ChebConv(128, 64, K=3)
        self.cheb4 = ChebConv(64, 32, K=3)

        self.conv_bn1 = nn.BatchNorm1d(64)
        self.conv_bn2 = nn.BatchNorm1d(128)
        self.conv_bn3 = nn.BatchNorm1d(64)
        self.conv_bn4 = nn.BatchNorm1d(32)

        self.out = nn.Linear(32, 1)
        self.apply(init_weights)

    def forward(self, data, attn=False):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x_ESG = x[:, 4:13]
        x_esg = x[:, 13:22]
        x_features = x[:, 0:4]

        if self.args.ca:
            x_q1 = self.beta_q1(x_esg)
            x_q2 = self.beta_q2(torch.mul(x_ESG, x_esg))
            x_q = self.beta_q0 + x_q1 + x_q2
            x_features = self.bn(x_features)

            x_k = self.w_k(x_features)
            x_v = self.w_v(x_features)

            attn_weight = torch.mm(x_q, x_k.T) / math.sqrt(x_q.size(-1))
            attn_weight = F.softmax(attn_weight, dim=1)

            x = attn_weight @ x_v
            x = self.bn1(x) + x_features
        else:
            x = self.bn(x_features)

        x = self.cheb1(x, edge_index, edge_attr)
        x = torch.relu(self.conv_bn1(x))

        x = self.cheb2(x, edge_index, edge_attr)
        x = torch.relu(self.conv_bn2(x))

        x = self.cheb3(x, edge_index, edge_attr)
        x = torch.relu(self.conv_bn3(x))

        x = self.cheb4(x, edge_index, edge_attr)
        x = torch.relu(self.conv_bn4(x))

        out = self.out(x)
        out = out.squeeze(-1)

        return out


class SageNet(torch.nn.Module):
    def __init__(self, args):
        super(SageNet, self).__init__()

        if args.ca:
            self.beta_q0 = torch.nn.Parameter(torch.zeros([1]))
            self.beta_q1 = torch.nn.Linear(9, args.model_dim, bias=False)
            self.beta_q2 = torch.nn.Linear(9, args.model_dim, bias=False)

            self.w_k = torch.nn.Linear(4, args.model_dim, bias=False)
            self.w_v = torch.nn.Linear(4, 4, bias=False)
            self.bn1 = nn.BatchNorm1d(4)
        self.args = args
        self.bn = nn.BatchNorm1d(4)

        self.sage1 = SAGEConv(4, 64)
        self.sage2 = SAGEConv(64, 128)
        self.sage3 = SAGEConv(128, 64)
        self.sage4 = SAGEConv(64, 32)

        self.conv_bn1 = nn.BatchNorm1d(64)
        self.conv_bn2 = nn.BatchNorm1d(128)
        self.conv_bn3 = nn.BatchNorm1d(64)
        self.conv_bn4 = nn.BatchNorm1d(32)

        self.out = nn.Linear(32, 1)


    def forward(self, data, attn=False):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x_ESG = x[:, 4:13]
        x_esg = x[:, 13:22]
        x_features = x[:, 0:4]

        if self.args.ca:
            x_q1 = self.beta_q1(x_esg)
            x_q2 = self.beta_q2(torch.mul(x_ESG, x_esg))
            x_q = self.beta_q0 + x_q1 + x_q2
            x_features = self.bn(x_features)

            x_k = self.w_k(x_features)
            x_v = self.w_v(x_features)

            attn_weight = torch.mm(x_q, x_k.T) / math.sqrt(x_q.size(-1))
            attn_weight = F.softmax(attn_weight, dim=1)

            x = attn_weight @ x_v

            x = self.bn1(x) + x_features
        else:
            x = self.bn(x_features)

        x = self.sage1(x, edge_index)
        x = torch.relu(self.conv_bn1(x))

        x = self.sage2(x, edge_index)
        x = torch.relu(self.conv_bn2(x))

        x = self.sage3(x, edge_index)
        x = torch.relu(self.conv_bn3(x))

        x = self.sage4(x, edge_index)
        x = torch.relu(self.conv_bn4(x))
        out = self.out(x)
        out = out.squeeze(-1)

        return out