import numpy as np
import random
import torch
import pickle
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit, RandomNodeSplit
from ModelZoo import GCNNet, GATNet, ChebNet, SageNet
import argparse
import warnings

warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser(description='ESG')
parser.add_argument('--gpu_id', type=str, default='1', help="device id to run")
parser.add_argument('--optimizer', type=str, default='adam', help="optimizer")
parser.add_argument('--model_dim', type=int, default=4, help="layers of GCN model")
parser.add_argument('--model', type=str, default='sage', choices=['ours', 'gcn', 'gat', 'sage', 'cheb'])
parser.add_argument('--task', type=str, default='OM', choices=['OM', 'RPCE'])
parser.add_argument('--p', type=float, default=0.1, help="dropout ratio")
parser.add_argument('--lr', type=float, default=1e-4, help="learning rate")
parser.add_argument('--iteration', type=int, default=10000, help="iteration")
parser.add_argument('--num_heads', type=int, default=4, help="num_heads for GAT")
parser.add_argument('--aggr', type=str, default='add', help="aggregation way")
parser.add_argument('--num_layers', type=int, default=4, help="number of layers")
parser.add_argument('--seed', type=int, default=1, help="random seed")
parser.add_argument('--ca', action='store_true', help="Cross-Attention")
args = parser.parse_args()

SEED = args.seed
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

log_file = open('./data/{}/log_{}{}_{}.txt'.format(args.task, args.model, '+CrossAttention' if args.ca else '', args.seed), 'w')


if args.task == 'OM':
    pkl_file = open('./data/data_OM.pkl', 'rb')
    data_task = pickle.load(pkl_file)
    data_task = Data(x=data_task[0], y=data_task[1], edge_index=data_task[2], edge_attr=data_task[3])

if args.task == 'RPCE':
    pkl_file = open('./data/data_RPCE.pkl', 'rb')
    data_task = pickle.load(pkl_file)
    data_task = Data(x=data_task[0], y=data_task[1], edge_index=data_task[2], edge_attr=data_task[3])

if args.model == 'gcn':
    print('gcn')
    model = GCNNet(args)
if args.model == 'gat':
    print('gat')
    model = GATNet(args)
if args.model == 'cheb':
    print('cheb')
    model = ChebNet(args)
if args.model == 'sage':
    print('sage')
    model = SageNet(args)


device = torch.device('cuda:{}'.format(args.gpu_id) if torch.cuda.is_available() else 'cpu')

transfrom = RandomLinkSplit(num_val=0.1, num_test=0.2, is_undirected=False,
                            add_negative_train_samples=False)
train_data, val_data, test_data = transfrom(data_task)
model.to(device)
train_data = train_data.to(device)
test_data = test_data.to(device)
val_data = val_data.to(device)


def Evaluation(target, pre):
    mae = np.mean(np.abs(target - pre))
    mape = np.mean(np.abs(target - pre) / (target + 2))
    rmse = np.sqrt(np.mean(np.power(target - pre, 2)))

    return mae, mape, rmse


if args.optimizer == 'adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
if args.optimizer == 'sgd':
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True, weight_decay=1e-3)
loss_func = torch.nn.MSELoss()


def train():
    min_loss = 1e10
    train_losses = []
    val_losses = []
    model.train()
    for epoch in range(args.iteration+1):
        optimizer.zero_grad()
        prediction = model(train_data)

        loss = torch.sqrt(loss_func(prediction, train_data.y).float())
        train_losses.append(loss)
        loss.backward()
        optimizer.step()

        model.eval()
        out = model(val_data)
        loss_val = torch.sqrt(loss_func(out, val_data.y).float())
        pre = out.reshape(-1).detach().cpu().numpy()
        target = val_data.y.reshape(-1).detach().cpu().numpy()
        mae, mape, rmse = Evaluation(target, pre)
        val_losses.append(rmse)
        model.train()

        if epoch % 1000 == 0:
            print('Epoch {:03d} loss: {:.4f} '.format(
                epoch, loss.item()))
            log_file.write('Epoch {:03d} loss: {:.4f} '.format(
                epoch, loss.item()) + '\n')
            log_file.flush()

            if min_loss > loss_val.item():
                min_loss = loss_val.item()
                torch.save(model.state_dict(), './data/{}/model_{}{}_{}.pth'.format(args.task, args.model, '+CrossAttention' if args.ca else '', args.seed))

    output = open('./data/losses/model_{}{}_{}.pth'.format(args.model, '+CrossAttention' if args.ca else '', args.seed), 'wb')
    tt = {'train_losses': train_losses, 'val_losses': val_losses}
    pickle.dump(tt, output, -1)
    output.close()


def test(path):
    if args.model == 'ours':
        print('ours')
        test_model = SpaceNet(args)
    if args.model == 'gcn':
        print('gcn')
        test_model = GCNNet(args)
    if args.model == 'gat':
        print('gat')
        test_model = GATNet(args)
    if args.model == 'cheb':
        print('cheb')
        test_model = ChebNet(args)
    if args.model == 'sage':
        print('sage')
        test_model = SageNet(args)
    test_model.load_state_dict(torch.load(path))
    test_model.to(device)
    test_model.eval()
    out = test_model(test_data)
    loss = torch.sqrt(loss_func(out, test_data.y).float())

    pre = out.reshape(-1).detach().cpu().numpy()
    target = test_data.y.reshape(-1).detach().cpu().numpy()

    mae, mape, rmse = Evaluation(target, pre)

    print("test_loss:{:.4f}  RMSE:{:.4f}  MAE:{:.4f}  MAPE:{:.4f}".format(loss.item(), rmse, mae, mape))

    log_file.write("test_loss:{:.4f}  RMSE:{:.4f}  MAE:{:.4f}  MAPE:{:.4f}".format(loss.item(), rmse, mae, mape))
    log_file.flush()


if __name__ == '__main__':
    train()
    test('./data/{}/model_{}{}_{}.pth'.format(args.task, args.model, '+CrossAttention' if args.ca else '', args.seed))
