import torch
from model import NET
from utils import LinearDecayLR
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from torch_geometric.loader import DataLoader
import argparse
from tqdm import tqdm
import os
import random

def eval_(model, device, loader, evaluator):
    model.eval()
    y_true = []
    y_pred = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                pred = model(batch)

            y_true.append(batch.y.view(-1,1).detach().cpu())
            y_pred.append(torch.argmax(pred.detach(), dim = 1).view(-1,1).cpu())

    y_true = torch.cat(y_true, dim = 0).numpy()
    y_pred = torch.cat(y_pred, dim = 0).numpy()

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return evaluator.eval(input_dict)
    
def train(net, train_iter, valid_iter, learning_rate, end_lr, num_epochs,\
              device,evaluator,warmup_updates=50,tot_updates=500000,):
    print('training on', device)
    net.to(device)
    optimizer = torch.optim.AdamW(
        net.parameters(),
        lr = learning_rate,
        weight_decay=1e-5)
    scheduler = LinearDecayLR(
                optimizer,
                warmup_updates=warmup_updates,
                tot_updates=tot_updates,
                lr=learning_rate,
                end_lr=end_lr,
    )
    loss = torch.nn.CrossEntropyLoss()
    
    num_batches = len(train_iter)
    for epoch in range(num_epochs):
        net.train()
        train_loss = 0
        print("training...")
        for i, X in enumerate(tqdm(train_iter, desc="Iteration")):
            y = X.y.view(-1)
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            scheduler.step()
            with torch.no_grad():
            	train_loss += l
        print("evaluating...")
        valid_acc = eval_(net, device, valid_iter, evaluator)['acc']
        print(f"epoch:{epoch+1}, train_loss:{train_loss/num_batches}, valid_acc:{valid_acc}")
    return valid_acc
    
def set_attr(data, max_nodes = 300):
    data.x = torch.zeros(max_nodes, dtype=torch.long)
    data.x[:data.num_nodes] = 1 # for padding
    data.n = data.num_nodes
    data.num_nodes = max_nodes # for data alignment
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='peak learning rate')
    parser.add_argument('--end_lr', type=float, default=5e-9,
                        help='end learning rate')
    parser.add_argument('--warmup_updates', type=int, default=5000,
                        help='learning rate warmup step(default: 5000)')
    parser.add_argument('--tot_updates', type=int, default=350000,
                        help='total steps of learning rate update(default: 350000)')
    parser.add_argument('--n_layers', type=int, default=3,
                        help='number of net layers (default: 3)')
    parser.add_argument('--feature_dim', type=int, default=384,
                        help='dimensionality of hidden units in GNNs (default: 384)')
    parser.add_argument('--head_num', type=int, default=16,
                        help='head number of hidden units in GNNs (default: 16)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--infla', nargs='+', type=float, default=[6.0],
                        help='inflation (default: [6.0])')
    parser.add_argument('--num_epochs', type=int, default=150,
                        help='number of epochs to train (default: 150)')
    parser.add_argument('--atte_drop_rate', type=float, default=0.5,
                        help='attention dropout rate (default: 0.5)')
    parser.add_argument('--input_drop_rate', type=float, default=0.2,
                        help='input dropout rate (default: 0.2)')
    parser.add_argument('--layer_drop', type=float, default=0.5,
                        help='layer dropout rate (default: 0.5)')
    parser.add_argument('--qkv_drop', type=float, default=0.5,
                        help='qkv dropout rate (default: 0.5)')
    parser.add_argument('--seed', type=int, default=None,
                        help='random seed (default: None)')
    parser.add_argument('--dataset', type=str, default="ogbg-ppa",
                        help='dataset name (default: ogbg-ppa)')
    parser.add_argument('--output', type=str, default="",
                        help='output file name (default: )')
    args = parser.parse_args()
    
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
            
            
    d_name=args.dataset
    dataset = PygGraphPropPredDataset(name = d_name, transform = set_attr) 

    split_idx = dataset.get_idx_split() 
    train_loader = DataLoader(dataset[split_idx["train"]], batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(dataset[split_idx["test"]], batch_size=args.batch_size, shuffle=False)
    
    device = torch.device(f"cuda:{str(args.device)}") if torch.cuda.is_available() else torch.device("cpu")
    evaluator = Evaluator("ogbg-ppa")
    print(args)
    net = NET(
        head_num=args.head_num,input_drop_rate=args.input_drop_rate,
        atte_drop_rate=args.atte_drop_rate,layer_drop=args.layer_drop,
        qkv_drop=args.qkv_drop,n_layers=args.n_layers,
        max_num_nodes=300,edge_dim=7,feature_dim=args.feature_dim,
        class_num=37,infla=args.infla,device=device
    )
    valid_acc = train(net, train_loader, valid_loader, args.lr, args.end_lr, args.num_epochs, device,evaluator,
          warmup_updates=args.warmup_updates,tot_updates=args.tot_updates)
    
    test_acc = eval_(net, device, test_loader, evaluator)['acc']
    train_acc = eval_(net, device, train_loader, evaluator)['acc']
    print(f"train_acc:{train_acc}, test_acc:{test_acc}")
    if args.output != '':
        final_result = f"train acc:{train_acc}, validation acc:{valid_acc}, test acc:{test_acc}"
        with open(args.output,'w') as f:
            f.write(f"parameters:{str(args.__dict__)}\n")
            f.write(final_result)
    
if __name__ == "__main__":
    main()
