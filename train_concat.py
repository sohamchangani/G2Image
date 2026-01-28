import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import KFold
from itertools import product
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
import statistics
import argparse
import sys
from models import *
from modules import *
from data_loader import *
from typing import Dict

# Argument parsing
#sys.argv = [sys.argv[0]]
parser = argparse.ArgumentParser()
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--folds', type=int, default=10)
parser.add_argument('--dataset', type=str, default='mutag')
parser.add_argument('--c_channels', type=int, default=64)
parser.add_argument('--d_model', type=int, default=32)
parser.add_argument('--lr', type=float, default=1e-4)
args = parser.parse_args()
device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')

print(f'\n=== Running on dataset: {args.dataset} ===')
dataset = load_data(args.dataset)
score_matrix=compute_node_features_hks_deg_kcore_pagerank(dataset)
sorted_scores, sort_idx_list, binned_list=sort_dataset_score_matrices(score_matrix,n_bins=10,col_order=(0, 1, 2, 3))
graphImage=[]
for graph_id in range(len(dataset)):
    image,adj,node_shorted=adjacency_from_sorted_order_nx(dataset[graph_id],sort_idx_list[graph_id],10)
    graphImage.append(image)

G_image = torch.tensor(np.array(graphImage),dtype=torch.float).unsqueeze(1)
G_image_expanded = G_image.expand(-1, 4, -1, -1)

#print(graphImage[0])
list_hks, thres_hks, label = get_thresh_hks(dataset, 9, 0.1)
list_deg, thres_deg = get_thresh(dataset, 9)
graph_features = []
#for graph_id in tqdm(range(len(dataset))):
for graph_id in range(len(dataset)):
    b0, b1, node, edge = Topo_Fe_TimeSeries_MP(dataset[graph_id], list_deg[graph_id], list_hks[graph_id],
                                               thres_deg, thres_hks)
    graph_features.append(torch.stack([b0, b1, node, edge], dim=0))
Toposurface = torch.stack(graph_features)

X = torch.cat((G_image_expanded, Toposurface), dim=2)

y = torch.tensor(label, dtype=torch.long)

@torch.no_grad()
def evaluate(model, loader, device) -> Dict[str, float]:
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0

    for x1, x2, y in loader:
        x1 = x1.to(device, non_blocking=True)
        x2 = x2.to(device, non_blocking=True)
        y  = y.to(device, non_blocking=True)

        logits, _, _, _, _ = model(x1, x2)
        loss = F.cross_entropy(logits, y)

        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
        loss_sum += loss.item() * y.size(0)

    return {
        "loss": loss_sum / max(total, 1),
        "acc": correct / max(total, 1),
    }

def run_experiment(lr, c_channels, d_model, drop_out, nhead, num_layers, batch_size):
    kf = KFold(n_splits=args.folds, shuffle=True, random_state=42)
    acc_per_fold = []
    num_classes = len(torch.unique(y))

    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        #print(train_idx)

        train_ds = TensorDataset(X_train, y_train)
        val_ds = TensorDataset(X_val, y_val)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size)

        model = CNNTransformer_image(
            num_classes=num_classes,
            in_channels=4,
            cnn_channels=c_channels,
            d_model=d_model,
            drop_out=drop_out,
            nhead=nhead,
            num_layers=num_layers
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        train_losses = []
        train_accuracies = []
        test_accuracies = []
        # Training

        #for epoch in tqdm(range(1, args.epochs + 1), desc=f"Epochs (Fold {fold})"):
        for epoch in tqdm(range(1, args.epochs + 1)):
            # Train
            model.train()
            correct_train = 0
            total_train = 0
            epoch_train_loss = 0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                output = model(xb)
                loss = criterion(output, yb)
                loss.backward()
                optimizer.step()

                epoch_train_loss += loss.item()

                pred = output.argmax(dim=1)
                correct_train += (pred == yb).sum().item()
                total_train += yb.size(0)
            avg_train_loss = epoch_train_loss / len(train_loader)
            train_acc = correct_train / total_train
            train_losses.append(avg_train_loss)
            train_accuracies.append(train_acc)

            # Validation
            model.eval()
            correct_val = 0
            total_val = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    output = model(xb)
                    pred = output.argmax(dim=1)
                    correct_val += (pred == yb).sum().item()
                    total_val += yb.size(0)

            val_acc = correct_val / total_val
            test_accuracies.append(val_acc)
            #tqdm.write(
             #   f"Epoch {epoch}:,Train loss = {avg_train_loss:.4f}, Train Acc = {train_acc:.4f}, Val Acc = {val_acc:.4f}")

        # acc_per_fold.append(val_acc)
        #         print(f'Score for fold {fold_no}: ')
        accuracy = print_stat(train_accuracies, test_accuracies)
        acc_per_fold.append(accuracy[0])
    return np.mean(acc_per_fold), np.std(acc_per_fold)


# ---- GRID SEARCH ----
param_grid = {
    'lr': [1e-4,5e-4, 1e-3],
    'c_channels': [16, 32, 64],
    'd_model': [32, 64, 128],
    'drop_out': [0.0,0.3],
    # 'lr': [1e-4],
    # 'c_channels': [32,64],
    # 'd_model': [64],
    # 'drop_out': [0.0],
    'nhead': [1],
    'num_layers': [2,4],
    'batch_size': [32]
}

all_params = list(product(*param_grid.values()))
best_result = {'acc': 0, 'std': 0, 'params': None}

for params in all_params:
    hyperparams = dict(zip(param_grid.keys(), params))
    mean_acc, std_acc = run_experiment(**hyperparams)
    print(f"Params: {hyperparams}, Mean Acc: {mean_acc:.4f} ± {std_acc:.4f}")
    if mean_acc > best_result['acc']:
        best_result['acc'] = mean_acc
        best_result['std'] = std_acc
        best_result['params'] = hyperparams

# --- SAVE RESULTS ---
result_str = (f"Dataset: {args.dataset} | Best Acc: {best_result['acc']:.4f} ± {best_result['std']:.4f} | "
              f"Params: {best_result['params']}\n")

print(result_str)

with open("results_summary_concat.txt", "a") as f:
    f.write(result_str)

print("\n=== BEST RESULT ===")
print(f"Best Mean Accuracy: {best_result['acc']:.4f} ± {best_result['std']:.4f}")
print(f"Best Params: {best_result['params']}")
