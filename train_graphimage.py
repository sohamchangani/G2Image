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
parser.add_argument('--dataset', type=str, default='proteins')
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

G_image = torch.tensor(graphImage,dtype=torch.float).unsqueeze(1)

print(graphImage[0])
list_hks, thres_hks, label = get_thresh_hks(dataset, 10, 0.1)
list_deg, thres_deg = get_thresh(dataset, 10)
graph_features = []
#for graph_id in tqdm(range(len(dataset))):
for graph_id in range(len(dataset)):
    b0, b1, node, edge = Topo_Fe_TimeSeries_MP(dataset[graph_id], list_deg[graph_id], list_hks[graph_id],
                                               thres_deg, thres_hks)
    graph_features.append(torch.stack([b0, b1, node, edge], dim=0))
Toposurface = torch.stack(graph_features)
X = Toposurface
print(X.shape)
#print(X[0])
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


def train_two_view(
    model,
    train_loader,
    val_loader,
    device,
    epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    lambda_contrast: float = 0.1,
    temperature: float = 0.2,
    use_amp: bool = True,
    grad_clip: float = 1.0,
):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_val_acc = -1.0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        running = {"loss": 0.0, "ce": 0.0, "con": 0.0, "n": 0}

        for x1, x2, y in train_loader:
            x1 = x1.to(device, non_blocking=True)
            x2 = x2.to(device, non_blocking=True)
            y  = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                logits, h1, h2, z1, z2 = model(x1, x2)

                ce = F.cross_entropy(logits, y)
                con = nt_xent_loss(z1, z2, temperature=temperature)

                loss = ce + lambda_contrast * con

            scaler.scale(loss).backward()

            if grad_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            scaler.step(optimizer)
            scaler.update()

            bs = y.size(0)
            running["loss"] += loss.item() * bs
            running["ce"]   += ce.item() * bs
            running["con"]  += con.item() * bs
            running["n"]    += bs

        train_metrics = {k: running[k] / running["n"] for k in ("loss", "ce", "con")}

        val_metrics = evaluate(model, val_loader, device) if val_loader is not None else {"loss": float("nan"), "acc": float("nan")}

        # Save best
        if val_metrics["acc"] > best_val_acc:
            best_val_acc = val_metrics["acc"]
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

       # print(
        #    f"Epoch {epoch:03d} | "
        #    f"train loss {train_metrics['loss']:.4f} (ce {train_metrics['ce']:.4f}, con {train_metrics['con']:.4f}) | "
         #   f"val loss {val_metrics['loss']:.4f} acc {val_metrics['acc']:.4f}"
        #)

    if best_state is not None:
        model.load_state_dict(best_state)
    #return model
    return best_val_acc


# Assumes these are already defined in your notebook/script:
# - TwoViewContrastiveClassifier
# - train_two_view(model, train_loader, val_loader, device, ...)
# - evaluate(model, loader, device)

def run_experiment_two_view(
    X_graph, X_topo, y,
    lr, c_channels, d_model, drop_out, nhead, num_layers, batch_size,
    lambda_contrast=0.1,
    temperature=0.2,
    proj_dim=128,
    fuse="concat",
    share_encoder=False,
    weight_decay=1e-4,
    use_amp=True,
    grad_clip=1.0,
):
    """
    K-fold experiment using:
      - TwoViewContrastiveClassifier
      - train_two_view
      - evaluate

    Inputs
    ------
    X_graph: torch.Tensor (N, C1, H, W)
    X_topo : torch.Tensor (N, C2, H, W)
    y      : torch.Tensor (N,)
    """
    kf = KFold(n_splits=args.folds, shuffle=True, random_state=42)
    acc_per_fold = []

    num_classes = int(torch.unique(y).numel())
    C1 = int(X_graph.shape[1])
    C2 = int(X_topo.shape[1])
    print(C1)
    print(C2)

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_graph), 1):
        # Split
        Xg_train, Xg_val = X_graph[train_idx], X_graph[val_idx]
        Xt_train, Xt_val = X_topo[train_idx],  X_topo[val_idx]
        y_train, y_val   = y[train_idx],       y[val_idx]

        # Datasets / loaders
        train_ds = TensorDataset(Xg_train, Xt_train, y_train)
        val_ds   = TensorDataset(Xg_val,   Xt_val,   y_val)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True)
        val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=True)

        # Model
        model = TwoViewContrastiveClassifier(
            num_classes=num_classes,
            in_channels_view1=C1,
            in_channels_view2=C2,
            cnn_channels=c_channels,
            d_model=d_model,
            drop_out=drop_out,
            nhead=nhead,
            num_layers=num_layers,
            proj_dim=proj_dim,
            share_encoder=share_encoder,
            fuse=fuse,
        )

        # Train using your earlier function
        val_acc = train_two_view(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=args.epochs,
            lr=lr,
            weight_decay=weight_decay,
            lambda_contrast=lambda_contrast,
            temperature=temperature,
            use_amp=use_amp,
            grad_clip=grad_clip,
        )

        # Final fold evaluation (best model already loaded inside train_two_view)
        #val_metrics = evaluate(model, val_loader, device)
        #acc_per_fold.append(val_metrics["acc"])
        acc_per_fold.append(val_acc)

        print(f"[Fold {fold}] Final Val Acc: {val_acc:.4f}")

    return float(np.mean(acc_per_fold)), float(np.std(acc_per_fold))

from itertools import product

# ---- GRID SEARCH ----
param_grid = {
    # model/optim params
    'lr': [1e-3],
    'c_channels': [16],
    'd_model': [32],
    'drop_out': [0.0],
    'nhead': [1],
    'num_layers': [2],
    'batch_size': [32],

    # (optional) contrastive params
    "lambda_contrast": [0.1],   # try [0.05, 0.1, 0.2, 0.5]
    "temperature": [0.2],       # try [0.07, 0.1, 0.2, 0.5]
    "proj_dim": [128],          # try [64, 128, 256]

    # (optional) fusion/architecture
    "fuse": ["concat"],         # try ["concat", "sum", "mean"]
    "share_encoder": [False],   # True only if C1 == C2 and you want weight sharing

    # (optional) regularization/training
    "weight_decay": [1e-4],
    "use_amp": [True],
    "grad_clip": [1.0],
}

all_params = list(product(*param_grid.values()))
best_result = {"acc": 0.0, "std": 0.0, "params": None}

for params in all_params:
    hyperparams = dict(zip(param_grid.keys(), params))

    mean_acc, std_acc = run_experiment_two_view(
        X_graph=G_image,
        X_topo=Toposurface,
        y=y,
        **hyperparams
    )

    print(f"Params: {hyperparams}, Mean Acc: {mean_acc:.4f} ± {std_acc:.4f}")

    if mean_acc > best_result["acc"]:
        best_result["acc"] = mean_acc
        best_result["std"] = std_acc
        best_result["params"] = hyperparams

print("\n=== BEST RESULT ===")
print(f"Best Mean Accuracy: {best_result['acc']:.4f} ± {best_result['std']:.4f}")
print(f"Best Params: {best_result['params']}")

