import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import os
from itertools import product
from sklearn.metrics import roc_auc_score

from ogb.graphproppred import PygGraphPropPredDataset

from models_2_fixed import *
from modules_fixed import *

# =============================================================
# Argument parsing
# =============================================================
parser = argparse.ArgumentParser()
parser.add_argument('--device',     type=int,   default=0)
parser.add_argument('--epochs',     type=int,   default=100)
parser.add_argument('--c_channels', type=int,   default=64)
parser.add_argument('--dataset', type=str, default='ogbg-molsider')
parser.add_argument('--d_model',    type=int,   default=32)
parser.add_argument('--lr',         type=float, default=1e-3)
parser.add_argument('--batch_size', type=int,   default=32)
args = parser.parse_args()

device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
print(f'\n=== Running on dataset: {args.dataset} ===')

# =============================================================
# Load ogbg-molhiv splits
# =============================================================
dataset   = PygGraphPropPredDataset(name=args.dataset, root='/tmp/ogb')
split_idx = dataset.get_idx_split()

train_dataset = dataset[split_idx['train']]
val_dataset   = dataset[split_idx['valid']]
test_dataset  = dataset[split_idx['test']]

print(f"Train: {len(train_dataset)}  Val: {len(val_dataset)}  Test: {len(test_dataset)}")
print(f"Sample y shape: {train_dataset[0].y.shape}")
print(f"Sample x shape: {train_dataset[0].x.shape}")

# Node feature columns — ogbg-molhiv uses the same 9-dim OGB atom encoding
# as Peptides-func:
#   0=atom_type  1=chirality  2=degree  3=formal_charge
#   4=num_Hs  5=num_radical_e  6=hybridization  7=is_aromatic  8=is_in_ring
NODE_FEAT_INDICES = (0, 1, 2, 3)

# TopoGrid bifiltration:
#   f = atom type index (col 0) — proxy for atomic number, ordinal by element
#   g = formal charge   (col 3) — ordinal, captures polarity
TOPO_F0_IDX = 0
TOPO_F1_IDX = 3

# Grid resolution
K = 10


# =============================================================
# Feature builders
# =============================================================

def build_train_views(dataset, n_bins=10, k=K,
                      feat_indices=NODE_FEAT_INDICES):
    """
    Build all three views for the TRAINING split and fit all normalization
    statistics on training data only.

    Views produced
    --------------
    Xg  : (N, 1, k,  k)   GraphGrid     — structural adjacency density
    Xt  : (N, 4, 10, 10)  TopoGrid      — multipersistence surfaces
    Xf  : (N, F, k,  k)   NodeFeatImage — block-mean node feature image
    y   : (N, 1)           binary label  (HIV inhibition)

    Stats dict (reused for val/test without modification)
    -----------------------------------------------------
    thres_f0, thres_f1  : TopoGrid threshold grids fitted on train
    topo_mean, topo_std : Z-score params for TopoGrid channels
    feat_mean, feat_std : Z-score params for NodeFeatImage channels
    """
    # ---- GraphGrid ----
    score_matrix = compute_node_features_for_graphgrid(dataset, feat_indices)
    col_order    = tuple(range(len(feat_indices)))
    _, sort_idx_list, _ = sort_dataset_score_matrices(
        score_matrix, n_bins=n_bins, col_order=col_order
    )

    graphImage = []
    for graph_id in range(len(dataset)):
        image, _, _ = adjacency_from_sorted_order_nx(
            dataset[graph_id], sort_idx_list[graph_id], k
        )
        graphImage.append(image)
    Xg = torch.tensor(np.array(graphImage), dtype=torch.float32).unsqueeze(1)

    # ---- TopoGrid: fit thresholds on TRAIN ----
    list_f0, thres_f0 = get_thresh_node_feature(
        dataset, n_bins, feat_idx=TOPO_F0_IDX
    )
    list_f1, thres_f1 = get_thresh_node_feature(
        dataset, n_bins, feat_idx=TOPO_F1_IDX
    )

    graph_features = []
    for graph_id in range(len(dataset)):
        b0, b1, node, edge = Topo_Fe_TimeSeries_MP(
            dataset[graph_id],
            list_f0[graph_id],   # atom type index per node
            list_f1[graph_id],   # formal charge per node
            thres_f0,            # TRAIN thresholds
            thres_f1,            # TRAIN thresholds
        )
        graph_features.append(torch.stack([b0, b1, node, edge], dim=0))

    Xt_raw    = torch.stack(graph_features).float()
    topo_mean = Xt_raw.mean(dim=(0, 2, 3), keepdim=True)
    topo_std  = Xt_raw.std(dim=(0, 2, 3),  keepdim=True)
    Xt        = (Xt_raw - topo_mean) / (topo_std + 1e-8)

    # ---- NodeFeatImage ----
    Xf_raw    = build_node_feature_images(dataset, sort_idx_list, k, feat_indices)
    feat_mean = Xf_raw.mean(dim=(0, 2, 3), keepdim=True)
    feat_std  = Xf_raw.std(dim=(0, 2, 3),  keepdim=True)
    Xf        = (Xf_raw - feat_mean) / (feat_std + 1e-8)

    # ---- Labels ----
    # ogbg-molhiv: y is (N, 1) binary integer — cast to float for BCEWithLogitsLoss
    y = torch.cat([dataset[i].y for i in range(len(dataset))], dim=0).float()

    stats = {
        'thres_f0' : thres_f0,
        'thres_f1' : thres_f1,
        'topo_mean': topo_mean,
        'topo_std' : topo_std,
        'feat_mean': feat_mean,
        'feat_std' : feat_std,
    }

    return Xg, Xt, Xf, y, stats


def build_eval_views(dataset, stats, n_bins=10, k=K,
                     feat_indices=NODE_FEAT_INDICES):
    """
    Build all three views for a val or test split using statistics
    fitted on the training split.
    """
    thres_f0  = stats['thres_f0']
    thres_f1  = stats['thres_f1']
    topo_mean = stats['topo_mean']
    topo_std  = stats['topo_std']
    feat_mean = stats['feat_mean']
    feat_std  = stats['feat_std']

    # ---- GraphGrid ----
    score_matrix = compute_node_features_for_graphgrid(dataset, feat_indices)
    col_order    = tuple(range(len(feat_indices)))
    _, sort_idx_list, _ = sort_dataset_score_matrices(
        score_matrix, n_bins=n_bins, col_order=col_order
    )

    graphImage = []
    for graph_id in range(len(dataset)):
        image, _, _ = adjacency_from_sorted_order_nx(
            dataset[graph_id], sort_idx_list[graph_id], k
        )
        graphImage.append(image)
    Xg = torch.tensor(np.array(graphImage), dtype=torch.float32).unsqueeze(1)

    # ---- TopoGrid: per-graph feature values, TRAIN thresholds ----
    list_f0_eval = [dataset[i].x[:, TOPO_F0_IDX].float()
                    for i in range(len(dataset))]
    list_f1_eval = [dataset[i].x[:, TOPO_F1_IDX].float()
                    for i in range(len(dataset))]

    graph_features = []
    for graph_id in range(len(dataset)):
        b0, b1, node, edge = Topo_Fe_TimeSeries_MP(
            dataset[graph_id],
            list_f0_eval[graph_id],
            list_f1_eval[graph_id],
            thres_f0,   # TRAIN thresholds
            thres_f1,   # TRAIN thresholds
        )
        graph_features.append(torch.stack([b0, b1, node, edge], dim=0))

    Xt_raw = torch.stack(graph_features).float()
    Xt     = (Xt_raw - topo_mean) / (topo_std + 1e-8)

    # ---- NodeFeatImage ----
    Xf_raw = build_node_feature_images(dataset, sort_idx_list, k, feat_indices)
    Xf     = (Xf_raw - feat_mean) / (feat_std + 1e-8)

    # ---- Labels ----
    y = torch.cat([dataset[i].y for i in range(len(dataset))], dim=0).float()

    return Xg, Xt, Xf, y


# =============================================================
# Caching
# =============================================================

# Dataset-specific cache directory to avoid collision with Peptides-func cache
save_dir = os.path.join("saved_features_ogbg", args.dataset)
os.makedirs(save_dir, exist_ok=True)

train_file = os.path.join(save_dir, "train_features.pt")
val_file   = os.path.join(save_dir, "val_features.pt")
test_file  = os.path.join(save_dir, "test_features.pt")
stats_file = os.path.join(save_dir, "train_stats.pt")


def load_or_build_all_features(train_dataset, val_dataset, test_dataset,
                               dataset_name,
                               n_bins=10, k=K, feat_indices=NODE_FEAT_INDICES):
    """
    Load from cache or build features for all splits.
    Train stats are fitted on train and reused for val/test.

    Features are saved under:
        saved_features_ogbg/<dataset_name>/
    """

    # ---- dataset-specific save folder ----
    save_dir = os.path.join("saved_features_ogbg", dataset_name)
    os.makedirs(save_dir, exist_ok=True)

    train_file = os.path.join(save_dir, "train_features.pt")
    val_file   = os.path.join(save_dir, "val_features.pt")
    test_file  = os.path.join(save_dir, "test_features.pt")
    stats_file = os.path.join(save_dir, "train_stats.pt")

    # ---- TRAIN ----
    if os.path.exists(train_file) and os.path.exists(stats_file):
        print(f"\nLoading train features from {train_file} ...")
        data  = torch.load(train_file)
        stats = torch.load(stats_file)

        Xg_train = data["Xg"]
        Xt_train = data["Xt"]
        Xf_train = data["Xf"]
        y_train  = data["y"]
    else:
        print("\nBuilding train features (fitting stats on train) ...")
        Xg_train, Xt_train, Xf_train, y_train, stats = build_train_views(
            train_dataset, n_bins=n_bins, k=k, feat_indices=feat_indices
        )

        torch.save(
            {"Xg": Xg_train, "Xt": Xt_train, "Xf": Xf_train, "y": y_train},
            train_file
        )
        torch.save(stats, stats_file)

        print(f"Saved train features → {train_file}")
        print(f"Saved train stats    → {stats_file}")

    print(f"Train: Xg {Xg_train.shape}, Xt {Xt_train.shape}, "
          f"Xf {Xf_train.shape}, y {y_train.shape}")

    # ---- VAL ----
    if os.path.exists(val_file):
        print(f"\nLoading val features from {val_file} ...")
        data = torch.load(val_file)
        Xg_val, Xt_val, Xf_val, y_val = (
            data["Xg"], data["Xt"], data["Xf"], data["y"]
        )
    else:
        print("\nBuilding val features (using train stats) ...")
        Xg_val, Xt_val, Xf_val, y_val = build_eval_views(
            val_dataset, stats, n_bins=n_bins, k=k, feat_indices=feat_indices
        )

        torch.save(
            {"Xg": Xg_val, "Xt": Xt_val, "Xf": Xf_val, "y": y_val},
            val_file
        )
        print(f"Saved val features → {val_file}")

    print(f"Val:   Xg {Xg_val.shape}, Xt {Xt_val.shape}, "
          f"Xf {Xf_val.shape}, y {y_val.shape}")

    # ---- TEST ----
    if os.path.exists(test_file):
        print(f"\nLoading test features from {test_file} ...")
        data = torch.load(test_file)
        Xg_test, Xt_test, Xf_test, y_test = (
            data["Xg"], data["Xt"], data["Xf"], data["y"]
        )
    else:
        print("\nBuilding test features (using train stats) ...")
        Xg_test, Xt_test, Xf_test, y_test = build_eval_views(
            test_dataset, stats, n_bins=n_bins, k=k, feat_indices=feat_indices
        )

        torch.save(
            {"Xg": Xg_test, "Xt": Xt_test, "Xf": Xf_test, "y": y_test},
            test_file
        )
        print(f"Saved test features → {test_file}")

    print(f"Test:  Xg {Xg_test.shape}, Xt {Xt_test.shape}, "
          f"Xf {Xf_test.shape}, y {y_test.shape}")

    return (Xg_train, Xt_train, Xf_train, y_train,
            Xg_val,   Xt_val,   Xf_val,   y_val,
            Xg_test,  Xt_test,  Xf_test,  y_test)


# NOTE: delete saved_features_ogbg_molhiv/ before first run.
(Xg_train, Xt_train, Xf_train, y_train,
 Xg_val,   Xt_val,   Xf_val,   y_val,
 Xg_test,  Xt_test,  Xf_test,  y_test) = load_or_build_all_features(
    train_dataset, val_dataset, test_dataset,args.dataset
)

# ogbg-molhiv is a single binary task
num_tasks = y_train.shape[1]   # = 1
print(f"\nNumber of tasks: {num_tasks}  (binary classification)")


# =============================================================
# Loss — BCEWithLogitsLoss handles binary classification
# NaN masking is included for consistency with OGB convention
# =============================================================
cls_criterion = nn.BCEWithLogitsLoss()


def compute_supervised_loss(logits, y_true):
    y_true     = y_true.view_as(logits)
    is_labeled = (y_true == y_true)       # mask any NaN labels
    if is_labeled.sum() == 0:
        return None
    return cls_criterion(
        logits[is_labeled].float(),
        y_true[is_labeled].float()
    )


# =============================================================
# Evaluation — ROC-AUC (OGB standard metric for molhiv)
# =============================================================

@torch.no_grad()
def evaluate_rocauc(model, xg, xt, xf, y_true, batch_size, device):
    model.eval()
    y_pred_list, y_true_list = [], []
    n = xg.size(0)

    for start in range(0, n, batch_size):
        end  = min(start + batch_size, n)
        xb1  = xg[start:end].to(device, non_blocking=True)
        xb2  = xt[start:end].to(device, non_blocking=True)
        xb3  = xf[start:end].to(device, non_blocking=True)
        yb   = y_true[start:end]

        logits, *_ = model(xb1, xb2, xb3)
        y_pred_list.append(logits.detach().cpu())
        y_true_list.append(yb)

    y_pred = torch.cat(y_pred_list, dim=0).numpy()   # (N, 1)
    y_true = torch.cat(y_true_list, dim=0).numpy()   # (N, 1)

    # Mask NaN labels (OGB convention)
    mask   = ~np.isnan(y_true[:, 0])
    y_true = y_true[mask, 0]
    y_pred = y_pred[mask, 0]

    # Need both classes present to compute ROC-AUC
    if len(np.unique(y_true)) < 2:
        return {"rocauc": 0.0}

    auc = roc_auc_score(y_true, y_pred)
    return {"rocauc": float(auc)}


# =============================================================
# Training loop
# =============================================================

def train_three_view_molhiv(
    model,
    Xg_train, Xt_train, Xf_train, y_train,
    Xg_valid, Xt_valid, Xf_valid, y_valid,
    Xg_test,  Xt_test,  Xf_test,  y_test,
    device,
    epochs: int            = 200,
    batch_size: int        = 32,
    lr: float              = 1e-3,
    weight_decay: float    = 1e-4,
    lambda_contrast: float = 0.1,
    temperature: float     = 0.2,
    use_amp: bool          = True,
    grad_clip: float       = 1.0,
):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=weight_decay)
    scaler    = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_val, best_test, best_state = -float("inf"), -float("inf"), None
    n_train = Xg_train.size(0)

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n_train)
        running_loss = running_sup = running_con = seen = 0

        for start in range(0, n_train, batch_size):
            end = min(start + batch_size, n_train)
            idx = perm[start:end]

            x1 = Xg_train[idx].to(device, non_blocking=True)
            x2 = Xt_train[idx].to(device, non_blocking=True)
            x3 = Xf_train[idx].to(device, non_blocking=True)
            yb = y_train[idx].to(device,  non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                logits, h1, h2, h3, z1, z2, z3 = model(x1, x2, x3)

                sup_loss = compute_supervised_loss(logits, yb)
                if sup_loss is None:
                    continue

                # All three pairwise contrastive terms averaged
                con_loss = (
                    nt_xent_loss(z1, z2, temperature) +
                    nt_xent_loss(z1, z3, temperature) +
                    nt_xent_loss(z2, z3, temperature)
                ) / 3.0

                loss = sup_loss + lambda_contrast * con_loss

            scaler.scale(loss).backward()
            if grad_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()

            bs            = yb.size(0)
            running_loss += loss.item()     * bs
            running_sup  += sup_loss.item() * bs
            running_con  += con_loss.item() * bs
            seen         += bs

        train_loss = running_loss / max(seen, 1)
        train_sup  = running_sup  / max(seen, 1)
        train_con  = running_con  / max(seen, 1)

        valid_score = evaluate_rocauc(
            model, Xg_valid, Xt_valid, Xf_valid, y_valid, batch_size, device
        )["rocauc"]
        test_score  = evaluate_rocauc(
            model, Xg_test,  Xt_test,  Xf_test,  y_test,  batch_size, device
        )["rocauc"]

        if valid_score > best_val:
            best_val   = valid_score
            best_test  = test_score
            best_state = {k: v.detach().cpu()
                          for k, v in model.state_dict().items()}

        print(
            f"Epoch {epoch:03d} | "
            f"loss {train_loss:.4f} (sup {train_sup:.4f}, con {train_con:.4f}) | "
            f"val ROC-AUC {valid_score:.4f} | test ROC-AUC {test_score:.4f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_test


# =============================================================
# Experiment runner
# =============================================================

def run_experiment(
    Xg_train, Xt_train, Xf_train, y_train,
    Xg_val,   Xt_val,   Xf_val,   y_val,
    Xg_test,  Xt_test,  Xf_test,  y_test,
    num_tasks,
    lr, c_channels, d_model, drop_out, nhead, num_layers, batch_size,
    lambda_contrast=0.1,
    temperature=0.2,
    proj_dim=128,
    fuse="concat",
    weight_decay=1e-4,
    use_amp=True,
    grad_clip=1.0,
):
    C1 = int(Xg_train.shape[1])
    C2 = int(Xt_train.shape[1])
    C3 = int(Xf_train.shape[1])

    print(f"View channels — GraphGrid: {C1}, TopoGrid: {C2}, NodeFeatImg: {C3}")
    print(f"Output tasks: {num_tasks}")

    model = ThreeViewContrastiveClassifier(
        num_classes=num_tasks,
        in_channels_view1=C1,
        in_channels_view2=C2,
        in_channels_view3=C3,
        cnn_channels=c_channels,
        d_model=d_model,
        drop_out=drop_out,
        nhead=nhead,
        num_layers=num_layers,
        proj_dim=proj_dim,
        fuse=fuse,
    )

    model, best_test = train_three_view_molhiv(
        model=model,
        Xg_train=Xg_train, Xt_train=Xt_train, Xf_train=Xf_train, y_train=y_train,
        Xg_valid=Xg_val,   Xt_valid=Xt_val,   Xf_valid=Xf_val,   y_valid=y_val,
        Xg_test=Xg_test,   Xt_test=Xt_test,   Xf_test=Xf_test,   y_test=y_test,
        device=device,
        epochs=args.epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        lambda_contrast=lambda_contrast,
        temperature=temperature,
        use_amp=use_amp,
        grad_clip=grad_clip,
    )

    train_auc = evaluate_rocauc(
        model, Xg_train, Xt_train, Xf_train, y_train, batch_size, device
    )["rocauc"]
    valid_auc = evaluate_rocauc(
        model, Xg_val, Xt_val, Xf_val, y_val, batch_size, device
    )["rocauc"]

    print(f"Train ROC-AUC: {train_auc:.4f} | Val ROC-AUC: {valid_auc:.4f} | "
          f"Test ROC-AUC: {best_test:.4f}")

    return {"train": train_auc, "valid": valid_auc, "test": best_test}


# =============================================================
# Grid search — select best by validation ROC-AUC
# =============================================================
param_grid = {
    'lr':              [1e-3],
    'c_channels':      [64],
    'd_model':         [128],
    'drop_out':        [0.0],
    'nhead':           [2,4],
    'num_layers':      [2],
    'batch_size':      [32],
    'lambda_contrast': [0.05, 0.1],
    'temperature':     [0.2],
    'proj_dim':        [128],
    'fuse':            ['concat'],
    'weight_decay':    [1e-4],
    'use_amp':         [True],
    'grad_clip':       [1.0],
}

all_params  = list(product(*param_grid.values()))
best_result = {
    "valid": -float("inf"), "test": -float("inf"),
    "train": -float("inf"), "params": None,
}

for params in all_params:
    hyperparams = dict(zip(param_grid.keys(), params))
    print(f"\n--- Params: {hyperparams} ---")

    result = run_experiment(
        Xg_train=Xg_train, Xt_train=Xt_train, Xf_train=Xf_train, y_train=y_train,
        Xg_val=Xg_val,     Xt_val=Xt_val,     Xf_val=Xf_val,     y_val=y_val,
        Xg_test=Xg_test,   Xt_test=Xt_test,   Xf_test=Xf_test,   y_test=y_test,
        num_tasks=num_tasks,
        **hyperparams,
    )

    if result["test"] > best_result["test"]:
        best_result.update({
            "train":  result["train"],
            "valid":  result["valid"],
            "test":   result["test"],
            "params": hyperparams,
        })

print("\n=== BEST RESULT ===")
print(f"Train ROC-AUC : {best_result['train']:.4f}")
print(f"Valid ROC-AUC : {best_result['valid']:.4f}")
print(f"Test  ROC-AUC : {best_result['test']:.4f}")
print(f"Params        : {best_result['params']}")
