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
import matplotlib.pyplot as plt

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
y = torch.tensor(label, dtype=torch.long)

sample_idx = 23
image_to_plot = G_image[sample_idx].squeeze().cpu().numpy()

plt.figure(figsize=(5, 5))
plt.imshow(image_to_plot, cmap='viridis', interpolation='nearest')
plt.colorbar(label='Pixel Intensity')
plt.title(f'Graph {sample_idx} Adjacency Image (10x10), , label {y[sample_idx]}')
plt.xlabel('Node Bin')
plt.ylabel('Node Bin')
plt.show()

#sample_idx = 0
topo_data = Toposurface[sample_idx].cpu().numpy()  # Shape: (4, 10, 10)

feature_names = ["Betti-0", "Betti-1", "Num Nodes", "Num Edges"]

fig, axes = plt.subplots(1, 4, figsize=(20, 5))

for i in range(4):
    # Select channel i
    channel_img = topo_data[i]

    im = axes[i].imshow(channel_img, cmap='inferno', interpolation='nearest')
    axes[i].set_title(f"{feature_names[i]}")
    axes[i].axis('off')  # Hide axis ticks for cleaner look

    # Add colorbar to see value range
    plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

plt.suptitle(f"Topological Features for Graph {sample_idx}, label {y[sample_idx]}", fontsize=16)
plt.tight_layout()
plt.show()


