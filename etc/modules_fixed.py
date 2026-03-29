import torch
from torch_geometric.utils import degree
import pyflagser
import numpy as np
import networkx as nx
import joblib
import statistics
import pandas as pd
import scipy.sparse as sp

from collections import Counter
from torch_geometric.utils import to_networkx


# -----------------------------------------------------------------------
# Threshold computation
# -----------------------------------------------------------------------

def process_thresholds(lst, N):
    """
    Compute N quantile thresholds from a list or tensor of scalar values.
    Uses evenly-spaced quantiles, which gives meaningful and stable threshold
    grids for continuous scores like HKS or node features.
    """
    if N < 2:
        raise ValueError("N must be at least 2 to include min and max thresholds.")
    arr = np.array([v.item() if hasattr(v, 'item') else float(v) for v in lst])
    quantile_points = np.linspace(0.0, 1.0, N)
    thresholds = np.quantile(arr, quantile_points).tolist()
    seen, unique_thresholds = set(), []
    for t in thresholds:
        if t not in seen:
            seen.add(t)
            unique_thresholds.append(t)
    return sorted(unique_thresholds)


# -----------------------------------------------------------------------
# Basic node-score helpers (structural — kept for TU datasets)
# -----------------------------------------------------------------------

def get_degree_centrality(data):
    edge_index = data.edge_index
    num_nodes  = data.num_nodes
    deg = degree(edge_index[0], num_nodes=num_nodes)
    return deg / (num_nodes - 1)


def get_atomic_weight(data):
    return torch.tensor([data.x[i][0] for i in range(len(data.x))])


def compute_hks(graph, t_values=0.1):
    L = nx.laplacian_matrix(graph).toarray()
    eigvals, eigvecs = np.linalg.eigh(L)
    heat_kernel = np.dot(eigvecs, np.dot(np.diag(np.exp(-t_values * eigvals)), eigvecs.T))
    hks = [heat_kernel[i, i] for i, _ in enumerate(graph.nodes())]
    return torch.tensor(hks, dtype=torch.float32).unsqueeze(1)


def compute_degree_centrality(graph):
    dc = nx.degree_centrality(graph)
    scores = [dc[node] for node in graph.nodes()]
    return torch.tensor(scores, dtype=torch.float32).unsqueeze(1)


def compute_pagerank(graph, alpha=0.85, max_iter=100, tol=1e-6, weight=None):
    pr = nx.pagerank(graph, alpha=alpha, max_iter=max_iter, tol=tol, weight=weight)
    scores = [pr[node] for node in graph.nodes()]
    return torch.tensor(scores, dtype=torch.float32).unsqueeze(1)


def compute_kcore_score(graph):
    G = graph.to_undirected() if graph.is_directed() else graph
    if G.number_of_nodes() == 0:
        return torch.empty((0,), dtype=torch.float32)
    core = nx.core_number(G) if G.number_of_edges() > 0 else {n: 0 for n in G.nodes()}
    scores = [core[node] for node in graph.nodes()]
    return torch.tensor(scores, dtype=torch.float32).unsqueeze(1)


def compute_node_features_hks_deg_kcore_pagerank(dataset):
    """Structural scores for GraphGrid/TopoGrid — used on TU datasets."""
    score = []
    for graph_id in range(len(dataset)):
        graph = to_networkx(dataset[graph_id], to_undirected=True)
        hks   = compute_hks(graph)
        dc    = compute_degree_centrality(graph)
        kcore = compute_kcore_score(graph)
        pr    = compute_pagerank(graph)
        score.append(torch.cat([hks, dc, kcore, pr], dim=1))
    return score


def get_thres_atom(dataset, number_threshold):
    graph_list = []
    for graph_id in range(len(dataset)):
        graph_list.append(get_atomic_weight(dataset[graph_id]))
    thresh = torch.cat(graph_list, dim=0)
    thresh = process_thresholds(thresh, number_threshold)
    return graph_list, torch.tensor(thresh)


def get_thresh(dataset, number_threshold):
    graph_list = []
    for graph_id in range(len(dataset)):
        graph_list.append(get_degree_centrality(dataset[graph_id]))
    thresh = torch.cat(graph_list, dim=0)
    thresh = process_thresholds(thresh, number_threshold)
    return graph_list, torch.tensor(thresh)


def get_thresh_hks(dataset, number_threshold, t_value):
    graph_list, label = [], []
    for graph_id in range(len(dataset)):
        graph = to_networkx(dataset[graph_id], to_undirected=True)
        graph_list.append(compute_hks(graph, t_value))
        label.append(dataset[graph_id].y)
    thresh = torch.cat(graph_list, dim=0)
    thresh = process_thresholds(thresh, number_threshold)
    return graph_list, torch.tensor(thresh), label


# -----------------------------------------------------------------------
# Node-feature-based scoring — for datasets with rich node attributes
# (e.g. Peptides-func, ogbg-molhiv, ogbg-molbace)
# -----------------------------------------------------------------------

def compute_node_feature_scores(data, feat_indices=(0, 1, 2, 3)):
    """
    Extract selected columns from data.x as per-node scores.

    For Peptides-func (9 node features):
      0 = atom type      1 = chirality     2 = degree
      3 = formal charge  4 = num_Hs        5 = num_radical_e
      6 = hybridization  7 = is_aromatic   8 = is_in_ring

    Default: (0, 1, 2, 3) — atom type, chirality, degree, formal charge.
    These four encode the most chemically discriminative information and
    mirror the 4-column format used by the structural scoring pipeline.

    Returns
    -------
    X : (N, len(feat_indices)) float tensor
    """
    if data.x is None:
        raise ValueError("Graph has no node features (data.x is None).")
    return data.x[:, list(feat_indices)].float()


def compute_node_features_for_graphgrid(dataset, feat_indices=(0, 1, 2, 3)):
    """
    Node-feature-based drop-in replacement for
    compute_node_features_hks_deg_kcore_pagerank.
    Use this for datasets with meaningful node attributes.

    Returns
    -------
    score : list of (N_i, 4) tensors
    """
    score = []
    for graph_id in range(len(dataset)):
        score.append(compute_node_feature_scores(dataset[graph_id], feat_indices))
    return score


def get_thresh_node_feature(dataset, number_threshold, feat_idx=0):
    """
    Compute quantile thresholds for a single node feature column,
    pooled across the dataset. Call only on the training split.

    Returns
    -------
    graph_list : list of (N_i,) tensors
    thresholds : 1-D float tensor of length number_threshold
    """
    graph_list = []
    for graph_id in range(len(dataset)):
        graph_list.append(dataset[graph_id].x[:, feat_idx].float())
    all_vals   = torch.cat(graph_list, dim=0)
    thresholds = process_thresholds(all_vals.tolist(), number_threshold)
    return graph_list, torch.tensor(thresholds, dtype=torch.float32)


# -----------------------------------------------------------------------
# Node-feature image (third view, Option 2)
# -----------------------------------------------------------------------

def compute_node_feature_image(data, sort_idx, k, feat_indices=(0, 1, 2, 3)):
    """
    Build a (F, k, k) node-feature image aligned with the GraphGrid ordering.

    Nodes are sorted by sort_idx (same as GraphGrid) and split into k blocks.
    For each block pair (i, j) and each feature dimension d:
        FeatImage[d, i, j] = mean_feat(block_i)[d] * mean_feat(block_j)[d]

    The elementwise product of block means encodes chemical similarity
    between interacting groups: it is high when both blocks share the same
    feature profile, and near zero when they differ. The diagonal captures
    group self-identity; off-diagonal captures cross-group chemistry.

    This representation is:
      - Permutation-invariant (nodes within each block are averaged)
      - Fixed-size (k × k × F regardless of graph size)
      - Consistent with the GraphGrid block structure

    Parameters
    ----------
    data         : PyG Data object with node features
    sort_idx     : (N,) array — same ordering used for GraphGrid
    k            : grid resolution (same k as GraphGrid)
    feat_indices : which columns of data.x to use

    Returns
    -------
    feat_image : (F, k, k) float32 numpy array
    """
    X_np     = compute_node_feature_scores(data, feat_indices).numpy()  # (N, F)
    F_dim    = X_np.shape[1]
    # Guard: sort_idx may be a CPU or GPU tensor — always convert to numpy int array
    if torch.is_tensor(sort_idx):
        sort_idx = sort_idx.cpu().numpy()
    sort_idx = np.asarray(sort_idx, dtype=int)
    X_sorted = X_np[sort_idx]                                            # reorder

    blocks     = np.array_split(np.arange(len(sort_idx)), k)
    feat_image = np.zeros((F_dim, k, k), dtype=np.float32)

    for i, Bi in enumerate(blocks):
        if len(Bi) == 0:
            continue
        mean_i = X_sorted[Bi].mean(axis=0)    # (F,)
        for j, Bj in enumerate(blocks):
            if len(Bj) == 0:
                continue
            mean_j = X_sorted[Bj].mean(axis=0)  # (F,)
            feat_image[:, i, j] = mean_i * mean_j

    return feat_image


def build_node_feature_images(dataset, sort_idx_list, k,
                               feat_indices=(0, 1, 2, 3)):
    """
    Build the (N_graphs, F, k, k) node-feature image tensor for a split.

    Parameters
    ----------
    dataset       : PyG dataset split
    sort_idx_list : list of (N_i,) arrays from sort_dataset_score_matrices
    k             : grid resolution
    feat_indices  : which columns of data.x to use

    Returns
    -------
    Xf : (N_graphs, F, k, k) float32 tensor
    """
    images = []
    for graph_id in range(len(dataset)):
        img = compute_node_feature_image(
            dataset[graph_id],
            sort_idx_list[graph_id],
            k,
            feat_indices=feat_indices,
        )
        images.append(img)
    return torch.tensor(np.array(images), dtype=torch.float32)


# -----------------------------------------------------------------------
# TopoGrid computation
# -----------------------------------------------------------------------

def Topo_Fe_TimeSeries_MP(graph, feature1, feature2, threshold1, threshold2):
    """
    Compute Betti-0, Betti-1, num_nodes, num_edges for all threshold pairs.
    feature1/feature2 are per-node scalar tensors (structural or node-feature based).
    threshold1/threshold2 are the fitted quantile grids.
    """
    betti_0_all   = []
    betti_1_all   = []
    num_nodes_all = []
    num_edges_all = []

    edge_index = graph.edge_index
    feature1   = feature1.view(-1)
    feature2   = feature2.view(-1)
    threshold1 = threshold1.view(-1)
    threshold2 = threshold2.view(-1)

    for p in range(threshold1.size(0)):
        betti0_row, betti1_row, nodes_row, edges_row = [], [], [], []

        for q in range(threshold2.size(0)):
            idx1     = torch.where(feature1 <= threshold1[p])[0]
            idx2     = torch.where(feature2 <= threshold2[q])[0]
            n_active = torch.tensor(
                list(set(idx1.tolist()) & set(idx2.tolist())), dtype=torch.long
            )

            if n_active.numel() == 0:
                betti0_row.append(0)
                betti1_row.append(0)
                nodes_row.append(0)
                edges_row.append(0)
            else:
                active_set = set(n_active.tolist())
                G = nx.Graph()
                G.add_nodes_from(active_set)
                for uu, vv in zip(edge_index[0].tolist(), edge_index[1].tolist()):
                    if uu in active_set and vv in active_set:
                        G.add_edge(int(uu), int(vv))

                Adj = nx.to_numpy_array(G, nodelist=sorted(active_set))
                my_flag = pyflagser.flagser_unweighted(
                    Adj, min_dimension=0, max_dimension=2,
                    directed=False, coeff=2, approximation=None
                )
                x = my_flag["betti"]
                betti0_row.append(int(x[0]))
                betti1_row.append(int(x[1]) if len(x) > 1 else 0)
                nodes_row.append(len(active_set))
                edges_row.append(G.number_of_edges())

        betti_0_all.append(betti0_row)
        betti_1_all.append(betti1_row)
        num_nodes_all.append(nodes_row)
        num_edges_all.append(edges_row)

    return (
        torch.tensor(betti_0_all,   dtype=torch.float),
        torch.tensor(betti_1_all,   dtype=torch.float),
        torch.tensor(num_nodes_all, dtype=torch.float),
        torch.tensor(num_edges_all, dtype=torch.float),
    )


# -----------------------------------------------------------------------
# GraphGrid adjacency helpers
# -----------------------------------------------------------------------

def block_pool_adjacency(A_star, k):
    A_star = np.asarray(A_star, dtype=float)
    if A_star.shape[0] != A_star.shape[1]:
        raise ValueError("A_star must be square.")
    blocks = np.array_split(np.arange(A_star.shape[0]), k)
    I = np.zeros((k, k), dtype=float)
    for i, Bi in enumerate(blocks):
        for j, Bj in enumerate(blocks):
            if len(Bi) > 0 and len(Bj) > 0:
                I[i, j] = A_star[np.ix_(Bi, Bj)].mean()
    return I


def adjacency_from_sorted_order_nx(
    graph, sort_idx, k, to_undirected=True, include_self_loops=False, weight=None
):
    G        = to_networkx(graph)
    nodes    = list(G.nodes())
    sort_idx = np.asarray(sort_idx, dtype=int)
    if len(sort_idx) != len(nodes):
        raise ValueError("sort_idx length must equal number of nodes in G.")
    nodes_sorted = [nodes[i] for i in sort_idx]
    H = G.to_undirected() if (to_undirected and G.is_directed()) else G
    A_sorted = nx.to_numpy_array(H, nodelist=nodes_sorted, weight=weight, dtype=float)
    if include_self_loops:
        np.fill_diagonal(A_sorted, 1.0)
    I = block_pool_adjacency(A_sorted, k)
    return I, A_sorted, nodes_sorted


# -----------------------------------------------------------------------
# Sorting / binning
# -----------------------------------------------------------------------

def quantile_bin_1d(x, n_bins=10):
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x.astype(np.int64)
    if np.allclose(x.min(), x.max()):
        return np.zeros_like(x, dtype=np.int64)
    qs    = np.linspace(0, 1, n_bins + 1)[1:-1]
    edges = np.quantile(x, qs, method="linear")
    return np.searchsorted(edges, x, side="right").astype(np.int64)


def sort_nodes_lexicographically_with_quantiles(
    score_mat, n_bins=10, col_order=None, tie_break_by_node_id=True
):
    """
    score_mat : (N, C) tensor or ndarray
    col_order : priority order of columns (most important = last in lexsort).
                Defaults to (0, 1, ..., C-1).
    """
    is_torch = torch.is_tensor(score_mat)
    M = score_mat.detach().cpu().numpy() if is_torch else np.asarray(score_mat)
    if M.ndim != 2:
        raise ValueError(f"Expected (N, C) matrix. Got {M.shape}.")

    N = M.shape[0]
    if col_order is None:
        col_order = tuple(range(M.shape[1]))

    binned = np.zeros_like(M, dtype=np.int64)
    for c in range(M.shape[1]):
        binned[:, c] = quantile_bin_1d(M[:, c], n_bins=n_bins)

    if tie_break_by_node_id:
        keys = [np.arange(N, dtype=np.int64)] + [binned[:, c] for c in reversed(col_order)]
    else:
        keys = [binned[:, c] for c in reversed(col_order)]

    sort_idx      = np.lexsort(keys)
    M_sorted      = M[sort_idx]
    binned_sorted = binned[sort_idx]

    if is_torch:
        return (
            torch.as_tensor(M_sorted, dtype=score_mat.dtype, device=score_mat.device),
            torch.as_tensor(sort_idx, dtype=torch.long,      device=score_mat.device),
            binned_sorted,
        )
    return M_sorted, sort_idx, binned_sorted


def sort_dataset_score_matrices(
    score_matrix_list, n_bins=10, col_order=None, tie_break_by_node_id=True
):
    sorted_mats, sort_indices, binned_mats = [], [], []
    for S in score_matrix_list:
        S_sorted, idx, binned = sort_nodes_lexicographically_with_quantiles(
            S, n_bins=n_bins, col_order=col_order,
            tie_break_by_node_id=tie_break_by_node_id,
        )
        sorted_mats.append(S_sorted)
        sort_indices.append(idx)
        binned_mats.append(binned)
    return sorted_mats, sort_indices, binned_mats


# -----------------------------------------------------------------------
# Utility
# -----------------------------------------------------------------------

def stat(acc_list, metric):
    mean  = statistics.mean(acc_list)
    stdev = statistics.stdev(acc_list)
    print('Final', metric, f'using 5 fold CV: {mean:.4f} \u00B1 {stdev:.4f}%')


def print_stat(train_acc, test_acc):
    argmax      = np.argmax(test_acc)
    best_result = test_acc[argmax]
    train_ac    = train_acc[argmax]
    print(f'Train accuracy = {train_ac:.4f}%, Test Accuracy = {best_result:.4f}%\n')
    return best_result, train_ac


def apply_Zscore(MP_tensor):
    mean = MP_tensor.mean(dim=(0, 2, 3), keepdim=True)
    std  = MP_tensor.std(dim=(0, 2, 3), keepdim=True)
    return (MP_tensor - mean) / (std + 1e-8)
