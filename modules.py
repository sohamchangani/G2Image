import torch
from torch_geometric.utils import degree
import pyflagser
import numpy as np
import networkx as nx
import joblib
import statistics
import pandas as pd

from collections import Counter


def process_thresholds(lst, N):
    if N < 2:
        raise ValueError("N must be at least 2 to include min and max thresholds.")

    # Count occurrences of each value
    count = Counter(lst)

    # Find the minimum and maximum values
    min_val, max_val = min(lst), max(lst)

    # Remove min and max from the counting
    count.pop(min_val, None)
    count.pop(max_val, None)

    # Select the N-1 values with the highest counts
    top_values = sorted(count.items(), key=lambda x: x[1], reverse=True)[:N - 1]

    # Prepare the thresholds: a_0=min, top N-1 values, a_N=max
    thresholds = [min_val] + [value for value, _ in top_values] + [max_val]

    return sorted(thresholds)
def get_degree_centrality(data):
    # Assume edge_index is of shape [2, num_edges]
    # and contains edges in COO format

    edge_index = data.edge_index  # shape: [2, E]
    num_nodes = data.num_nodes    # or x.shape[0]

    # Compute degree for each node
    deg = degree(edge_index[0], num_nodes=num_nodes)  # degree of each node

    # Normalize to get degree centrality (divide by max possible degree)
    deg_centrality = deg / (num_nodes - 1)
    return deg_centrality

def get_atomic_weight(data):
    # Assume edge_index is of shape [2, num_edges]
    # and contains edges in COO format
    Atomic_weight=[data.x[i][0] for i in range(len(data.x))]
    return torch.tensor(Atomic_weight)

from torch_geometric.utils import to_networkx


def compute_hks(graph, t_values=0.1):
    """
    Compute the Heat Kernel Signature (HKS) for each node in the graph.
    :param graph: NetworkX graph (undirected, unweighted)
    :param t_values: List of diffusion time values to compute HKS
    :return: Dictionary with nodes as keys and HKS values as lists
    """
    L = nx.laplacian_matrix(graph).toarray()
    eigvals, eigvecs = np.linalg.eigh(L)

    hks = []

    heat_kernel = np.dot(eigvecs, np.dot(np.diag(np.exp(-t_values * eigvals)), eigvecs.T))
    for i, node in enumerate(graph.nodes()):
        hks.append(heat_kernel[i, i])
    return torch.tensor(hks, dtype=torch.float32).unsqueeze(1)

def compute_degree_centrality(graph):
    """
    Degree centrality per node.
    :param graph: NetworkX graph
    :return: torch.tensor of shape (num_nodes,)
    """
    dc = nx.degree_centrality(graph)  # dict: node -> score
    scores = [dc[node] for node in graph.nodes()]  # preserve node order
    return torch.tensor(scores, dtype=torch.float32).unsqueeze(1)


def compute_pagerank(graph, alpha=0.85, max_iter=100, tol=1e-6, weight=None):
    """
    PageRank score per node (sums to 1).
    :param graph: NetworkX graph (Graph or DiGraph)
    :param alpha: damping factor
    :param max_iter: max iterations
    :param tol: convergence tolerance
    :param weight: edge attribute name to use as weight (None = unweighted)
    :return: torch.tensor of shape (num_nodes,)
    """
    pr = nx.pagerank(graph, alpha=alpha, max_iter=max_iter, tol=tol, weight=weight)
    scores = [pr[node] for node in graph.nodes()]  # preserve node order
    return torch.tensor(scores, dtype=torch.float32).unsqueeze(1)

def compute_kcore_score(graph):
    """
    K-core score per node = core number (max k such that node is in k-core).
    Note: core_number is defined for undirected graphs; if graph is directed,
    we convert to undirected (common practice).
    :param graph: NetworkX graph
    :return: torch.tensor of shape (num_nodes,)
    """
    G = graph.to_undirected() if graph.is_directed() else graph

    if G.number_of_nodes() == 0:
        return torch.empty((0,), dtype=torch.float32)

    # If graph has no edges, core_number would still be defined as 0 for all nodes
    core = nx.core_number(G) if G.number_of_edges() > 0 else {n: 0 for n in G.nodes()}

    scores = [core[node] for node in graph.nodes()]  # preserve original node order
    return torch.tensor(scores, dtype=torch.float32).unsqueeze(1)


def compute_graph_scores(graph, alpha=0.85, weight=None):
    """
    Convenience wrapper: returns a dict of tensors (all aligned to graph.nodes()).
    """
    return {
        "degree_centrality": compute_degree_centrality(graph),
        "pagerank": compute_pagerank(graph, alpha=alpha, weight=weight),
        "kcore": compute_kcore_score(graph),
    }

def compute_node_features_hks_deg_kcore_pagerank(dataset):
    score=[]
    for graph_id in range(len(dataset)):
        graph = to_networkx(dataset[graph_id], to_undirected=True)
        hks = compute_hks(graph)
        dc=compute_degree_centrality(graph)
        kcore=compute_kcore_score(graph)
        pr=compute_pagerank(graph)
        X = torch.cat([hks, dc, kcore, pr], dim=1)
        score.append(X)
    return score




def get_thres_atom(dataset,number_threshold):
    thresh = []
    graph_list = []

    for graph_id in range(len(dataset)):
        atomic_values=get_atomic_weight(dataset[graph_id])
        graph_list.append(atomic_values)
    thresh=torch.cat(graph_list,dim=0)
    thresh = process_thresholds(thresh, number_threshold)

    return graph_list, torch.tensor(thresh)
def get_thresh(dataset,number_threshold):
    thresh = []
    graph_list = []

    for graph_id in range(len(dataset)):
        degree_centrality_values=get_degree_centrality(dataset[graph_id])
        graph_list.append(degree_centrality_values)
    thresh=torch.cat(graph_list,dim=0)
    thresh = process_thresholds(thresh, number_threshold)

    return graph_list, torch.tensor(thresh)
def get_thresh_hks(dataset,number_threshold,t_value):
    graph_list = []
    label=[]
    for graph_id in range(len(dataset)):
        graph = to_networkx(dataset[graph_id], to_undirected=True)
        hks_values=compute_hks(graph,t_value)
        graph_list.append(hks_values)
        label.append(dataset[graph_id].y)
    thresh=torch.cat(graph_list,dim=0)
    thresh = process_thresholds(thresh, number_threshold)

    return graph_list, torch.tensor(thresh),label

import numpy as np
import networkx as nx
import pyflagser
import torch
import networkx as nx
import pyflagser

def Topo_Fe_TimeSeries_MP(graph, feature1, feature2, threshold1, threshold2):
    """
    Compute Betti-0, Betti-1, num_nodes, num_edges for combinations of thresholds.
    All inputs are torch tensors.
    """
    betti_0_all = []
    betti_1_all = []
    num_nodes_all = []
    num_edges_all = []

    # edge_index is [2, num_edges]
    edge_index = graph.edge_index  # already a torch tensor

    # Ensure 1D for features and thresholds
    feature1 = feature1.view(-1)
    feature2 = feature2.view(-1)

    # Convert thresholds to 1D tensor if needed
    threshold1 = threshold1.view(-1)
    threshold2 = threshold2.view(-1)

    num_nodes_total = feature1.shape[0]

    for p in range(threshold1.size(0)):
        betti0_row = []
        betti1_row = []
        nodes_row = []
        edges_row = []

        for q in range(threshold2.size(0)):
            # Find active nodes as intersection
            idx1 = torch.where(feature1 <= threshold1[p])[0]
            idx2 = torch.where(feature2 <= threshold2[q])[0]
            n_active = torch.tensor(list(set(idx1.tolist()) & set(idx2.tolist())), dtype=torch.long)

            if n_active.numel() == 0:
                betti0_row.append(0)
                betti1_row.append(0)
                nodes_row.append(0)
                edges_row.append(0)
            else:
                # Create graph using only active nodes
                active_set = set(n_active.tolist())
                G = nx.Graph()
                G.add_nodes_from(active_set)

                # Add edges where both endpoints are active
                u = edge_index[0].tolist()
                v = edge_index[1].tolist()
                for uu, vv in zip(u, v):
                    if uu in active_set and vv in active_set:
                        G.add_edge(int(uu), int(vv))

                # Compute Betti numbers
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

    # Convert results to torch tensors for consistency
    return (torch.tensor(betti_0_all, dtype=torch.float),
            torch.tensor(betti_1_all, dtype=torch.float),
            torch.tensor(num_nodes_all, dtype=torch.float),
            torch.tensor(num_edges_all, dtype=torch.float))
def stat(acc_list, metric):
    mean = statistics.mean(acc_list)
    stdev = statistics.stdev(acc_list)
    print('Final', metric, f'using 5 fold CV: {mean:.4f} \u00B1 {stdev:.4f}%')


def print_stat(train_acc, test_acc):
    argmax = np.argmax(train_acc)
    best_result = test_acc[argmax]
    train_ac = np.max(train_acc)
    test_ac = np.max(test_acc)
    #print(f'Train accuracy = {train_ac:.4f}%,Test Accuracy = {test_ac:.4f}%\n')
    return test_ac, best_result
def apply_Zscore(MP_tensor):
# MP_tensor: (N, 4, 10, 10)
    # Compute mean and std per channel (dim=0 = graphs, dim=2,3 = grid)
    mean = MP_tensor.mean(dim=(0, 2, 3), keepdim=True)   # shape (1,4,1,1)
    std  = MP_tensor.std(dim=(0, 2, 3), keepdim=True)    # shape (1,4,1,1)

    # Apply z-score normalization per channel
    MP_tensor_z = (MP_tensor - mean) / (std + 1e-8)  # avoid div by zero
    return MP_tensor_z


def quantile_bin_1d(x, n_bins=10):
    """
    Quantile-bin a 1D array into n_bins bins: returns integers in [0, n_bins-1].
    Uses empirical quantiles; if x is constant, all bins become 0.
    """
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x.astype(np.int64)

    # Constant column -> single bin
    if np.allclose(x.min(), x.max()):
        return np.zeros_like(x, dtype=np.int64)

    # Edges at 10%,20%,...,90% for n_bins=10 (internal cut points)
    qs = np.linspace(0, 1, n_bins + 1)[1:-1]  # exclude 0 and 1
    edges = np.quantile(x, qs, method="linear")  # shape (n_bins-1,)

    # bucketize: bin i means x is between edges[i-1] and edges[i]
    # right=True makes values equal to an edge go to the higher bin
    bins = np.searchsorted(edges, x, side="right")  # 0..n_bins-1
    return bins.astype(np.int64)


def sort_nodes_lexicographically_with_quantiles(score_mat, n_bins=10, col_order=(0, 1, 2, 3), tie_break_by_node_id=True):
    """
    score_mat: (N,4) torch.Tensor or np.ndarray. Columns: [hks, deg, kcore, pagerank]
    Returns:
      sorted_mat: (N,4) same type as input (torch.Tensor if input torch, else np.ndarray)
      sort_idx: (N,) indices that sort the nodes
      binned_mat: (N,4) np.ndarray of integer bins used for sorting
    """
    is_torch = torch.is_tensor(score_mat)
    M = score_mat.detach().cpu().numpy() if is_torch else np.asarray(score_mat)
    if M.ndim != 2 or M.shape[1] != 4:
        raise ValueError(f"Expected score_mat shape (N,4). Got {M.shape}.")

    N = M.shape[0]

    # Build integer bin matrix (N,4)
    binned = np.zeros_like(M, dtype=np.int64)
    for c in range(M.shape[1]):
        binned[:, c] = quantile_bin_1d(M[:, c], n_bins=n_bins)

    # Lexicographic sort by (bin[col_order[0]], bin[col_order[1]], ...)
    # np.lexsort uses the LAST key as primary, so pass keys reversed.
    keys = [binned[:, c] for c in reversed(col_order)]

    # Optional stable tie-breaker by node id (original index)
    if tie_break_by_node_id:
        keys = keys + [np.arange(N, dtype=np.int64)]  # last key = primary, so put this first? careful

        # We want node id to be the *last* tie-breaker (least important),
        # so it should be the FIRST key passed to lexsort.
        # Because lexsort’s last key is most important, we should prepend it:
        keys = [np.arange(N, dtype=np.int64)] + [binned[:, c] for c in reversed(col_order)]

    sort_idx = np.lexsort(keys)

    # Apply
    M_sorted = M[sort_idx]
    binned_sorted = binned[sort_idx]

    if is_torch:
        return torch.as_tensor(M_sorted, dtype=score_mat.dtype, device=score_mat.device), \
               torch.as_tensor(sort_idx, dtype=torch.long, device=score_mat.device), \
               binned_sorted
    else:
        return M_sorted, sort_idx, binned_sorted

def sort_dataset_score_matrices(score_matrix_list, n_bins=10, col_order=(0, 1, 2, 3), tie_break_by_node_id=True):
    """
    score_matrix_list: list of (N_i,4) matrices
    Returns lists: sorted_mats, sort_indices, binned_mats
    """
    sorted_mats, sort_indices, binned_mats = [], [], []
    for S in score_matrix_list:
        S_sorted, idx, binned = sort_nodes_lexicographically_with_quantiles(
            S, n_bins=n_bins, col_order=col_order, tie_break_by_node_id=tie_break_by_node_id
        )
        sorted_mats.append(S_sorted)
        sort_indices.append(idx)
        binned_mats.append(binned)
    return sorted_mats, sort_indices, binned_mats
import numpy as np
import networkx as nx
import scipy.sparse as sp


def block_pool_adjacency(A_star, k):
    """
    Block-pooled adjacency image as in the paper.

    Parameters
    ----------
    A_star : (N,N) np.ndarray
        Reordered adjacency matrix A*(G).
    k : int
        Number of blocks.

    Returns
    -------
    I : (k,k) np.ndarray
        Pooled image where I[i,j] is average edge weight between blocks i and j.
    """
    A_star = np.asarray(A_star, dtype=float)
    N = A_star.shape[0]
    if A_star.shape[0] != A_star.shape[1]:
        raise ValueError("A_star must be square (N,N).")
    if k <= 0:
        raise ValueError("k must be positive.")

    # Partition indices into k consecutive blocks, equal size up to one
    blocks = np.array_split(np.arange(N), k)

    I = np.zeros((k, k), dtype=float)
    for i, Bi in enumerate(blocks):
        for j, Bj in enumerate(blocks):
            if len(Bi) == 0 or len(Bj) == 0:
                I[i, j] = 0.0
            else:
                sub = A_star[np.ix_(Bi, Bj)]
                I[i, j] = sub.mean()  # equals (1/|Bi||Bj|) sum_{u in Bi, v in Bj} A*[u,v]
    return I

def adjacency_from_sorted_order_nx(graph, sort_idx,k, to_undirected=True, include_self_loops=False, weight=None):
    """
    Create adjacency matrix consistent with your sorted node order.

    Parameters
    ----------
    G : networkx.Graph or networkx.DiGraph
        Original graph.
    sort_idx : array-like of shape (N,)
        Indices that sort nodes (the same ones you used to sort your score matrix).
        IMPORTANT: sort_idx must correspond to the order of nodes = list(G.nodes()).
    to_undirected : bool
        If True, make graph undirected before adjacency.
    as_dense : bool
        If True, return dense numpy array; otherwise return scipy sparse csr.
    weight : str or None
        Edge attribute name to use as weights. None = unweighted (1s).

    Returns
    -------
    A_sorted : (N,N) csr_matrix or ndarray
        Adjacency matrix in sorted node order.
    nodes_sorted : list
        Nodes in sorted order.
    """
    G = to_networkx(graph)
    nodes = list(G.nodes())
    sort_idx = np.asarray(sort_idx, dtype=int)
    if len(sort_idx) != len(nodes):
        raise ValueError("sort_idx length must equal number of nodes in G (in list(G.nodes()) order).")

    nodes_sorted = [nodes[i] for i in sort_idx]

    H = G.to_undirected() if (to_undirected and G.is_directed()) else G
    # Dense adjacency aligned to sorted nodes
    A_sorted = nx.to_numpy_array(H, nodelist=nodes_sorted, weight=weight, dtype=float)

    if include_self_loops:
        np.fill_diagonal(A_sorted, 1.0)

    # # Adjacency aligned to nodes_sorted
    # A = nx.to_scipy_sparse_array(H, nodelist=nodes_sorted, weight=weight, format="csr")
    #
    # if as_dense:
    #     A = A.toarray()
    I = block_pool_adjacency(A_sorted, k)

    return I,A_sorted, nodes_sorted
