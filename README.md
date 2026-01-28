# G2Image
From Nodes to Pixels: Topology-Guided Two-View Graph Imaging

G2Image is a deep learning framework that transforms graph-structured data into grid-based image representations. It leverages a dual-view approach—combining structural information (via Adjacency Grid Images) and topological features (via Multipersistence Toposurfaces)—to classify graphs using a CNN-Transformer architecture with Supervised Contrastive Learning.

### Key Features

- Graph-to-Image Conversion: Converts irregular graphs into fixed-size grid representations (GraphGrid).

- Topological Feature Extraction: Generates "Toposurfaces" using Multipersistence (Betti-0, Betti-1, Node/Edge densities) over varying filtration thresholds.

- Dual-View Architecture: A two-branch model that processes structural and topological views independently before fusion.

- Supervised Contrastive Learning: Uses contrastive loss to learn robust representations by pulling similar graphs together in the embedding space.

- Flexible Fusion: Supports Late Fusion, Early Concatenation (5-channel), and Multi-Task Learning.

### Methodology

The Pipeline
1. Input Graph: Takes a raw graph (nodes, edges, features).

2. View 1: GraphGrid: Nodes are sorted by centrality (PageRank, K-Core, etc.) and mapped to a 2D grid adjacency image.

3. View 2: Toposurface: A 4-channel heatmap representing topological features (Betti numbers, node/edge counts) across different filtration thresholds (Degree & Heat Kernel Signature).

4. Encoder: A CNN-Transformer extracts features from both views.

5. Contrastive Projection: Embeddings are projected to a latent space to calculate Supervised Contrastive Loss.

6. Fusion & Classification: Features are fused (concatenated) and passed to a classifier for the final prediction.

### Results and Efficiency

G2Image achieves State of the Art performance across different datasets and domains while being computationally efficient. For results and comparison to baselines, refer to Table 1 in our paper.

### Requirements

This project is built using Python 3.8+ and relies on the following major libraries. It is recommended to use a virtual environment or Conda environment to avoid conflicts.

| Package          | Version     | Description                                                                 |
|------------------|-------------|-----------------------------------------------------------------------------|
| python           | >= 3.8      | Core programming language                                                   |
| torch            | >= 1.12.0   | Deep learning framework (ensure CUDA support if using GPU)                  |
| torch-geometric  | >= 2.3.0    | Graph Neural Network primitives and data loading                            |
| networkx         | >= 2.8      | Graph creation, manipulation, and feature extraction                        |
| numpy            | >= 1.21.0   | Numerical computing and array manipulation                                  |
| scikit-learn     | >= 1.0      | Metrics (accuracy, ROC-AUC) and data splitting (K-Fold)                     |
| tqdm             | >= 4.64.0   | Progress bars for training loops                                            |
| matplotlib       | >= 3.5.0    | (Optional) For visualizing graph grids and toposurfaces                     |


