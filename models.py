
import torch
import torch.nn as nn

from torch.nn import Linear, Embedding
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import GATv2Conv
import torch.nn.functional as F
from torch_geometric.nn import (
    GCNConv, SAGEConv, GATConv, GINConv, TAGConv, ChebConv, ARMAConv,
    TransformerConv, GPSConv, global_mean_pool
)

class CNNTransformer(nn.Module):
    def __init__(self, num_classes, cnn_channels, d_model, drop_out, nhead=4, num_layers=2):
        super().__init__()

        # CNN feature extractor
        self.cnn = nn.Sequential(
            nn.Conv2d(4, cnn_channels, kernel_size=3, padding=1),  # (B, 64, 10, 10)
            nn.BatchNorm2d(cnn_channels),
            nn.ReLU(),
            nn.Dropout(p=drop_out),  # Dropout after ReLU for regularization
            nn.Conv2d(cnn_channels, cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(cnn_channels),
            nn.ReLU(),
            nn.Dropout(p=drop_out)  # Dropout after second conv
        )

        # Flatten spatial grid into sequence of patches
        self.flatten_patches = lambda x: x.flatten(2).transpose(1, 2)  # (B, N=100, D=64)

        # Linear projection to d_model
        self.embedding = nn.Linear(cnn_channels, d_model)
        self.embedding_dropout = nn.Dropout(p=drop_out)  # Dropout after embedding

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        # Transformer encoder (with dropout inside encoder layer)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True,
            dropout=drop_out  # Applies dropout to attention and feed-forward layers
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Dropout before classifier
        self.final_dropout = nn.Dropout(p=drop_out)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x: (B, 4, 20, 10)
        x = self.cnn(x)  # (B, 64, 20, 10)
        x = self.flatten_patches(x)  # (B, 200, 64)
        x = self.embedding(x)  # (B, 200, d_model)
        x = self.embedding_dropout(x)  # Apply dropout to embedding

        # Add CLS token
        B = x.size(0)
        cls_token = self.cls_token.expand(B, -1, -1)  # (B, 1, d_model)
        x = torch.cat((cls_token, x), dim=1)  # (B, 201, d_model)

        x = self.transformer(x)  # (B, 201, d_model)
        cls_output = x[:, 0]  # (B, d_model)
        cls_output = self.final_dropout(cls_output)  # Dropout before classifier

        return self.classifier(cls_output)
class CNNTransformer_image(nn.Module):
    def __init__(
        self,
        num_classes: int,
        in_channels: int,          # <-- NEW: any input channels
        cnn_channels: int,
        d_model: int,
        drop_out: float,
        nhead: int = 4,
        num_layers: int = 2,
    ):
        super().__init__()

        # CNN feature extractor
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(cnn_channels),
            nn.ReLU(),
            nn.Dropout(p=drop_out),

            nn.Conv2d(cnn_channels, cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(cnn_channels),
            nn.ReLU(),
            nn.Dropout(p=drop_out),
        )

        # Linear projection to transformer dim
        self.embedding = nn.Linear(cnn_channels, d_model)
        self.embedding_dropout = nn.Dropout(p=drop_out)

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True,
            dropout=drop_out,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classifier head
        self.final_dropout = nn.Dropout(p=drop_out)
        self.classifier = nn.Linear(d_model, num_classes)

        self._init_parameters()

    def _init_parameters(self):
        # Optional but helpful initialization
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

    @staticmethod
    def flatten_patches(x: torch.Tensor) -> torch.Tensor:
        """
        (B, C, H, W) -> (B, H*W, C)
        """
        return x.flatten(2).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, in_channels, H, W)  where in_channels can be any positive int
        """
        x = self.cnn(x)                         # (B, cnn_channels, H, W)
        x = self.flatten_patches(x)             # (B, H*W, cnn_channels)
        x = self.embedding(x)                   # (B, H*W, d_model)
        x = self.embedding_dropout(x)

        # Add CLS token
        B = x.size(0)
        cls_token = self.cls_token.expand(B, 1, -1)   # (B, 1, d_model)
        x = torch.cat([cls_token, x], dim=1)          # (B, 1+H*W, d_model)

        x = self.transformer(x)                 # (B, 1+H*W, d_model)
        cls_output = self.final_dropout(x[:, 0])
        return self.classifier(cls_output)


class CNNTransformer_image1(nn.Module):
    def __init__(self, num_classes, in_channels, cnn_channels, d_model, drop_out, nhead=4, num_layers=2):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(cnn_channels),
            nn.ReLU(),
            nn.Dropout(p=drop_out),
            nn.Conv2d(cnn_channels, cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(cnn_channels),
            nn.ReLU(),
            nn.Dropout(p=drop_out),
        )
        self.embedding = nn.Linear(cnn_channels, d_model)
        self.embedding_dropout = nn.Dropout(p=drop_out)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True, dropout=drop_out)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.final_dropout = nn.Dropout(p=drop_out)
        self.classifier = nn.Linear(d_model, num_classes)
        self._init_parameters()

    def _init_parameters(self):
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

    @staticmethod
    def flatten_patches(x):
        return x.flatten(2).transpose(1, 2)

    def forward(self, x, return_feats=False):
        """
        Added return_feats argument to switch between full classification or just extraction
        """
        x = self.cnn(x)
        x = self.flatten_patches(x)
        x = self.embedding(x)
        x = self.embedding_dropout(x)

        B = x.size(0)
        cls_token = self.cls_token.expand(B, 1, -1)
        x = torch.cat([cls_token, x], dim=1)

        x = self.transformer(x)

        # Extract the CLS token representation
        cls_output = self.final_dropout(x[:, 0])

        # If we are doing fusion, we just want the vector, not the class prediction
        if return_feats:
            return cls_output

        return self.classifier(cls_output)


class LateFusionModel(nn.Module):
    def __init__(self, num_classes, cnn_channels, d_model, drop_out, nhead, num_layers):
        super().__init__()

        # Branch 1: Handles G_image (1 Channel)
        self.image_branch = CNNTransformer_image1(
            num_classes=num_classes,
            in_channels=1,
            cnn_channels=cnn_channels,
            d_model=d_model,
            drop_out=drop_out,
            nhead=nhead,
            num_layers=num_layers
        )

        # Branch 2: Handles Toposurface (4 Channels)
        self.topo_branch = CNNTransformer_image1(
            num_classes=num_classes,
            in_channels=4,
            cnn_channels=cnn_channels,
            d_model=d_model,
            drop_out=drop_out,
            nhead=nhead,
            num_layers=num_layers
        )

        # Fusion Head
        # We concatenate two vectors of size 'd_model', so input is d_model * 2
        self.fusion_classifier = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(drop_out),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x_img, x_topo):
        # 1. Get features from Image Branch
        feat_img = self.image_branch(x_img, return_feats=True)  # Shape: (B, d_model)

        # 2. Get features from Topo Branch
        feat_topo = self.topo_branch(x_topo, return_feats=True)  # Shape: (B, d_model)

        # 3. Concatenate (Late Fusion)
        combined = torch.cat([feat_img, feat_topo], dim=1)  # Shape: (B, d_model * 2)

        # 4. Classify
        out = self.fusion_classifier(combined)
        return out

# Transformer-based GNNs
class GraphTransformer(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes, heads=4):
        super().__init__()
        self.conv1 = TransformerConv(in_channels, hidden_channels, heads=heads, concat=False)
        self.conv2 = TransformerConv(hidden_channels, hidden_channels, heads=heads, concat=False)
        self.lin = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)
        return self.lin(x)


class GPSModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes):
        super().__init__()
        self.input_proj = torch.nn.Linear(in_channels, hidden_channels)

        self.conv1 = GPSConv(
            channels=hidden_channels,
            conv=GCNConv(hidden_channels, hidden_channels),
            heads=2
        )
        self.conv2 = GPSConv(
            channels=hidden_channels,
            conv=GCNConv(hidden_channels, hidden_channels),
            heads=2
        )
        self.lin = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        x = self.input_proj(x)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)
        return self.lin(x)



class Graphormer(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes, num_layers=2, heads=4, max_degree=10):
        super().__init__()
        self.input_proj = Linear(in_channels, hidden_channels)

        # Structural encodings (e.g., node degree encoding as Graphormer does)
        self.degree_emb = Embedding(max_degree + 1, hidden_channels)

        self.layers = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                GATv2Conv(hidden_channels, hidden_channels // heads, heads=heads, concat=True)
            )

        self.norms = torch.nn.ModuleList([torch.nn.LayerNorm(hidden_channels) for _ in range(num_layers)])

        self.classifier = Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch, deg=None):
        x = self.input_proj(x)

        if deg is not None:
            deg = deg.clamp(max=self.degree_emb.num_embeddings - 1)
            x = x + self.degree_emb(deg)

        for conv, norm in zip(self.layers, self.norms):
            residual = x
            x = conv(x, edge_index)
            x = F.relu(x)
            x = norm(x + residual)

        x = global_mean_pool(x, batch)
        return self.classifier(x)


#
# class CNNTransformer(nn.Module):
#     def __init__(self, num_classes, cnn_channels, d_model,drop_out, nhead=4, num_layers=2):
#         super().__init__()
#         # CNN feature extractor
#         self.cnn = nn.Sequential(
#             nn.Conv2d(4, cnn_channels, kernel_size=3, padding=1),  # (B, 64, 20, 10)
#             nn.BatchNorm2d(cnn_channels),
#             nn.ReLU(),
#             nn.Conv2d(cnn_channels, cnn_channels, kernel_size=3, padding=1),  # (B, 64, 20, 10)
#             nn.BatchNorm2d(cnn_channels),
#             nn.Dropout(p=drop_out),
#             nn.ReLU()
#         )
#
#         # Flatten spatial grid into sequence of patches
#         self.flatten_patches = lambda x: x.flatten(2).transpose(1, 2)  # (B, N=200, D=64)
#
#         # Linear projection to d_model
#         self.embedding = nn.Linear(cnn_channels, d_model)
#
#         # CLS token
#         self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
#
#         # Transformer encoder
#         encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
#         self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
#
#         # Classifier
#         self.classifier = nn.Linear(d_model, num_classes)
#
#     def forward(self, x):
#         # x: (B, 4, 20, 10)
#         x = self.cnn(x)  # (B, 64, 20, 10)
#         x = self.flatten_patches(x)  # (B, 200, 64)
#         x = self.embedding(x)  # (B, 200, d_model)
#
#         # Add CLS token
#         B = x.size(0)
#         cls_token = self.cls_token.expand(B, -1, -1)  # (B, 1, d_model)
#         x = torch.cat((cls_token, x), dim=1)  # (B, 201, d_model)
#
#         x = self.transformer(x)  # (B, 201, d_model)
#         cls_output = x[:, 0]  # (B, d_model)
#         return self.classifier(cls_output)

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNTransformerEncoder(nn.Module):
    """
    CNN + Transformer encoder that returns a CLS embedding.
    Works with any input channel count.
    """
    def __init__(
        self,
        in_channels: int,
        cnn_channels: int,
        d_model: int,
        drop_out: float,
        nhead: int = 4,
        num_layers: int = 2,
    ):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(cnn_channels),
            nn.ReLU(),
            nn.Dropout(p=drop_out),

            nn.Conv2d(cnn_channels, cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(cnn_channels),
            nn.ReLU(),
            nn.Dropout(p=drop_out),
        )

        self.embedding = nn.Linear(cnn_channels, d_model)
        self.embedding_dropout = nn.Dropout(p=drop_out)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True,
            dropout=drop_out,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.final_dropout = nn.Dropout(p=drop_out)

        self._init_parameters()

    def _init_parameters(self):
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

    @staticmethod
    def flatten_patches(x: torch.Tensor) -> torch.Tensor:
        # (B, C, H, W) -> (B, H*W, C)
        return x.flatten(2).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W)
        returns: (B, d_model) CLS embedding
        """
        x = self.cnn(x)                          # (B, cnn_channels, H, W)
        x = self.flatten_patches(x)              # (B, H*W, cnn_channels)
        x = self.embedding(x)                    # (B, H*W, d_model)
        x = self.embedding_dropout(x)

        B = x.size(0)
        cls = self.cls_token.expand(B, 1, -1)    # (B, 1, d_model)
        x = torch.cat([cls, x], dim=1)           # (B, 1+H*W, d_model)

        x = self.transformer(x)                  # (B, 1+H*W, d_model)
        z = self.final_dropout(x[:, 0])          # (B, d_model)
        return z


class ProjectionHead(nn.Module):
    """
    Small MLP projection head for contrastive learning.
    """
    def __init__(self, in_dim: int, proj_dim: int = 128, drop_out: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Dropout(p=drop_out),
            nn.Linear(in_dim, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    """
    Symmetric NT-Xent (InfoNCE) for two views.
    z1, z2: (B, D) projected embeddings
    """
    # Normalize
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    # Similarity matrix: (B, B)
    logits = (z1 @ z2.t()) / temperature

    # Targets are diagonal (positive pairs)
    targets = torch.arange(z1.size(0), device=z1.device)

    loss_12 = F.cross_entropy(logits, targets)
    loss_21 = F.cross_entropy(logits.t(), targets)
    return 0.5 * (loss_12 + loss_21)


class TwoViewContrastiveClassifier(nn.Module):
    """
    Two encoders (one per view) + contrastive projection heads + fused classifier.
    """
    def __init__(
        self,
        num_classes: int,
        in_channels_view1: int,
        in_channels_view2: int,
        cnn_channels: int,
        d_model: int,
        drop_out: float,
        nhead: int = 4,
        num_layers: int = 2,
        proj_dim: int = 128,
        share_encoder: bool = False,   # if True, same encoder weights used for both views (requires same in_channels)
        fuse: str = "concat",          # "concat" or "sum" or "mean"
    ):
        super().__init__()
        self.fuse = fuse

        if share_encoder:
            if in_channels_view1 != in_channels_view2:
                raise ValueError("share_encoder=True requires same channel count for both views.")
            encoder = CNNTransformerEncoder(
                in_channels=in_channels_view1,
                cnn_channels=cnn_channels,
                d_model=d_model,
                drop_out=drop_out,
                nhead=nhead,
                num_layers=num_layers,
            )
            self.encoder1 = encoder
            self.encoder2 = encoder
        else:
            self.encoder1 = CNNTransformerEncoder(
                in_channels=in_channels_view1,
                cnn_channels=cnn_channels,
                d_model=d_model,
                drop_out=drop_out,
                nhead=nhead,
                num_layers=num_layers,
            )
            self.encoder2 = CNNTransformerEncoder(
                in_channels=in_channels_view2,
                cnn_channels=cnn_channels,
                d_model=d_model,
                drop_out=drop_out,
                nhead=nhead,
                num_layers=num_layers,
            )

        self.proj1 = ProjectionHead(d_model, proj_dim=proj_dim, drop_out=drop_out)
        self.proj2 = ProjectionHead(d_model, proj_dim=proj_dim, drop_out=drop_out)

        if fuse == "concat":
            clf_in = 2 * d_model
        elif fuse in ("sum", "mean"):
            clf_in = d_model
        else:
            raise ValueError("fuse must be one of: 'concat', 'sum', 'mean'.")

        self.classifier = nn.Sequential(
            nn.Dropout(p=drop_out),
            nn.Linear(clf_in, num_classes)
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        """
        Returns:
          logits: (B, num_classes)
          h1, h2: (B, d_model) CLS embeddings
          z1, z2: (B, proj_dim) projected embeddings
        """
        h1 = self.encoder1(x1)
        h2 = self.encoder2(x2)

        z1 = self.proj1(h1)
        z2 = self.proj2(h2)

        if self.fuse == "concat":
            h = torch.cat([h1, h2], dim=1)
        elif self.fuse == "sum":
            h = h1 + h2
        else:  # mean
            h = 0.5 * (h1 + h2)

        logits = self.classifier(h)
        return logits, h1, h2, z1, z2
