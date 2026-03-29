import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import Linear, Embedding
from torch_geometric.nn import global_mean_pool, GATv2Conv
from torch_geometric.nn import (
    GCNConv, SAGEConv, GATConv, GINConv, TAGConv, ChebConv, ARMAConv,
    TransformerConv, GPSConv,
)


# -----------------------------------------------------------------------
# Shared building blocks
# -----------------------------------------------------------------------

class CNNTransformerEncoder(nn.Module):
    """
    CNN + Transformer encoder that returns a single CLS embedding.
    Works with any input channel count and any spatial grid size.
    Used as the shared backbone for all three views.
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

        self.embedding         = nn.Linear(cnn_channels, d_model)
        self.embedding_dropout = nn.Dropout(p=drop_out)
        self.cls_token         = nn.Parameter(torch.zeros(1, 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True,
            dropout=drop_out,
            norm_first=True,   # pre-LN: more stable for small d_model
        )
        self.transformer   = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.final_dropout = nn.Dropout(p=drop_out)

        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

    @staticmethod
    def flatten_patches(x: torch.Tensor) -> torch.Tensor:
        # (B, C, H, W) -> (B, H*W, C)
        return x.flatten(2).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W)  ->  (B, d_model) CLS embedding"""
        x   = self.cnn(x)                         # (B, cnn_channels, H, W)
        x   = self.flatten_patches(x)             # (B, H*W, cnn_channels)
        x   = self.embedding(x)                   # (B, H*W, d_model)
        x   = self.embedding_dropout(x)
        B   = x.size(0)
        cls = self.cls_token.expand(B, 1, -1)
        x   = torch.cat([cls, x], dim=1)          # (B, 1+H*W, d_model)
        x   = self.transformer(x)
        return self.final_dropout(x[:, 0])        # (B, d_model)


class ProjectionHead(nn.Module):
    """
    MLP projection head for contrastive learning.
    Output is L2-normalised so NT-Xent receives unit-norm embeddings directly.
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
        return F.normalize(self.net(x), dim=1)   # unit-norm output


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor,
                 temperature: float = 0.2) -> torch.Tensor:
    """
    Symmetric NT-Xent (InfoNCE) loss for a positive pair (z1, z2).
    z1, z2 are expected to already be L2-normalised (ProjectionHead does this).
    """
    logits   = (z1 @ z2.t()) / temperature
    targets  = torch.arange(z1.size(0), device=z1.device)
    loss_12  = F.cross_entropy(logits,   targets)
    loss_21  = F.cross_entropy(logits.t(), targets)
    return 0.5 * (loss_12 + loss_21)


# -----------------------------------------------------------------------
# Two-view model (unchanged — used for TU datasets)
# -----------------------------------------------------------------------

class TwoViewContrastiveClassifier(nn.Module):
    """
    Two encoders (GraphGrid + TopoGrid) with contrastive alignment.
    Used for TU datasets where node features are absent or uninformative.
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
        share_encoder: bool = False,
        fuse: str = "concat",
    ):
        super().__init__()
        self.fuse = fuse

        if share_encoder:
            if in_channels_view1 != in_channels_view2:
                raise ValueError("share_encoder=True requires same channel count.")
            enc = CNNTransformerEncoder(in_channels_view1, cnn_channels,
                                        d_model, drop_out, nhead, num_layers)
            self.encoder1 = self.encoder2 = enc
        else:
            self.encoder1 = CNNTransformerEncoder(
                in_channels_view1, cnn_channels, d_model, drop_out, nhead, num_layers)
            self.encoder2 = CNNTransformerEncoder(
                in_channels_view2, cnn_channels, d_model, drop_out, nhead, num_layers)

        self.proj1 = ProjectionHead(d_model, proj_dim, drop_out)
        self.proj2 = ProjectionHead(d_model, proj_dim, drop_out)

        clf_in = 2 * d_model if fuse == "concat" else d_model
        self.classifier = nn.Sequential(
            nn.Dropout(p=drop_out),
            nn.Linear(clf_in, num_classes),
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        h1 = self.encoder1(x1)
        h2 = self.encoder2(x2)
        z1 = self.proj1(h1)
        z2 = self.proj2(h2)

        if self.fuse == "concat":
            h = torch.cat([h1, h2], dim=1)
        elif self.fuse == "sum":
            h = h1 + h2
        else:
            h = 0.5 * (h1 + h2)

        return self.classifier(h), h1, h2, z1, z2


# -----------------------------------------------------------------------
# Three-view model — GraphGrid + TopoGrid + NodeFeatImage
# -----------------------------------------------------------------------

class ThreeViewContrastiveClassifier(nn.Module):
    """
    Three-view model for datasets with rich node features (e.g. Peptides-func,
    ogbg-molhiv, ogbg-molbace).

    Views
    -----
    view1 : GraphGrid    (B, 1,       k,  k ) — structural adjacency density
    view2 : TopoGrid     (B, 4,       10, 10) — multipersistence surfaces
    view3 : NodeFeatImg  (B, F,       k,  k ) — block-mean node feature image

    Each view is encoded by its own CNNTransformerEncoder producing a
    d_model-dimensional CLS embedding.  The three embeddings are fused
    (concat by default → 3*d_model) and passed to a linear classifier.

    Contrastive alignment is applied to all three pairwise combinations:
      L_align = NT-Xent(z1,z2) + NT-Xent(z1,z3) + NT-Xent(z2,z3)

    This encourages each view to produce representations consistent with
    the other two, improving robustness beyond standard pairwise alignment.
    """
    def __init__(
        self,
        num_classes: int,
        in_channels_view1: int,        # GraphGrid channels  (usually 1)
        in_channels_view2: int,        # TopoGrid channels   (usually 4)
        in_channels_view3: int,        # NodeFeatImg channels (= F, e.g. 4)
        cnn_channels: int,
        d_model: int,
        drop_out: float,
        nhead: int = 4,
        num_layers: int = 2,
        proj_dim: int = 128,
        fuse: str = "concat",          # "concat" | "sum" | "mean"
    ):
        super().__init__()
        self.fuse = fuse

        self.encoder1 = CNNTransformerEncoder(
            in_channels_view1, cnn_channels, d_model, drop_out, nhead, num_layers)
        self.encoder2 = CNNTransformerEncoder(
            in_channels_view2, cnn_channels, d_model, drop_out, nhead, num_layers)
        self.encoder3 = CNNTransformerEncoder(
            in_channels_view3, cnn_channels, d_model, drop_out, nhead, num_layers)

        self.proj1 = ProjectionHead(d_model, proj_dim, drop_out)
        self.proj2 = ProjectionHead(d_model, proj_dim, drop_out)
        self.proj3 = ProjectionHead(d_model, proj_dim, drop_out)

        if fuse == "concat":
            clf_in = 3 * d_model
        elif fuse in ("sum", "mean"):
            clf_in = d_model
        else:
            raise ValueError("fuse must be 'concat', 'sum', or 'mean'.")

        self.classifier = nn.Sequential(
            nn.Dropout(p=drop_out),
            nn.Linear(clf_in, num_classes),
        )

    def forward(
        self,
        x1: torch.Tensor,   # (B, C1, H, W)  GraphGrid
        x2: torch.Tensor,   # (B, C2, H, W)  TopoGrid
        x3: torch.Tensor,   # (B, C3, H, W)  NodeFeatImg
    ):
        """
        Returns
        -------
        logits : (B, num_classes)
        h1, h2, h3 : (B, d_model) CLS embeddings
        z1, z2, z3 : (B, proj_dim) L2-normalised projection embeddings
        """
        h1 = self.encoder1(x1)
        h2 = self.encoder2(x2)
        h3 = self.encoder3(x3)

        z1 = self.proj1(h1)
        z2 = self.proj2(h2)
        z3 = self.proj3(h3)

        if self.fuse == "concat":
            h = torch.cat([h1, h2, h3], dim=1)   # (B, 3*d_model)
        elif self.fuse == "sum":
            h = h1 + h2 + h3
        else:
            h = (h1 + h2 + h3) / 3.0

        return self.classifier(h), h1, h2, h3, z1, z2, z3


# -----------------------------------------------------------------------
# Legacy / unused model classes (kept for backward compatibility)
# -----------------------------------------------------------------------

class CNNTransformer(nn.Module):
    """Original single-view classifier — kept for backward compatibility."""
    def __init__(self, num_classes, cnn_channels, d_model, drop_out, nhead=4, num_layers=2):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(4, cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(cnn_channels), nn.ReLU(), nn.Dropout(p=drop_out),
            nn.Conv2d(cnn_channels, cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(cnn_channels), nn.ReLU(), nn.Dropout(p=drop_out),
        )
        self.flatten_patches   = lambda x: x.flatten(2).transpose(1, 2)
        self.embedding         = nn.Linear(cnn_channels, d_model)
        self.embedding_dropout = nn.Dropout(p=drop_out)
        self.cls_token         = nn.Parameter(torch.zeros(1, 1, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True,
            dropout=drop_out, norm_first=True,
        )
        self.transformer   = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.final_dropout = nn.Dropout(p=drop_out)
        self.classifier    = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x   = self.cnn(x)
        x   = self.flatten_patches(x)
        x   = self.embedding_dropout(self.embedding(x))
        B   = x.size(0)
        x   = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1)
        x   = self.transformer(x)
        return self.classifier(self.final_dropout(x[:, 0]))
