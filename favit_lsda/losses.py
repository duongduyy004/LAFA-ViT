from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def balanced_binary_cross_entropy(
    logits: Tensor, labels: Tensor, label_smoothing: float = 0.0
) -> Tensor:
    """Average real and fake losses equally regardless of grouped class ratio."""

    if logits.ndim != 2 or logits.shape[0] != labels.numel():
        raise ValueError("logits and labels have incompatible shapes")
    flattened_labels = labels.reshape(-1)
    per_sample = F.cross_entropy(
        logits,
        flattened_labels,
        reduction="none",
        label_smoothing=float(label_smoothing),
    )
    class_losses = [
        per_sample[flattened_labels == label].mean()
        for label in (0, 1)
        if torch.any(flattened_labels == label)
    ]
    if len(class_losses) != 2:
        raise ValueError("balanced binary loss requires both real and fake samples")
    return torch.stack(class_losses).mean()


class FineGrainedAdaptiveLoss(nn.Module):
    """FA-ViT FAL for aligned genuine/manipulated features."""

    def __init__(self, scale: float = 24.0, margin: float = 0.25) -> None:
        super().__init__()
        self.scale = float(scale)
        self.margin = float(margin)

    def forward(
        self,
        real_prototype: Tensor,
        real_features: Tensor,
        fake_features: Tensor,
    ) -> Tensor:
        if real_features.shape != fake_features.shape:
            raise ValueError("real_features and fake_features must have the same shape")
        if real_prototype.ndim == 1:
            real_prototype = real_prototype.unsqueeze(0).expand_as(real_features)
        if real_prototype.shape != real_features.shape:
            raise ValueError("prototype must be [D] or match the feature shape")
        positive = F.cosine_similarity(real_prototype, real_features, dim=-1)
        negative = F.cosine_similarity(real_prototype, fake_features, dim=-1)
        alpha_p = F.relu(1.0 + self.margin - positive)
        alpha_n = F.relu(self.margin + negative)
        logit_p = self.scale * alpha_p * (positive - (1.0 - self.margin))
        logit_n = self.scale * alpha_n * (negative - self.margin)
        return F.softplus(logit_n - logit_p).mean()
