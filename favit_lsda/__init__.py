"""FA-ViT with latent-space augmentation for generalized deepfake detection."""

from .losses import FineGrainedAdaptiveLoss
from .lsda import LatentSpaceAugmenter
from .model import ForgeryAwareLSDAViT, create_favit_lsda

__all__ = [
    "FineGrainedAdaptiveLoss",
    "LatentSpaceAugmenter",
    "ForgeryAwareLSDAViT",
    "create_favit_lsda",
]
