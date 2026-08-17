from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("configuration root must be a mapping")
    return config


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(requested)


def build_model_from_config(model_config: dict[str, Any], pretrained: bool | None = None):
    from .model import create_favit_lsda

    return create_favit_lsda(
        model_name=model_config["backbone"],
        pretrained=model_config.get("pretrained", True) if pretrained is None else pretrained,
        num_classes=model_config.get("num_classes", 2),
        forgery_methods=model_config.get(
            "forgery_methods",
            ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"],
        ),
        gam_reduction=model_config.get("gam_reduction", 2),
        inject_layers=model_config.get("inject_layers", [0, 3, 6]),
        train_backbone_norms=model_config.get("train_backbone_norms", True),
        train_cls_token=model_config.get("train_cls_token", True),
        latent_transforms=model_config.get(
            "latent_transforms",
            [
                "hard_interpolation",
                "centrifugal",
                "gaussian",
                "rotation",
                "difference",
            ],
        ),
        max_rotation_degrees=model_config.get("max_rotation_degrees", 30.0),
        latent_noise_std=model_config.get("latent_noise_std", 1.0),
        mixup_concentration_min=model_config.get("mixup_concentration_min", 0.5),
        mixup_concentration_max=model_config.get("mixup_concentration_max", 2.0),
        feature_dropout=model_config.get("feature_dropout", 0.0),
        unfreeze_last_blocks=model_config.get("unfreeze_last_blocks", 0),
        domain_adversarial_strength=model_config.get(
            "domain_adversarial_strength", 1.0
        ),
    )
