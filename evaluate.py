from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from favit_lsda.config import build_model_from_config, load_config, resolve_device
from favit_lsda.data import FaceTransform, FrameFaceDataset
from favit_lsda.engine import evaluate_video_level


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate FA-ViT + LSDA")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = resolve_device(args.device or config.get("device", "cuda"))
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if checkpoint.get("architecture") not in (None, "favit_lsda"):
        raise ValueError("checkpoint is not a FA-ViT + LSDA checkpoint")
    model_config = checkpoint.get("config", config)["model"]
    model = build_model_from_config(model_config, pretrained=False).to(device)
    incompatible = model.load_state_dict(checkpoint["model"], strict=False)
    allowed_missing_prefixes = (
        "latent_augmenter.comprehensive_scale",
        "student_domain_classifier.",
    )
    invalid_missing = [
        key
        for key in incompatible.missing_keys
        if not key.startswith(allowed_missing_prefixes)
    ]
    if invalid_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            "checkpoint/model mismatch: "
            f"missing={invalid_missing[:5]} "
            f"unexpected={incompatible.unexpected_keys[:5]}"
        )
    if incompatible.missing_keys:
        print("note: evaluating a legacy checkpoint with inference-neutral new layers")
    data_config = config["data"]
    dataset = FrameFaceDataset(
        args.manifest or data_config["celebdf_test_frames"],
        data_config["root"],
        FaceTransform(int(data_config.get("image_size", 224))),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config["train"].get("eval_image_batch_size", 32)),
        shuffle=False,
        num_workers=int(data_config.get("num_workers", 8)),
        pin_memory=device.type == "cuda",
    )
    print(json.dumps(evaluate_video_level(model, loader, device), indent=2))


if __name__ == "__main__":
    main()
