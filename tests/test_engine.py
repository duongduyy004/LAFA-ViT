"""End-to-end wiring test for the training/evaluation loops.

`favit_lsda/engine.py` is the seam where dataset output order meets model input
order. Both datasets emit RGB first and the artifact tensor second, and both
loops must forward them in that order. Nothing else in the suite exercises that
contract with real datasets, a real DataLoader and a real model at once, so an
accidental argument swap between `(rgb, cnn)` and `(cnn, rgb)` would otherwise
only surface at training time.

The fixture deliberately uses a non-`rgb` artifact mode so the two tensors have
different channel widths (3 vs 6): a swap then cannot silently type-check.
"""

from __future__ import annotations

import csv
import math

import torch
from PIL import Image
from torch.utils.data import DataLoader

from favit_lsda.data import FaceTransform, FrameFaceDataset, GroupedForgeryDataset
from favit_lsda.engine import evaluate_at_level, train_one_epoch
from favit_lsda.losses import FineGrainedAdaptiveLoss
from favit_lsda.model import create_favit_lsda

ARTIFACT_MODE = "rgb_srm"
CNN_IN_CHANNELS = 6
METHODS = ("DF", "F2F")


def _write_fixture(tmp_path):
    """Two complete LSDA groups plus a two-class frame manifest."""
    reals = ("real_a.jpg", "real_b.jpg")
    for name in reals:
        Image.new("RGB", (24, 24), "white").save(tmp_path / name)
    pair_rows = []
    for real in reals:
        for method in METHODS:
            fake = f"{real[:-4]}_{method}.jpg"
            Image.new("RGB", (24, 24), "red").save(tmp_path / fake)
            pair_rows.append({"fake_path": fake, "real_path": real, "method": method})
    pairs = tmp_path / "pairs.csv"
    with pairs.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["fake_path", "real_path", "method"])
        writer.writeheader()
        writer.writerows(pair_rows)

    frames = tmp_path / "frames.csv"
    with frames.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "label", "video_id"])
        writer.writeheader()
        writer.writerows(
            [
                {"path": "real_a.jpg", "label": "0", "video_id": "v_real"},
                {"path": "real_b.jpg", "label": "0", "video_id": "v_real"},
                {"path": "real_a_DF.jpg", "label": "1", "video_id": "v_fake"},
                {"path": "real_b_F2F.jpg", "label": "1", "video_id": "v_fake"},
            ]
        )
    return pairs, frames


def test_engine_loops_match_dataset_output_to_model_input_order(tmp_path):
    pairs, frames = _write_fixture(tmp_path)
    device = torch.device("cpu")
    transform = FaceTransform(224, artifact_mode=ARTIFACT_MODE)
    model = create_favit_lsda(
        model_name="vit_tiny_patch16_224",
        pretrained=False,
        forgery_methods=METHODS,
        train_backbone_norms=False,
        train_cls_token=False,
        artifact_mode=ARTIFACT_MODE,
        cnn_in_channels=CNN_IN_CHANNELS,
    ).to(device)

    train_loader = DataLoader(
        GroupedForgeryDataset(pairs, tmp_path, transform, METHODS),
        batch_size=2,
        shuffle=False,
        num_workers=0,
        drop_last=True,
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1e-4,
    )
    train_metrics = train_one_epoch(
        model,
        train_loader,
        optimizer,
        FineGrainedAdaptiveLoss(),
        {"binary": 1.0, "domain": 1.0, "invariance": 1.0, "distill": 1.0, "fal": 1.0},
        device,
    )
    assert math.isfinite(train_metrics["loss"])
    for key in ("binary", "domain", "invariance", "distill", "fal", "binary_accuracy"):
        assert math.isfinite(train_metrics[key])

    eval_loader = DataLoader(
        FrameFaceDataset(frames, tmp_path, transform),
        batch_size=2,
        shuffle=False,
        num_workers=0,
    )
    metrics = evaluate_at_level(model, eval_loader, device, level="frame")
    assert metrics["level"] == "frame"
    assert metrics["num_frames"] == 4
    assert 0.0 <= metrics["auc"] <= 1.0
    assert math.isfinite(metrics["accuracy"])
