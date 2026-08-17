import csv

import torch
from PIL import Image

from favit_lsda.data import FaceTransform, GroupedForgeryDataset


def test_domain_shift_transform_preserves_shape_and_range():
    transform = FaceTransform(
        64,
        crop_scale_min=0.85,
        color_jitter=0.1,
        grayscale_probability=1.0,
        blur_probability=1.0,
        degradation_probability=1.0,
        jpeg_probability=1.0,
        jpeg_quality_min=30,
    )
    tensor = transform(Image.new("RGB", (80, 72), "red"))
    assert tensor.shape == (3, 64, 64)
    assert torch.isfinite(tensor).all()
    assert -1.0 <= tensor.min() <= tensor.max() <= 1.0


def test_grouped_dataset_builds_canonical_real_fake_domains(tmp_path):
    Image.new("RGB", (20, 20), "white").save(tmp_path / "real.jpg")
    methods = ("Deepfakes", "Face2Face")
    rows = []
    for index, method in enumerate(methods):
        fake_name = f"fake_{index}.jpg"
        Image.new("RGB", (20, 20), "red").save(tmp_path / fake_name)
        rows.append(
            {"fake_path": fake_name, "real_path": "real.jpg", "method": method}
        )
    manifest = tmp_path / "pairs.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["fake_path", "real_path", "method"]
        )
        writer.writeheader()
        writer.writerows(rows)
    dataset = GroupedForgeryDataset(
        manifest, tmp_path, FaceTransform(224), forgery_methods=methods
    )
    images, labels = dataset[0]
    assert images.shape == (3, 3, 224, 224)
    assert torch.equal(labels, torch.tensor([0, 1, 2]))


def test_incomplete_domain_groups_are_reported_and_dropped(tmp_path):
    Image.new("RGB", (20, 20), "white").save(tmp_path / "real_a.jpg")
    Image.new("RGB", (20, 20), "white").save(tmp_path / "real_b.jpg")
    for name in ("a_df.jpg", "a_f2f.jpg", "b_df.jpg"):
        Image.new("RGB", (20, 20), "red").save(tmp_path / name)
    rows = [
        {"fake_path": "a_df.jpg", "real_path": "real_a.jpg", "method": "DF"},
        {"fake_path": "a_f2f.jpg", "real_path": "real_a.jpg", "method": "F2F"},
        {"fake_path": "b_df.jpg", "real_path": "real_b.jpg", "method": "DF"},
    ]
    manifest = tmp_path / "pairs.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    dataset = GroupedForgeryDataset(
        manifest, tmp_path, FaceTransform(32), forgery_methods=("DF", "F2F")
    )
    assert len(dataset) == 1
    assert dataset.dropped_incomplete_groups == 1
