import csv
import random

import pytest
import torch
from PIL import Image

from favit_lsda import data
from favit_lsda.data import (
    FaceTransform,
    FrameFaceDataset,
    GroupedForgeryDataset,
    artifact_channels,
    build_cnn_input,
)


@pytest.mark.parametrize(
    ("mode", "channels"),
    [
        ("rgb", 3), ("rgb_srm", 6), ("rgb_fft", 6),
        ("rgb_wavelet", 6), ("rgb_srm_fft", 9),
        ("rgb_srm_wavelet", 9),
    ],
)
def test_artifact_modes_return_finite_normalized_cnn_inputs(mode, channels):
    rgb = torch.linspace(-1, 1, 3 * 19 * 23).reshape(3, 19, 23)
    cnn = build_cnn_input(rgb, mode)
    assert artifact_channels(mode) == channels
    assert cnn.shape == (channels, 19, 23)
    assert torch.equal(cnn[:3], rgb)
    assert torch.isfinite(cnn).all()
    # "rgb" derives no extra channels, so cnn[3:] is empty and has no min/max.
    if channels > 3:
        assert -1.0 <= cnn[3:].min() <= cnn[3:].max() <= 1.0


@pytest.mark.parametrize("mode", ["rgb_srm", "rgb_fft", "rgb_wavelet"])
def test_constant_artifacts_normalize_to_finite_zero_tensors(mode):
    cnn = build_cnn_input(torch.full((3, 16, 16), 0.25), mode)
    assert torch.equal(cnn[3:], torch.zeros_like(cnn[3:]))


@pytest.mark.parametrize("mode", ["rgb_srm", "rgb_fft", "rgb_wavelet"])
def test_near_constant_artifacts_are_zeroed_instead_of_amplified(mode):
    """Catches min-max normalization amplifying float noise to full range.

    Unlike the exactly-constant case above, this input varies -- just by an
    amount far below any meaningful image signal. Dividing by that span would
    stretch pure noise across [-1, 1].
    """
    torch.manual_seed(0)
    rgb = torch.full((3, 16, 16), 0.25) + torch.randn(3, 16, 16) * 1e-6
    assert not torch.equal(rgb, torch.full_like(rgb, 0.25))
    cnn = build_cnn_input(rgb, mode)
    assert torch.isfinite(cnn).all()
    assert torch.equal(cnn[3:], torch.zeros_like(cnn[3:]))


@pytest.mark.parametrize("mode", ["rgb_srm", "rgb_fft", "rgb_wavelet"])
def test_solid_non_gray_frame_is_treated_as_constant(mode):
    """Catches a per-channel-flat frame slipping past a global-range check.

    Each channel is individually constant, so there is no signal for the
    derived artifacts to pick up -- even though the channels differ from each
    other and the *global* min/max range is therefore large.
    """
    rgb = torch.stack(
        [torch.full((16, 16), 0.9), torch.full((16, 16), -0.9), torch.full((16, 16), 0.0)]
    )
    cnn = build_cnn_input(rgb, mode)
    assert torch.equal(cnn[3:], torch.zeros_like(cnn[3:]))


def test_unknown_artifact_mode_reports_mode_and_sample_path():
    with pytest.raises(ValueError, match=r"rgb_dct.*bad.png"):
        build_cnn_input(torch.ones(3, 8, 8), "rgb_dct", "bad.png")


def test_invalid_artifact_input_reports_mode_and_sample_path():
    with pytest.raises(ValueError, match=r"rgb_fft.*bad.png"):
        build_cnn_input(torch.ones(1, 8, 8), "rgb_fft", "bad.png")
    with pytest.raises(ValueError, match="unknown artifact mode: rgb_dct"):
        artifact_channels("rgb_dct")


def test_rgb_mode_skips_derived_artifact_builders(monkeypatch):
    def fail(*_args):
        raise AssertionError("derived artifact builder invoked")

    monkeypatch.setattr(data, "_srm_artifact", fail)
    monkeypatch.setattr(data, "_fft_artifact", fail)
    monkeypatch.setattr(data, "_wavelet_artifact", fail)
    rgb = torch.linspace(-1, 1, 3 * 8 * 8).reshape(3, 8, 8)
    assert torch.equal(build_cnn_input(rgb, "rgb"), rgb)


def test_transform_returns_rgb_and_artifact_tensor_after_augmentation():
    transform = FaceTransform(32, color_jitter=0.2, jpeg_probability=1.0, artifact_mode="rgb_srm")
    rgb, cnn = transform(Image.new("RGB", (40, 36), "red"), sample_path="face.jpg")
    assert rgb.shape == (3, 32, 32)
    assert cnn.shape == (6, 32, 32)
    assert torch.equal(cnn[:3], rgb)


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
    tensor, cnn = transform(Image.new("RGB", (80, 72), "red"))
    assert tensor.shape == (3, 64, 64)
    assert cnn.shape == tensor.shape
    assert torch.isfinite(tensor).all()
    assert -1.0 <= tensor.min() <= tensor.max() <= 1.0


def _write_group_manifest(tmp_path):
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
    return manifest, methods


def test_grouped_dataset_builds_canonical_real_fake_domains(tmp_path):
    manifest, methods = _write_group_manifest(tmp_path)
    dataset = GroupedForgeryDataset(
        manifest, tmp_path, FaceTransform(224), forgery_methods=methods
    )
    rgb, cnn, labels = dataset[0]
    assert rgb.shape == (3, 3, 224, 224)
    assert torch.equal(cnn, rgb)
    assert torch.equal(labels, torch.tensor([0, 1, 2]))


def test_grouped_dataset_returns_rgb_cnn_and_shared_group_geometry(tmp_path):
    manifest, methods = _write_group_manifest(tmp_path)
    dataset = GroupedForgeryDataset(
        manifest,
        tmp_path,
        FaceTransform(32, crop_scale_min=0.75, artifact_mode="rgb_fft"),
        methods,
    )
    rgb, cnn, labels = dataset[0]
    assert rgb.shape == (3, 3, 32, 32)
    assert cnn.shape == (3, 6, 32, 32)
    assert torch.equal(cnn[:, :3], rgb)
    assert torch.equal(labels, torch.tensor([0, 1, 2]))


def test_artifact_mode_does_not_change_augmented_rgb_rng_sequence(tmp_path):
    manifest, methods = _write_group_manifest(tmp_path)
    random.seed(11); torch.manual_seed(11)
    rgb_a, cnn_a, _ = GroupedForgeryDataset(
        manifest,
        tmp_path,
        FaceTransform(32, color_jitter=0.2, jpeg_probability=1.0, artifact_mode="rgb"),
        methods,
    )[0]
    random.seed(11); torch.manual_seed(11)
    rgb_b, cnn_b, _ = GroupedForgeryDataset(
        manifest,
        tmp_path,
        FaceTransform(32, color_jitter=0.2, jpeg_probability=1.0, artifact_mode="rgb_srm_fft"),
        methods,
    )[0]
    assert torch.equal(rgb_a, rgb_b)
    assert torch.equal(cnn_a, rgb_a)
    assert cnn_b.shape[1] == 9


def test_frame_dataset_returns_rgb_cnn_label_and_video_id(tmp_path):
    Image.new("RGB", (20, 20), "white").save(tmp_path / "face.jpg")
    manifest = tmp_path / "frames.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "label", "video_id"])
        writer.writeheader()
        writer.writerow({"path": "face.jpg", "label": "1", "video_id": "video"})
    rgb, cnn, label, video_id = FrameFaceDataset(
        manifest, tmp_path, FaceTransform(32, artifact_mode="rgb_srm")
    )[0]
    assert rgb.shape == (3, 32, 32)
    assert cnn.shape == (6, 32, 32)
    assert torch.equal(cnn[:3], rgb)
    assert (label, video_id) == (1, "video")


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
