from pathlib import Path

import pytest

from favit_lsda.config import load_config, validate_model_config


@pytest.mark.parametrize(
    ("name", "mode", "channels"),
    [
        ("rgb", "rgb", 3), ("rgb_srm", "rgb_srm", 6),
        ("rgb_fft", "rgb_fft", 6), ("rgb_wavelet", "rgb_wavelet", 6),
        ("rgb_srm_fft", "rgb_srm_fft", 9),
        ("rgb_srm_wavelet", "rgb_srm_wavelet", 9),
    ],
)
def test_cnn_experiment_config_has_exact_mapping(name, mode, channels):
    config = load_config(Path("configs") / f"favit_lsda_cnn_{name}.yaml")
    validate_model_config(config["model"])
    assert config["model"]["artifact_mode"] == mode
    assert config["model"]["cnn_in_channels"] == channels
    assert config["output_dir"] == f"outputs/favit_lsda_cnn_{name}"
    assert config["data"]["validation_frames"].endswith("ffpp_c23_val_frames.csv")


def test_model_config_rejects_artifact_width_mismatch():
    with pytest.raises(ValueError, match=r"rgb_srm.*expects 6.*got 3"):
        validate_model_config({"artifact_mode": "rgb_srm", "cnn_in_channels": 3})


def test_model_config_accepts_matching_width():
    validate_model_config({"artifact_mode": "rgb_srm_wavelet", "cnn_in_channels": 9})


def test_model_config_defaults_channels_from_mode_when_unset():
    validate_model_config({"artifact_mode": "rgb_fft"})


def test_model_config_defaults_to_rgb_when_mode_unset():
    validate_model_config({})
