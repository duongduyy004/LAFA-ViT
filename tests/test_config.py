import pytest

from favit_lsda.config import validate_model_config


def test_model_config_rejects_artifact_width_mismatch():
    with pytest.raises(ValueError, match=r"rgb_srm.*expects 6.*got 3"):
        validate_model_config({"artifact_mode": "rgb_srm", "cnn_in_channels": 3})


def test_model_config_accepts_matching_width():
    validate_model_config({"artifact_mode": "rgb_srm_wavelet", "cnn_in_channels": 9})


def test_model_config_defaults_channels_from_mode_when_unset():
    validate_model_config({"artifact_mode": "rgb_fft"})


def test_model_config_defaults_to_rgb_when_mode_unset():
    validate_model_config({})
