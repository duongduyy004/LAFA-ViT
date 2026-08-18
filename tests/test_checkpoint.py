import pytest
import torch

from train import load_favit_initialization, save_checkpoint


def test_checkpoint_is_atomic(tmp_path):
    path = tmp_path / "model.pt"
    save_checkpoint(path, {"architecture": "favit_lsda", "epoch": 1})
    assert torch.load(path, weights_only=False)["epoch"] == 1
    assert not (tmp_path / "model.pt.tmp").exists()


def test_favit_initialization_loads_only_compatible_tensors(tmp_path):
    model = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.Linear(2, 1))
    model.artifact_cnn = torch.nn.Module()
    model.artifact_cnn.stem = torch.nn.Sequential(torch.nn.Linear(4, 4))
    original = model[0].weight.detach().clone()
    original_cnn_weight = model.artifact_cnn.stem[0].weight.detach().clone()
    checkpoint = {
        "model": {
            "0.weight": torch.full_like(model[0].weight, 7.0),
            "0.bias": torch.zeros(99),
            "not_in_target": torch.ones(1),
        }
    }
    path = tmp_path / "favit.pt"
    torch.save(checkpoint, path)
    assert load_favit_initialization(model, path) == 1
    assert not torch.equal(model[0].weight, original)
    assert torch.equal(model[0].weight, torch.full_like(model[0].weight, 7.0))
    # The CNN artifact branch has no counterpart in a FA-ViT-only source
    # checkpoint, so it must stay freshly initialized rather than being
    # partially/incorrectly loaded.
    assert torch.equal(model.artifact_cnn.stem[0].weight, original_cnn_weight)


def test_favit_initialization_excludes_head_even_when_shape_matches(tmp_path):
    model = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.Linear(2, 1))
    model.head = torch.nn.Linear(2, 2, bias=False)
    original_head_weight = model.head.weight.detach().clone()
    checkpoint = {
        "model": {
            "0.weight": torch.full_like(model[0].weight, 7.0),
            "head.weight": torch.full_like(model.head.weight, 3.0),
        }
    }
    path = tmp_path / "favit.pt"
    torch.save(checkpoint, path)
    loaded = load_favit_initialization(model, path)
    assert loaded == 1
    assert torch.equal(model[0].weight, torch.full_like(model[0].weight, 7.0))
    assert torch.equal(model.head.weight, original_head_weight)


def test_checkpoint_rejects_artifact_metadata_mismatch(tmp_path):
    from favit_lsda.checkpoints import validate_checkpoint_artifacts
    checkpoint = {"architecture": "favit_lsda_cnn", "artifact_mode": "rgb_fft", "cnn_in_channels": 6}
    with pytest.raises(ValueError, match=r"checkpoint/config artifact mismatch.*rgb_srm"):
        validate_checkpoint_artifacts(checkpoint, {"artifact_mode": "rgb_srm", "cnn_in_channels": 6}, tmp_path / "last.pt")


def test_checkpoint_rejects_legacy_architecture(tmp_path):
    from favit_lsda.checkpoints import validate_checkpoint_artifacts
    checkpoint = {"architecture": "favit_lsda", "artifact_mode": "rgb", "cnn_in_channels": 3}
    with pytest.raises(ValueError, match=r"legacy architecture.*--init-favit"):
        validate_checkpoint_artifacts(
            checkpoint, {"artifact_mode": "rgb", "cnn_in_channels": 3}, tmp_path / "old.pt"
        )
