import torch

from train import load_favit_initialization, save_checkpoint


def test_checkpoint_is_atomic(tmp_path):
    path = tmp_path / "model.pt"
    save_checkpoint(path, {"architecture": "favit_lsda", "epoch": 1})
    assert torch.load(path, weights_only=False)["epoch"] == 1
    assert not (tmp_path / "model.pt.tmp").exists()


def test_favit_initialization_loads_only_compatible_tensors(tmp_path):
    model = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.Linear(2, 1))
    original = model[0].weight.detach().clone()
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
