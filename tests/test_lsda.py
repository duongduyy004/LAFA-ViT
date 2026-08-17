import pytest
import torch

from favit_lsda.lsda import LatentSpaceAugmenter


@pytest.mark.parametrize(
    "transform",
    ["hard_interpolation", "centrifugal", "gaussian", "rotation", "difference"],
)
def test_each_within_domain_transform_preserves_shape_and_backpropagates(transform):
    augmenter = LatentSpaceAugmenter(8, transforms=[transform])
    features = torch.randn(3, 4, 8, 5, 5, requires_grad=True)
    output, details = augmenter(features, return_details=True)
    assert output.shape == features.shape
    assert details["within"].shape == features.shape
    assert details["cross"].shape == features.shape
    output.square().mean().backward()
    assert features.grad is not None
    assert torch.isfinite(features.grad).all()


def test_cross_domain_requires_multiple_fake_domains():
    augmenter = LatentSpaceAugmenter(4)
    with pytest.raises(ValueError, match="at least two"):
        augmenter(torch.randn(2, 1, 4, 3, 3))


def test_invalid_transform_is_rejected():
    with pytest.raises(ValueError, match="invalid latent"):
        LatentSpaceAugmenter(4, transforms=["pixel_blend"])


def test_comprehensive_target_starts_as_identity():
    augmenter = LatentSpaceAugmenter(4)
    features = torch.randn(2, 3, 4, 3, 3)
    with torch.no_grad():
        output = augmenter(features)
    assert torch.equal(output, features)
