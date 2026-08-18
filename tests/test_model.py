import pytest
import torch
from torch.nn import functional as F

from favit_lsda.losses import FineGrainedAdaptiveLoss, balanced_binary_cross_entropy
from favit_lsda.model import create_favit_lsda, gradient_reverse


def _tiny_model(artifact_mode: str = "rgb", cnn_in_channels: int = 3):
    return create_favit_lsda(
        model_name="vit_tiny_patch16_224",
        pretrained=False,
        forgery_methods=("DF", "F2F"),
        train_backbone_norms=False,
        train_cls_token=False,
        artifact_mode=artifact_mode,
        cnn_in_channels=cnn_in_channels,
    )


@pytest.mark.parametrize(
    ("mode", "channels"),
    [("rgb", 3), ("rgb_srm", 6), ("rgb_srm_fft", 9)],
)
def test_cnn_modes_forward_and_backpropagate(mode, channels):
    """Catches disconnected CNN or late-fusion detector paths."""
    model = _tiny_model(mode, channels)
    rgb = torch.randn(2, 3, 3, 224, 224)
    cnn = torch.randn(2, 3, channels, 224, 224)
    output = model.forward_group(rgb, cnn)
    loss = output["logits"].square().mean() + output["distill_real"] + output["distill_fake"]
    loss.backward()
    assert output["features"].shape == (2, 3, model.embed_dim)
    assert model.artifact_cnn.stem[0].weight.grad is not None
    assert model.head.weight.grad is not None


def test_late_fused_features_are_compatible_with_fal():
    """Catches returned FAL features bypassing CNN late fusion."""
    model = _tiny_model("rgb_fft", 6)
    output = model.forward_group(
        torch.randn(1, 3, 3, 224, 224), torch.randn(1, 3, 6, 224, 224)
    )
    real = output["features"][:, :1].expand(-1, 2, -1).reshape(-1, model.embed_dim)
    fake = output["features"][:, 1:].reshape(-1, model.embed_dim)
    FineGrainedAdaptiveLoss()(model.head.weight[0], real, fake).backward()
    assert model.artifact_cnn.stem[0].weight.grad is not None


def test_late_fused_inference_skips_teachers(monkeypatch):
    """Catches inference taking grouped teacher-only LSDA path."""
    model = _tiny_model("rgb_fft", 6).eval()
    monkeypatch.setattr(
        model.real_teacher,
        "forward",
        lambda _: (_ for _ in ()).throw(AssertionError("teacher called")),
    )
    logits, features = model(
        torch.randn(1, 3, 224, 224),
        torch.randn(1, 6, 224, 224),
        return_features=True,
    )
    assert logits.shape == (1, 2)
    assert features.shape == (1, model.embed_dim)


def test_model_rejects_artifact_mode_channel_mismatch():
    """Catches construction with config width not declared by artifact mode."""
    with pytest.raises(ValueError, match=r"artifact mode/channel mismatch: mode='rgb_srm' expects 6, got 3"):
        _tiny_model("rgb_srm", 3)


def test_group_forward_rejects_non_rgb_inputs_before_backbone():
    """Catches malformed RGB widths reaching Conv2d instead of model validation."""
    model = _tiny_model()
    with pytest.raises(ValueError, match=r"grouped_images must be \[G, 3, 3, H, W\]; got 1 channels"):
        model.forward_group(
            torch.randn(1, 3, 1, 224, 224), torch.randn(1, 3, 3, 224, 224)
        )


def test_group_invariance_classifier_uses_vit_features_only():
    """Catches domain-adversarial classifier receiving fused CNN features."""
    model = _tiny_model("rgb_fft", 6).eval()
    rgb = torch.randn(1, 3, 3, 224, 224)
    cnn = torch.randn(1, 3, 6, 224, 224)
    # torch.no_grad rather than torch.inference_mode: this reference pass warms
    # LocalAdaptiveAttention's lazily cached `relative_indices` buffer, and an
    # inference-mode tensor there cannot later be reused by the autograd-tracked
    # forward_group call below.
    with torch.no_grad():
        cls, patch_maps = model.encode_latents(rgb.flatten(0, 1))
        vit_features, _ = model._student_features(cls, patch_maps)
    captured = []
    hook = model.student_domain_classifier.register_forward_pre_hook(
        lambda _, inputs: captured.append(inputs[0].detach().clone())
    )
    try:
        model.forward_group(rgb, cnn)
    finally:
        hook.remove()
    assert torch.equal(captured[0], vit_features.reshape(1, 3, -1)[:, 1:])


def test_group_training_forward_and_full_backward():
    model = _tiny_model()
    images = torch.randn(2, 3, 3, 224, 224)
    cnn_images = torch.randn(2, 3, 3, 224, 224)
    domain_labels = torch.arange(3).expand(2, -1)
    output = model.forward_group(images, cnn_images)
    assert output["logits"].shape == (2, 3, 2)
    assert output["features"].shape == (2, 3, model.embed_dim)
    assert output["domain_logits"].shape == (2, 3, 3)
    assert output["invariance_logits"].shape == (2, 2, 2)
    assert output["distill_real"].ndim == 0
    assert output["distill_fake"].ndim == 0

    binary = F.cross_entropy(
        output["logits"].flatten(0, 1), (domain_labels > 0).long().flatten()
    )
    domain = F.cross_entropy(output["domain_logits"].flatten(0, 1), domain_labels.flatten())
    real = output["features"][:, :1].expand(-1, 2, -1).reshape(-1, model.embed_dim)
    fake = output["features"][:, 1:].reshape(-1, model.embed_dim)
    fal = FineGrainedAdaptiveLoss()(model.head.weight[0], real, fake)
    loss = binary + domain + output["distill_real"] + output["distill_fake"] + fal
    loss.backward()
    assert model.head.weight.grad is not None
    assert model.artifact_cnn.stem[0].weight.grad is not None
    assert model.latent_augmenter.comprehensive_fusion[0].weight.grad is not None


def test_gradient_reversal_only_negates_feature_gradient():
    features = torch.ones(2, 3, requires_grad=True)
    gradient_reverse(features, strength=0.25).sum().backward()
    assert torch.allclose(features.grad, torch.full_like(features, -0.25))


def test_binary_loss_balances_one_real_against_many_fakes():
    logits = torch.tensor([[3.0, -1.0]] + [[-0.5, 0.5]] * 4)
    labels = torch.tensor([0, 1, 1, 1, 1])
    loss = balanced_binary_cross_entropy(logits, labels)
    real = F.cross_entropy(logits[:1], labels[:1])
    fake = F.cross_entropy(logits[1:], labels[1:])
    assert torch.allclose(loss, (real + fake) / 2)


def test_inference_uses_single_image_student_path():
    model = _tiny_model().eval()
    with torch.inference_mode():
        logits, features = model(
            torch.randn(1, 3, 224, 224),
            torch.randn(1, 3, 224, 224),
            return_features=True,
        )
    assert logits.shape == (1, 2)
    assert features.shape == (1, model.embed_dim)


def test_favit_adapters_still_start_as_noops_and_backbone_is_frozen():
    model = _tiny_model()
    tokens = torch.randn(1, 197, model.embed_dim)
    assert torch.count_nonzero(model.backbone.blocks[0].attn.gam(tokens)) == 0
    assert torch.count_nonzero(model.injectors[0].scale) == 0
    assert not model.backbone.patch_embed.proj.weight.requires_grad
    assert model.student_adapter.scale.requires_grad
