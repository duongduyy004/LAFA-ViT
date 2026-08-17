from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class ResidualLatentAdapter(nn.Module):
    """A lightweight domain/student encoder operating on ViT patch maps."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
        )
        self.scale = nn.Parameter(torch.zeros(channels))

    def forward(self, features: Tensor) -> Tensor:
        residual = self.block(features)
        return features + self.scale.view(1, -1, 1, 1) * residual


class LatentSpaceAugmenter(nn.Module):
    """LSDA within/cross-domain augmentation and Eq. (7)--(8) fusion.

    Input has shape ``[groups, fake_domains, channels, height, width]``. The
    default transformation pool combines all four paper transformations and
    the difference transformation present in the supplied reference code.
    """

    SUPPORTED_TRANSFORMS = {
        "hard_interpolation",
        "centrifugal",
        "gaussian",
        "rotation",
        "difference",
    }

    def __init__(
        self,
        channels: int,
        transforms: Sequence[str] = (
            "hard_interpolation",
            "centrifugal",
            "gaussian",
            "rotation",
            "difference",
        ),
        max_rotation_degrees: float = 30.0,
        noise_std: float = 1.0,
        mixup_concentration_min: float = 0.5,
        mixup_concentration_max: float = 2.0,
    ) -> None:
        super().__init__()
        invalid = set(transforms) - self.SUPPORTED_TRANSFORMS
        if invalid or not transforms:
            raise ValueError(f"invalid latent transformations: {sorted(invalid)}")
        if mixup_concentration_min <= 0 or mixup_concentration_max < mixup_concentration_min:
            raise ValueError("invalid mixup concentration range")
        self.channels = int(channels)
        self.transforms = tuple(transforms)
        self.max_rotation_degrees = float(max_rotation_degrees)
        self.noise_std = float(noise_std)
        self.mixup_concentration_min = float(mixup_concentration_min)
        self.mixup_concentration_max = float(mixup_concentration_max)
        self.augmented_fusion = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GELU(),
        )
        self.comprehensive_fusion = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
        )
        # Start from the domain-teacher feature instead of asking the student to
        # imitate a random convolutional target during the first optimizer steps.
        self.comprehensive_scale = nn.Parameter(torch.zeros(channels))

    @staticmethod
    def _sample_scale(reference: Tensor) -> Tensor:
        return torch.rand(
            (reference.shape[0], 1, 1, 1),
            device=reference.device,
            dtype=reference.dtype,
        )

    def _rotate(self, features: Tensor) -> Tensor:
        batch = features.shape[0]
        angles = (
            torch.rand(batch, device=features.device, dtype=features.dtype) * 2.0 - 1.0
        ) * (self.max_rotation_degrees * math.pi / 180.0)
        cosine, sine = angles.cos(), angles.sin()
        theta = torch.zeros(batch, 2, 3, device=features.device, dtype=features.dtype)
        theta[:, 0, 0] = cosine
        theta[:, 0, 1] = -sine
        theta[:, 1, 0] = sine
        theta[:, 1, 1] = cosine
        grid = F.affine_grid(theta, features.shape, align_corners=False)
        return F.grid_sample(
            features, grid, mode="bilinear", padding_mode="border", align_corners=False
        )

    def _within_domain(self, fake_features: Tensor) -> Tensor:
        groups, domains = fake_features.shape[:2]
        domain_means = fake_features.mean(dim=0, keepdim=True)
        distances = (fake_features - domain_means).flatten(2).norm(dim=2)
        hard_indices = distances.argmax(dim=0)
        domain_indices = torch.arange(domains, device=fake_features.device)
        hard_examples = fake_features[hard_indices, domain_indices]

        augmented_domains: list[Tensor] = []
        for domain in range(domains):
            features = fake_features[:, domain]
            transform_index = int(torch.randint(len(self.transforms), (1,)).item())
            transform = self.transforms[transform_index]
            scale = self._sample_scale(features)
            if transform == "hard_interpolation":
                augmented = features + scale * (hard_examples[domain] - features)
            elif transform == "centrifugal":
                augmented = features + scale * (features - domain_means[:, domain])
            elif transform == "gaussian":
                noise = torch.randn_like(features) * self.noise_std
                augmented = features + scale * noise
            elif transform == "rotation":
                augmented = self._rotate(features)
            else:
                first = fake_features[:, (domain + 1) % domains]
                second = fake_features[:, (domain + 2) % domains]
                augmented = features + scale * (first - second)
            augmented_domains.append(augmented)
        return torch.stack(augmented_domains, dim=1).reshape_as(fake_features)

    def _cross_domain(self, fake_features: Tensor) -> Tensor:
        groups, domains = fake_features.shape[:2]
        if domains < 2:
            raise ValueError("cross-domain augmentation requires at least two fake domains")
        shift = int(torch.randint(1, domains, (1,)).item())
        partners = fake_features.roll(shifts=shift, dims=1)
        # Beta sampling is kept in float32 for CUDA AMP compatibility.
        concentration = torch.rand(
            groups, 1, 1, 1, 1, device=fake_features.device, dtype=torch.float32
        )
        concentration = concentration * (
            self.mixup_concentration_max - self.mixup_concentration_min
        ) + self.mixup_concentration_min
        mix = torch.distributions.Beta(concentration, concentration).sample().to(
            fake_features.dtype
        )
        return mix * fake_features + (1.0 - mix) * partners

    def forward(
        self, fake_features: Tensor, return_details: bool = False
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        if fake_features.ndim != 5 or fake_features.shape[2] != self.channels:
            raise ValueError(
                f"fake_features must be [G, K, {self.channels}, H, W]"
            )
        groups, domains, channels, height, width = fake_features.shape
        within = self._within_domain(fake_features)
        cross = self._cross_domain(fake_features)
        augmented = self.augmented_fusion(
            torch.cat((within, cross), dim=2).reshape(
                groups * domains, channels * 2, height, width
            )
        )
        original = fake_features.reshape(groups * domains, channels, height, width)
        residual = self.comprehensive_fusion(
            torch.cat((augmented, original), dim=1)
        )
        scale = self.comprehensive_scale.view(1, channels, 1, 1)
        comprehensive = (original + scale * residual).reshape_as(fake_features)
        if return_details:
            return comprehensive, {"within": within, "cross": cross}
        return comprehensive
