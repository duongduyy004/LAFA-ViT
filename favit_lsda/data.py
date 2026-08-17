from __future__ import annotations

import csv
import io
import random
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
from torchvision.transforms import ColorJitter
from torchvision.transforms.functional import InterpolationMode


class FaceTransform:
    """Face transform with train-only perturbations aimed at codec/domain shift.

    Geometric crop parameters can be shared by every image in an LSDA group,
    while photometric and codec perturbations remain independent per image.
    Defaults intentionally reproduce the old deterministic evaluation transform.
    """

    def __init__(
        self,
        image_size: int = 224,
        horizontal_flip: float = 0.0,
        crop_scale_min: float = 1.0,
        color_jitter: float = 0.0,
        grayscale_probability: float = 0.0,
        blur_probability: float = 0.0,
        degradation_probability: float = 0.0,
        jpeg_probability: float = 0.0,
        jpeg_quality_min: int = 40,
    ) -> None:
        self.image_size = image_size
        self.horizontal_flip = horizontal_flip
        self.crop_scale_min = float(crop_scale_min)
        self.grayscale_probability = float(grayscale_probability)
        self.blur_probability = float(blur_probability)
        self.degradation_probability = float(degradation_probability)
        self.jpeg_probability = float(jpeg_probability)
        self.jpeg_quality_min = int(jpeg_quality_min)
        probabilities = (
            horizontal_flip,
            grayscale_probability,
            blur_probability,
            degradation_probability,
            jpeg_probability,
        )
        if not all(0.0 <= probability <= 1.0 for probability in probabilities):
            raise ValueError("augmentation probabilities must be in [0, 1]")
        if not 0.0 < self.crop_scale_min <= 1.0:
            raise ValueError("crop_scale_min must be in (0, 1]")
        if not 1 <= self.jpeg_quality_min <= 100:
            raise ValueError("jpeg_quality_min must be in [1, 100]")
        jitter = float(color_jitter)
        self.color_jitter = (
            ColorJitter(
                brightness=jitter,
                contrast=jitter,
                saturation=jitter,
                hue=min(jitter / 4.0, 0.1),
            )
            if jitter > 0.0
            else None
        )

    def sample_flip(self) -> bool:
        return random.random() < self.horizontal_flip

    def sample_crop(self) -> tuple[float, float, float]:
        scale = random.uniform(self.crop_scale_min, 1.0)
        return scale, random.random(), random.random()

    def _resize_crop(
        self, image: Image.Image, crop: tuple[float, float, float]
    ) -> Image.Image:
        scale, vertical_position, horizontal_position = crop
        width, height = image.size
        side = max(1, round(min(width, height) * scale))
        top = round((height - side) * vertical_position)
        left = round((width - side) * horizontal_position)
        return TF.resized_crop(
            image,
            top,
            left,
            side,
            side,
            [self.image_size, self.image_size],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )

    def _apply_degradation(self, image: Image.Image) -> Image.Image:
        if random.random() < self.blur_probability:
            image = TF.gaussian_blur(image, kernel_size=5, sigma=random.uniform(0.1, 2.0))
        if random.random() < self.degradation_probability:
            ratio = random.uniform(0.45, 0.9)
            reduced_size = max(16, round(self.image_size * ratio))
            image = TF.resize(
                image,
                [reduced_size, reduced_size],
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            )
            image = TF.resize(
                image,
                [self.image_size, self.image_size],
                interpolation=random.choice(
                    (InterpolationMode.BILINEAR, InterpolationMode.BICUBIC)
                ),
                antialias=True,
            )
        if random.random() < self.jpeg_probability:
            buffer = io.BytesIO()
            image.save(
                buffer,
                format="JPEG",
                quality=random.randint(self.jpeg_quality_min, 100),
                subsampling=random.choice((0, 1, 2)),
            )
            buffer.seek(0)
            with Image.open(buffer) as encoded:
                image = encoded.convert("RGB").copy()
        return image

    def __call__(
        self,
        image: Image.Image,
        flip: bool = False,
        crop: tuple[float, float, float] | None = None,
    ) -> Tensor:
        image = self._resize_crop(image.convert("RGB"), crop or self.sample_crop())
        if flip:
            image = TF.hflip(image)
        if self.color_jitter is not None:
            image = self.color_jitter(image)
        if random.random() < self.grayscale_probability:
            image = TF.rgb_to_grayscale(image, num_output_channels=3)
        image = self._apply_degradation(image)
        return TF.normalize(TF.to_tensor(image), [0.5] * 3, [0.5] * 3)


def _read_manifest(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"manifest is empty: {path}")
    return rows


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


class GroupedForgeryDataset(Dataset[tuple[Tensor, Tensor]]):
    """Build LSDA groups in canonical order: real, then each forgery method."""

    REQUIRED_COLUMNS = {"fake_path", "real_path", "method"}

    def __init__(
        self,
        manifest: str | Path,
        data_root: str | Path,
        transform: FaceTransform,
        forgery_methods: tuple[str, ...] = (
            "Deepfakes",
            "Face2Face",
            "FaceSwap",
            "NeuralTextures",
        ),
    ) -> None:
        rows = _read_manifest(manifest)
        missing = self.REQUIRED_COLUMNS - rows[0].keys()
        if missing:
            raise ValueError(f"LSDA manifest is missing columns: {sorted(missing)}")
        self.data_root = Path(data_root)
        self.transform = transform
        self.forgery_methods = tuple(forgery_methods)
        canonical = {method.casefold(): method for method in self.forgery_methods}
        grouped: dict[str, dict[str, list[dict[str, str]]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for row in rows:
            method = canonical.get(row["method"].casefold())
            if method is not None:
                grouped[row["real_path"]][method].append(row)
        self.groups = [
            (real_path, method_rows)
            for real_path, method_rows in grouped.items()
            if all(method_rows[method] for method in self.forgery_methods)
        ]
        self.dropped_incomplete_groups = len(grouped) - len(self.groups)
        if not self.groups:
            raise ValueError(
                "manifest has no complete real + forgery-method groups for LSDA"
            )

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        real_path, method_rows = self.groups[index]
        with Image.open(_resolve(self.data_root, real_path)) as image:
            real = image.copy()
        selected_fakes = []
        for method in self.forgery_methods:
            row = random.choice(method_rows[method])
            with Image.open(_resolve(self.data_root, row["fake_path"])) as image:
                selected_fakes.append(image.copy())
        flip = self.transform.sample_flip()
        crop = self.transform.sample_crop()
        images = torch.stack(
            [self.transform(real, flip, crop)]
            + [self.transform(image, flip, crop) for image in selected_fakes]
        )
        domain_labels = torch.arange(len(self.forgery_methods) + 1, dtype=torch.long)
        return images, domain_labels


class FrameFaceDataset(Dataset[tuple[Tensor, int, str]]):
    REQUIRED_COLUMNS = {"path", "label", "video_id"}

    def __init__(
        self, manifest: str | Path, data_root: str | Path, transform: FaceTransform
    ) -> None:
        self.rows = _read_manifest(manifest)
        missing = self.REQUIRED_COLUMNS - self.rows[0].keys()
        if missing:
            raise ValueError(f"frame manifest is missing columns: {sorted(missing)}")
        self.data_root = Path(data_root)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[Tensor, int, str]:
        row = self.rows[index]
        with Image.open(_resolve(self.data_root, row["path"])) as image:
            tensor = self.transform(image.copy())
        return tensor, int(row["label"]), row["video_id"]
