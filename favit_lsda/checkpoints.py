"""Checkpoint metadata validation shared by training and evaluation.

Kept separate from ``train.py`` so that ``favit_lsda/evaluation.py`` can
validate checkpoint compatibility without importing the CLI training module.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

EXPECTED_ARCHITECTURE = "favit_lsda_cnn"
SUPPORTED_FORMAT_VERSION = 3


def validate_checkpoint_artifacts(
    checkpoint: dict[str, Any],
    model_config: dict[str, Any],
    checkpoint_path: Path,
) -> None:
    """Reject a checkpoint before its state is loaded or used for inference.

    Three independent checks run, in order:

    1. Legacy architecture rejection: a checkpoint saved before the CNN
       artifact branch existed (``architecture`` other than
       ``"favit_lsda_cnn"``) cannot be resumed or evaluated directly; it must
       be loaded with ``--init-favit`` into a fresh model instead.
    2. Format version rejection: ``format_version`` is written on save but
       would otherwise never be read back, so a future bump of the on-disk
       layout would load silently against code that cannot interpret it.
       Only ``SUPPORTED_FORMAT_VERSION`` is accepted.
    3. Artifact mismatch rejection: the checkpoint's saved ``artifact_mode``
       and ``cnn_in_channels`` must match what ``model_config`` requests, so
       a checkpoint trained with one CNN artifact mode is never silently
       resumed or evaluated under a different one.
    """
    from .data import artifact_channels

    architecture = checkpoint.get("architecture")
    if architecture != EXPECTED_ARCHITECTURE:
        raise ValueError(
            f"checkpoint at {checkpoint_path} has legacy architecture {architecture!r}; "
            f"expected {EXPECTED_ARCHITECTURE!r}. Start a new run and load this "
            "checkpoint with --init-favit instead of --resume or evaluation."
        )

    format_version = checkpoint.get("format_version")
    if format_version != SUPPORTED_FORMAT_VERSION:
        raise ValueError(
            f"checkpoint at {checkpoint_path} has unsupported format_version "
            f"{format_version!r}; this build only reads format_version "
            f"{SUPPORTED_FORMAT_VERSION}. Use a build matching the checkpoint, "
            "or start a new run and load it with --init-favit."
        )

    checkpoint_mode = checkpoint.get("artifact_mode")
    checkpoint_width = checkpoint.get("cnn_in_channels")
    config_mode = str(model_config.get("artifact_mode", "rgb"))
    config_width = int(
        model_config.get("cnn_in_channels", artifact_channels(config_mode))
    )
    if checkpoint_mode != config_mode or checkpoint_width != config_width:
        raise ValueError(
            "checkpoint/config artifact mismatch: "
            f"checkpoint mode={checkpoint_mode!r} width={checkpoint_width!r}, "
            f"config mode={config_mode!r} width={config_width!r} "
            f"({checkpoint_path})"
        )
