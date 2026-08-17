from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import Literal

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


EvaluationLevel = Literal["frame", "video"]


def _binary_metrics(
    probabilities: Sequence[float],
    labels: Sequence[int],
    threshold: float,
) -> dict[str, float]:
    if len(probabilities) != len(labels):
        raise ValueError("probabilities and labels must have equal length")
    if len(probabilities) == 0:
        raise ValueError("metrics require at least one sample")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")

    probability_array = np.asarray(probabilities, dtype=np.float64)
    label_array = np.asarray(labels, dtype=np.int64)
    if not np.isfinite(probability_array).all():
        raise ValueError("probabilities must be finite")
    if not np.isin(label_array, (0, 1)).all():
        raise ValueError("labels must contain only 0 (real) and 1 (fake)")
    if np.unique(label_array).size != 2:
        raise ValueError("AUC requires both real and fake samples")

    predictions = (probability_array >= threshold).astype(np.int64)
    return {
        "accuracy": float(accuracy_score(label_array, predictions)),
        "f1_score": float(f1_score(label_array, predictions, zero_division=0)),
        "precision": float(
            precision_score(label_array, predictions, zero_division=0)
        ),
        "recall": float(recall_score(label_array, predictions, zero_division=0)),
        "auc": float(roc_auc_score(label_array, probability_array)),
    }


def evaluation_metrics(
    frame_probabilities: Sequence[float],
    frame_labels: Sequence[int],
    video_ids: Sequence[str],
    level: EvaluationLevel = "video",
    threshold: float = 0.5,
) -> dict[str, float | int | str]:
    """Calculate binary deepfake metrics at frame or video level.

    Video probabilities are the arithmetic mean of all frame probabilities for
    the same ``video_id``. Precision, recall and F1 treat fake (label 1) as the
    positive class. AUC always uses probabilities instead of thresholded labels.
    """

    if not (
        len(frame_probabilities) == len(frame_labels) == len(video_ids)
    ):
        raise ValueError("probabilities, labels, and video_ids must have equal length")
    if level not in ("frame", "video"):
        raise ValueError("level must be 'frame' or 'video'")

    if level == "frame":
        metrics: dict[str, float | int | str] = {
            "level": "frame",
            "threshold": float(threshold),
            **_binary_metrics(frame_probabilities, frame_labels, threshold),
            "num_frames": len(frame_probabilities),
        }
        return metrics

    grouped_probabilities: dict[str, list[float]] = defaultdict(list)
    grouped_labels: dict[str, set[int]] = defaultdict(set)
    for probability, label, video_id in zip(
        frame_probabilities, frame_labels, video_ids, strict=True
    ):
        grouped_probabilities[str(video_id)].append(float(probability))
        grouped_labels[str(video_id)].add(int(label))
    inconsistent = [key for key, labels in grouped_labels.items() if len(labels) != 1]
    if inconsistent:
        raise ValueError(f"videos have inconsistent labels: {inconsistent[:3]}")

    ordered_ids = sorted(grouped_probabilities)
    video_probabilities = [
        float(np.mean(grouped_probabilities[key])) for key in ordered_ids
    ]
    video_labels = [next(iter(grouped_labels[key])) for key in ordered_ids]
    return {
        "level": "video",
        "threshold": float(threshold),
        **_binary_metrics(video_probabilities, video_labels, threshold),
        "num_videos": len(ordered_ids),
        "num_frames": len(frame_probabilities),
    }


def video_level_metrics(
    frame_probabilities: list[float], frame_labels: list[int], video_ids: list[str]
) -> dict[str, float | int]:
    """Backward-compatible video metrics used by the training loop."""

    metrics = evaluation_metrics(
        frame_probabilities, frame_labels, video_ids, level="video"
    )
    return {
        "video_auc": float(metrics["auc"]),
        "video_accuracy": float(metrics["accuracy"]),
        "video_f1_score": float(metrics["f1_score"]),
        "video_precision": float(metrics["precision"]),
        "video_recall": float(metrics["recall"]),
        "num_videos": int(metrics["num_videos"]),
        "num_frames": int(metrics["num_frames"]),
    }
