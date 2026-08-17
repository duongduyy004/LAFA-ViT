import pytest

from favit_lsda.metrics import evaluation_metrics, video_level_metrics


def test_video_level_aggregation():
    result = video_level_metrics(
        [0.1, 0.2, 0.8, 0.9],
        [0, 0, 1, 1],
        ["real", "real", "fake", "fake"],
    )
    assert result["video_auc"] == 1.0
    assert result["video_accuracy"] == 1.0
    assert result["video_f1_score"] == 1.0
    assert result["video_precision"] == 1.0
    assert result["video_recall"] == 1.0
    assert result["num_videos"] == 2


def test_frame_level_binary_metrics():
    result = evaluation_metrics(
        [0.1, 0.8, 0.9, 0.4],
        [0, 0, 1, 1],
        ["real-1", "real-2", "fake-1", "fake-2"],
        level="frame",
    )
    assert result["level"] == "frame"
    assert result["accuracy"] == 0.5
    assert result["f1_score"] == 0.5
    assert result["precision"] == 0.5
    assert result["recall"] == 0.5
    assert result["auc"] == 0.75
    assert result["num_frames"] == 4


def test_video_level_averages_frame_probabilities():
    result = evaluation_metrics(
        [0.1, 0.3, 0.7, 0.9],
        [0, 0, 1, 1],
        ["real", "real", "fake", "fake"],
        level="video",
    )
    assert result["accuracy"] == 1.0
    assert result["f1_score"] == 1.0
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["auc"] == 1.0
    assert result["num_videos"] == 2
    assert result["num_frames"] == 4


def test_video_level_rejects_inconsistent_labels():
    with pytest.raises(ValueError, match="inconsistent labels"):
        evaluation_metrics(
            [0.1, 0.9, 0.8],
            [0, 1, 1],
            ["same-video", "same-video", "fake"],
            level="video",
        )
