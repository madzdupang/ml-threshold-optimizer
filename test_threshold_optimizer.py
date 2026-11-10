import pytest
from threshold_optimizer import find_best_threshold


def test_find_best_threshold():
    result = find_best_threshold(
        thresholds=[0.2, 0.5, 0.7],
        precisions=[0.4, 0.8, 0.9],
        recalls=[1.0, 0.85, 0.6],
        minimum_recall=0.8
    )

    assert result == (0.5, 0.8)


def test_invalid_minimum_recall():
    with pytest.raises(ValueError):
        find_best_threshold(
            thresholds=[0.2, 0.5, 0.7],
            precisions=[0.4, 0.8, 0.9],
            recalls=[1.0, 0.85, 0.6],
            minimum_recall=1.5
        )


def test_nan_precision():
    with pytest.raises(ValueError):
        find_best_threshold(
            thresholds=[0.2, 0.5, 0.7],
            precisions=[0.4, float('nan'), 0.9],
            recalls=[1.0, 0.85, 0.6],
            minimum_recall=0.8
        )
