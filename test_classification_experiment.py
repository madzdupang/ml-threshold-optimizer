import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from classification_experiment import split_dataset
from classification_experiment import evaluate_thresholds


def test_no_data_overlap():

    data = load_breast_cancer()
    indices = np.arange(len(data.target))

    train_indices, val_indices, test_indices, _, _, _ = split_dataset(
        indices, data.target)

    # Verify dataset sizes

    assert len(train_indices) == 341
    assert len(val_indices) == 114
    assert len(test_indices) == 114

    print("Training:", len(train_indices))
    print("Validation:", len(val_indices))
    print("Testing:", len(test_indices))

    # Verify that datasets do not overlap
    assert len(np.intersect1d(train_indices, val_indices)) == 0
    assert len(np.intersect1d(train_indices, test_indices)) == 0
    assert len(np.intersect1d(val_indices, test_indices)) == 0


def test_eval_thresholds():
    probabilities = np.array([0.9, 0.8, 0.4, 0.1])
    y_val = np.array([0, 0, 1, 1])

    thresholds = [0.5, 0.7]
    minimum_recall = 1.0

    result = evaluate_thresholds(
        probabilities, y_val, thresholds, minimum_recall)

    assert result is not None, "Expected a valid result, but got None."

    best_threshold, best_precision, best_recall = result

    assert best_threshold == 0.5
    assert best_precision == 1.0
    assert best_recall == 1.0


def test_no_qualifying_threshold():
    probabilities = np.array([0.9, 0.8, 0.4, 0.1])
    y_val = np.array([0, 0, 1, 1])

    thresholds = [0.85, 0.95]
    minimum_recall = 1.0

    result = evaluate_thresholds(
        probabilities, y_val, thresholds, minimum_recall)

    assert result is None
