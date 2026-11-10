import math


def find_best_threshold(
    thresholds: list[float],
    precisions: list[float],
    recalls: list[float],
    minimum_recall: float
) -> tuple[float, float] | None:

    if len(thresholds) != len(precisions) or len(thresholds) != len(recalls):
        raise ValueError("Input lists must have the same length.")

    for i in range(len(precisions)):
        if not math.isfinite(precisions[i]) or not (0 <= precisions[i] <= 1):
            raise ValueError(
                f"Precision value at index {i} is out of range: {precisions[i]}")
        if not math.isfinite(recalls[i]) or not (0 <= recalls[i] <= 1):
            raise ValueError(
                f"Recall value at index {i} is out of range: {recalls[i]}")

    if not math.isfinite(minimum_recall) or minimum_recall < 0 or minimum_recall > 1:
        raise ValueError(
            f"Minimum recall must be between 0 and 1.")

    best_threshold = None
    best_precision = -1.0

    for i in range(len(thresholds)):
        current_threshold = thresholds[i]
        current_precision = precisions[i]
        current_recall = recalls[i]

        if current_recall >= minimum_recall:
            if current_precision > best_precision:
                best_precision = current_precision
                best_threshold = current_threshold

    if best_threshold is None:
        return None

    return (best_threshold, best_precision)
