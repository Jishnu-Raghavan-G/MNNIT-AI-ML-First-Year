import numpy as np


def confusion_matrix_binary(y_true, y_pred):
    """
    Return:

        [[TN, FP],
         [FN, TP]]
    """

    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    tn = np.sum(
        (y_true == 0) & (y_pred == 0)
    )

    fp = np.sum(
        (y_true == 0) & (y_pred == 1)
    )

    fn = np.sum(
        (y_true == 1) & (y_pred == 0)
    )

    tp = np.sum(
        (y_true == 1) & (y_pred == 1)
    )

    return np.array(
        [
            [tn, fp],
            [fn, tp],
        ]
    )


def accuracy(y_true, y_pred):
    return np.mean(
        np.asarray(y_true).ravel()
        == np.asarray(y_pred).ravel()
    )


def precision(y_true, y_pred):
    matrix = confusion_matrix_binary(
        y_true,
        y_pred,
    )

    _, fp = matrix[0]
    _, tp = matrix[1]

    if tp + fp == 0:
        return 0.0

    return tp / (tp + fp)


def recall(y_true, y_pred):
    matrix = confusion_matrix_binary(
        y_true,
        y_pred,
    )

    fn, tp = matrix[1]

    if tp + fn == 0:
        return 0.0

    return tp / (tp + fn)


def f1_score(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)

    if p + r == 0:
        return 0.0

    return 2 * p * r / (p + r)


def predictions_from_threshold(
    probabilities,
    threshold,
):
    """
    Convert probabilities to binary predictions.
    """

    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    return (
        probabilities >= threshold
    ).astype(int)


def evaluate_threshold(
    y_true,
    probabilities,
    threshold,
):
    """
    Calculate classification metrics
    for one probability threshold.
    """

    predictions = predictions_from_threshold(
        probabilities,
        threshold,
    )

    matrix = confusion_matrix_binary(
        y_true,
        predictions,
    )

    return {
        "threshold": threshold,
        "accuracy": accuracy(
            y_true,
            predictions,
        ),
        "precision": precision(
            y_true,
            predictions,
        ),
        "recall": recall(
            y_true,
            predictions,
        ),
        "f1": f1_score(
            y_true,
            predictions,
        ),
        "tp": matrix[1, 1],
        "fp": matrix[0, 1],
        "fn": matrix[1, 0],
        "tn": matrix[0, 0],
    }


def run_threshold_experiment():
    """
    Demonstrate how changing the threshold
    changes classification metrics.
    """

    y_true = np.array(
        [
            0,
            0,
            1,
            1,
            0,
            1,
            0,
            1,
            1,
            0,
        ]
    )

    probabilities = np.array(
        [
            0.10,
            0.25,
            0.35,
            0.55,
            0.60,
            0.65,
            0.72,
            0.80,
            0.90,
            0.95,
        ]
    )

    thresholds = [
        0.30,
        0.40,
        0.50,
        0.60,
        0.70,
        0.80,
    ]

    print("=" * 90)
    print("CLASSIFICATION THRESHOLD EXPERIMENT")
    print("=" * 90)

    print("\nActual labels:")
    print(y_true)

    print("\nPredicted probabilities:")
    print(probabilities)

    print(
        "\n"
        "Threshold controls when a probability "
        "becomes class 1."
    )

    print(
        "\n"
        "threshold | accuracy | precision | "
        "recall | F1 | TP | FP | FN | TN"
    )

    print("-" * 90)

    results = []

    for threshold in thresholds:
        result = evaluate_threshold(
            y_true,
            probabilities,
            threshold,
        )

        results.append(result)

        print(
            f"{result['threshold']:9.2f} | "
            f"{result['accuracy']:8.3f} | "
            f"{result['precision']:9.3f} | "
            f"{result['recall']:6.3f} | "
            f"{result['f1']:5.3f} | "
            f"{result['tp']:2d} | "
            f"{result['fp']:2d} | "
            f"{result['fn']:2d} | "
            f"{result['tn']:2d}"
        )

    return results


def compare_extreme_thresholds():
    """
    Show the effect of very low and very high thresholds.
    """

    y_true = np.array(
        [0, 0, 1, 1, 0, 1, 0, 1]
    )

    probabilities = np.array(
        [0.10, 0.30, 0.45, 0.55,
         0.65, 0.75, 0.85, 0.95]
    )

    thresholds = [
        0.10,
        0.50,
        0.90,
    ]

    print("\n" + "=" * 90)
    print("LOW vs DEFAULT vs HIGH THRESHOLD")
    print("=" * 90)

    for threshold in thresholds:
        predictions = predictions_from_threshold(
            probabilities,
            threshold,
        )

        print(
            f"\nThreshold = {threshold:.2f}"
        )

        print(
            "Predictions:",
            predictions,
        )

        print(
            f"Precision: "
            f"{precision(y_true, predictions):.3f}"
        )

        print(
            f"Recall: "
            f"{recall(y_true, predictions):.3f}"
        )

        print(
            f"F1-score: "
            f"{f1_score(y_true, predictions):.3f}"
        )


if __name__ == "__main__":
    run_threshold_experiment()

    compare_extreme_thresholds()
