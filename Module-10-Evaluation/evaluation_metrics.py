import numpy as np
import pandas as pd
from pathlib import Path


def confusion_matrix_binary(y_true, y_pred):
    """
    Binary confusion matrix:

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
    """
    Accuracy = (TP + TN) / Total
    """

    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    return np.mean(y_true == y_pred)


def precision(y_true, y_pred):
    """
    Precision = TP / (TP + FP)
    """

    matrix = confusion_matrix_binary(
        y_true,
        y_pred,
    )

    _, fp = matrix[0]
    _, tp = matrix[1]

    denominator = tp + fp

    if denominator == 0:
        return 0.0

    return tp / denominator


def recall(y_true, y_pred):
    """
    Recall = TP / (TP + FN)
    """

    matrix = confusion_matrix_binary(
        y_true,
        y_pred,
    )

    fn, tp = matrix[1]

    denominator = tp + fn

    if denominator == 0:
        return 0.0

    return tp / denominator


def f1_score(y_true, y_pred):
    """
    F1 = harmonic mean of precision and recall.
    """

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)

    denominator = p + r

    if denominator == 0:
        return 0.0

    return 2 * p * r / denominator


def mean_absolute_error(y_true, y_pred):
    """
    MAE = average absolute error.
    """

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    return np.mean(
        np.abs(y_true - y_pred)
    )


def mean_squared_error(y_true, y_pred):
    """
    MSE = average squared error.
    """

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    return np.mean(
        (y_true - y_pred) ** 2
    )


def root_mean_squared_error(y_true, y_pred):
    """
    RMSE = sqrt(MSE).
    """

    return np.sqrt(
        mean_squared_error(
            y_true,
            y_pred,
        )
    )


def sum_absolute_error(y_true, y_pred):
    """
    SAE = sum of absolute errors.
    """

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    return np.sum(
        np.abs(y_true - y_pred)
    )


def sum_squared_error(y_true, y_pred):
    """
    SSE = sum of squared errors.
    """

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    return np.sum(
        (y_true - y_pred) ** 2
    )


def classification_demo():
    """
    Demonstrate classification metrics
    using a small binary classification example.
    """

    y_true = np.array(
        [1, 1, 1, 1, 0, 0, 0, 0]
    )

    y_pred = np.array(
        [1, 1, 0, 1, 1, 0, 0, 0]
    )

    matrix = confusion_matrix_binary(
        y_true,
        y_pred,
    )

    print("=" * 70)
    print("CLASSIFICATION EVALUATION")
    print("=" * 70)

    print("\nActual labels:")
    print(y_true)

    print("\nPredicted labels:")
    print(y_pred)

    print("\nConfusion matrix:")
    print(matrix)

    print(
        f"\nAccuracy: "
        f"{accuracy(y_true, y_pred):.4f}"
    )

    print(
        f"Precision: "
        f"{precision(y_true, y_pred):.4f}"
    )

    print(
        f"Recall: "
        f"{recall(y_true, y_pred):.4f}"
    )

    print(
        f"F1-score: "
        f"{f1_score(y_true, y_pred):.4f}"
    )


def regression_demo():
    """
    Demonstrate regression evaluation metrics.
    """

    y_true = np.array(
        [10, 20, 30, 40, 50],
        dtype=float,
    )

    y_pred = np.array(
        [12, 18, 27, 44, 48],
        dtype=float,
    )

    errors = y_true - y_pred

    print("\n" + "=" * 70)
    print("REGRESSION EVALUATION")
    print("=" * 70)

    print("\nActual values:")
    print(y_true)

    print("\nPredicted values:")
    print(y_pred)

    print("\nErrors:")
    print(errors)

    print(
        f"\nSAE: "
        f"{sum_absolute_error(y_true, y_pred):.4f}"
    )

    print(
        f"SSE: "
        f"{sum_squared_error(y_true, y_pred):.4f}"
    )

    print(
        f"MAE: "
        f"{mean_absolute_error(y_true, y_pred):.4f}"
    )

    print(
        f"MSE: "
        f"{mean_squared_error(y_true, y_pred):.4f}"
    )

    print(
        f"RMSE: "
        f"{root_mean_squared_error(y_true, y_pred):.4f}"
    )


def student_regression_demo():
    """
    Evaluate a simple prediction rule on the
    repository student performance dataset.

    This is intentionally a simple baseline:
    predicted score = average training score.
    """

    repo_root = Path(__file__).resolve().parents[2]

    dataset_path = (
        repo_root
        / "datasets"
        / "student_performance.csv"
    )

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}"
        )

    df = pd.read_csv(dataset_path)

    scores = df["score"].to_numpy(
        dtype=float
    )

    split_index = int(
        len(scores) * 0.8
    )

    train_scores = scores[:split_index]
    test_scores = scores[split_index:]

    baseline_prediction = np.mean(
        train_scores
    )

    predictions = np.full(
        len(test_scores),
        baseline_prediction,
    )

    print("\n" + "=" * 70)
    print("STUDENT SCORE REGRESSION BASELINE")
    print("=" * 70)

    print(
        f"\nTraining samples: "
        f"{len(train_scores)}"
    )

    print(
        f"Test samples: "
        f"{len(test_scores)}"
    )

    print(
        f"Baseline predicted score: "
        f"{baseline_prediction:.4f}"
    )

    print(
        f"\nMAE: "
        f"{mean_absolute_error(test_scores, predictions):.4f}"
    )

    print(
        f"MSE: "
        f"{mean_squared_error(test_scores, predictions):.4f}"
    )

    print(
        f"RMSE: "
        f"{root_mean_squared_error(test_scores, predictions):.4f}"
    )


def main():
    classification_demo()
    regression_demo()
    student_regression_demo()


if __name__ == "__main__":
    main()
