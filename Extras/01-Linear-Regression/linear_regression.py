import numpy as np
import pandas as pd
from pathlib import Path


def add_intercept(X):
    """Add a column of ones for the intercept."""
    X = np.asarray(X, dtype=float)
    return np.column_stack((np.ones(X.shape[0]), X))


def normal_equation(X, y):
    """
    Solve linear regression using:

        beta = (X^T X)^-1 X^T y

    Uses solve() instead of explicitly calculating the inverse.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    X_with_intercept = add_intercept(X)

    beta = np.linalg.solve(
        X_with_intercept.T @ X_with_intercept,
        X_with_intercept.T @ y
    )

    return beta


def pseudoinverse_solution(X, y):
    """Solve linear regression using the Moore-Penrose pseudoinverse."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    X_with_intercept = add_intercept(X)

    beta = np.linalg.pinv(X_with_intercept) @ y

    return beta


def svd_solution(X, y):
    """Solve linear regression explicitly using SVD."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    X_with_intercept = add_intercept(X)

    U, singular_values, Vt = np.linalg.svd(
        X_with_intercept,
        full_matrices=False
    )

    # Construct the pseudoinverse of the diagonal singular-value matrix.
    tolerance = np.finfo(float).eps * max(X_with_intercept.shape) * singular_values[0]

    inverse_singular_values = np.array([
        1 / value if value > tolerance else 0.0
        for value in singular_values
    ])

    X_pseudoinverse = (
        Vt.T
        @ np.diag(inverse_singular_values)
        @ U.T
    )

    beta = X_pseudoinverse @ y

    return beta


def predict(X, beta):
    """Generate predictions using learned coefficients."""
    X = np.asarray(X, dtype=float)
    X_with_intercept = add_intercept(X)

    return X_with_intercept @ beta


def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def rmse(y_true, y_pred):
    return np.sqrt(mse(y_true, y_pred))


def demonstrate_simple_regression():
    """Small example of y = beta_0 + beta_1 x."""
    X = np.array([
        [1],
        [2],
        [3],
        [4],
        [5]
    ], dtype=float)

    y = np.array([
        3,
        5,
        7,
        9,
        11
    ], dtype=float)

    beta = pseudoinverse_solution(X, y)
    predictions = predict(X, beta)

    print("\n=== SIMPLE LINEAR REGRESSION ===")
    print("Learned coefficients:")
    print(f"Intercept: {beta[0]:.4f}")
    print(f"Slope:     {beta[1]:.4f}")

    print("\nPredictions:")
    for x_value, actual, predicted in zip(X.ravel(), y, predictions):
        print(
            f"x={x_value:.1f} | "
            f"actual={actual:.2f} | "
            f"predicted={predicted:.2f}"
        )

    print(f"\nMAE:  {mae(y, predictions):.4f}")
    print(f"MSE:  {mse(y, predictions):.4f}")
    print(f"RMSE: {rmse(y, predictions):.4f}")


def demonstrate_matrix_methods():
    """Compare normal equation, pseudoinverse and SVD solutions."""
    X = np.array([
        [2, 70, 5],
        [4, 80, 7],
        [6, 90, 9],
        [3, 75, 6],
        [8, 95, 10],
        [5, 85, 8],
        [7, 88, 9],
        [1, 65, 4]
    ], dtype=float)

    y = np.array([
        55,
        68,
        82,
        61,
        91,
        75,
        85,
        48
    ], dtype=float)

    beta_normal = normal_equation(X, y)
    beta_pinv = pseudoinverse_solution(X, y)
    beta_svd = svd_solution(X, y)

    predictions_normal = predict(X, beta_normal)
    predictions_pinv = predict(X, beta_pinv)
    predictions_svd = predict(X, beta_svd)

    print("\n=== MATRIX REGRESSION ===")

    print("\nCoefficients:")
    print("Feature order: intercept, hours, attendance, assignments")

    print("\nNormal Equation:")
    print(beta_normal)

    print("\nPseudoinverse:")
    print(beta_pinv)

    print("\nSVD:")
    print(beta_svd)

    print("\nTraining-set metrics:")
    print(
        f"Normal Equation -> "
        f"MAE={mae(y, predictions_normal):.4f}, "
        f"MSE={mse(y, predictions_normal):.4f}"
    )

    print(
        f"Pseudoinverse   -> "
        f"MAE={mae(y, predictions_pinv):.4f}, "
        f"MSE={mse(y, predictions_pinv):.4f}"
    )

    print(
        f"SVD             -> "
        f"MAE={mae(y, predictions_svd):.4f}, "
        f"MSE={mse(y, predictions_svd):.4f}"
    )


def load_student_dataset():
    """Load the repository student-performance dataset."""
    dataset_path = (
        Path(__file__).resolve().parents[2]
        / "datasets"
        / "student_performance.csv"
    )

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}"
        )

    return pd.read_csv(dataset_path)


def demonstrate_repository_dataset():
    """Run matrix-based regression on student performance data."""
    df = load_student_dataset()

    feature_columns = [
        "hours",
        "attendance",
        "assignments"
    ]

    X = df[feature_columns].to_numpy(dtype=float)
    y = df["score"].to_numpy(dtype=float)

    beta = pseudoinverse_solution(X, y)
    predictions = predict(X, beta)

    print("\n=== STUDENT PERFORMANCE DATASET ===")

    print("\nLearned coefficients:")
    for name, value in zip(
        ["intercept"] + feature_columns,
        beta
    ):
        print(f"{name:12s}: {value:.6f}")

    print("\nTraining-set evaluation:")
    print(f"MAE:  {mae(y, predictions):.4f}")
    print(f"MSE:  {mse(y, predictions):.4f}")
    print(f"RMSE: {rmse(y, predictions):.4f}")

    new_student = np.array([
        [6, 90, 9]
    ], dtype=float)

    new_prediction = predict(new_student, beta)[0]

    print("\nNew student:")
    print("Hours=6, Attendance=90, Assignments=9")
    print(f"Predicted score: {new_prediction:.2f}")


if __name__ == "__main__":
    demonstrate_simple_regression()
    demonstrate_matrix_methods()
    demonstrate_repository_dataset()
