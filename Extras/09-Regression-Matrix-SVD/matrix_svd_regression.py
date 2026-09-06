"""
Extra 09: Regression - Matrix Approach and SVD

This program demonstrates:

1. Linear Regression using the Normal Equation
2. Linear Regression using the Moore-Penrose Pseudo-Inverse
3. Linear Regression using SVD explicitly
4. Singular / rank-deficient matrices
5. MAE, SAE, SSE, MSE and RMSE
6. The SVD regression example from the lecture
"""

import numpy as np


# ============================================================
# 1. NORMAL EQUATION
# ============================================================

def normal_equation(X, y):
    """
    Calculate:

        beta = (X^T X)^(-1) X^T y

    This requires X^T X to be invertible.
    """

    XtX = X.T @ X
    Xty = X.T @ y

    beta = np.linalg.inv(XtX) @ Xty

    return beta


# ============================================================
# 2. PSEUDO-INVERSE SOLUTION
# ============================================================

def pseudoinverse_solution(X, y):
    """
    Calculate:

        beta = X^+ y

    np.linalg.pinv() computes the Moore-Penrose
    pseudo-inverse.
    """

    X_pinv = np.linalg.pinv(X)

    beta = X_pinv @ y

    return beta


# ============================================================
# 3. SVD SOLUTION
# ============================================================

def svd_solution(X, y):
    """
    Calculate:

        X = U Sigma V^T

        X^+ = V Sigma^+ U^T

        beta = V Sigma^+ U^T y

    The pseudo-inverse is constructed explicitly.
    """

    U, singular_values, Vt = np.linalg.svd(
        X,
        full_matrices=False
    )

    # --------------------------------------------------------
    # Construct Sigma+
    # --------------------------------------------------------

    sigma_pinv = np.zeros_like(singular_values)

    tolerance = (
        np.finfo(float).eps
        * max(X.shape)
        * singular_values[0]
    )

    for i, sigma in enumerate(singular_values):

        if sigma > tolerance:
            sigma_pinv[i] = 1.0 / sigma
        else:
            sigma_pinv[i] = 0.0

    Sigma_pinv = np.diag(sigma_pinv)

    # --------------------------------------------------------
    # beta = V Sigma+ U^T y
    # --------------------------------------------------------

    beta = (
        Vt.T
        @ Sigma_pinv
        @ U.T
        @ y
    )

    return beta


# ============================================================
# 4. REGRESSION METRICS
# ============================================================

def regression_metrics(y_true, y_pred):
    """
    Calculate:

        Absolute Error
        SAE
        MAE
        SSE
        MSE
        RMSE
    """

    errors = y_true - y_pred

    absolute_errors = np.abs(errors)
    squared_errors = errors ** 2

    sae = np.sum(absolute_errors)
    mae = np.mean(absolute_errors)

    sse = np.sum(squared_errors)
    mse = np.mean(squared_errors)
    rmse = np.sqrt(mse)

    return {
        "SAE": sae,
        "MAE": mae,
        "SSE": sse,
        "MSE": mse,
        "RMSE": rmse,
    }


def print_metrics(y_true, y_pred):
    metrics = regression_metrics(y_true, y_pred)

    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")


# ============================================================
# 5. LECTURE SVD EXAMPLE
# ============================================================

def lecture_svd_example():

    print("=" * 70)
    print("LECTURE SVD REGRESSION EXAMPLE")
    print("=" * 70)

    X = np.array(
        [
            [1.0, 2.0],
            [1.0, 4.0],
            [1.0, 6.0],
            [1.0, 8.0],
        ]
    )

    y = np.array(
        [3.0, 7.0, 5.0, 10.0]
    )

    print("\nX:")
    print(X)

    print("\ny:")
    print(y)

    # --------------------------------------------------------
    # Step 1: X^T X
    # --------------------------------------------------------

    XtX = X.T @ X

    print("\nStep 1: X^T X")
    print(XtX)

    # --------------------------------------------------------
    # Eigenvalues
    # --------------------------------------------------------

    eigenvalues, eigenvectors = np.linalg.eigh(XtX)

    print("\nEigenvalues of X^T X:")
    print(eigenvalues)

    # --------------------------------------------------------
    # Singular values
    # --------------------------------------------------------

    singular_values_from_eigenvalues = np.sqrt(
        np.maximum(eigenvalues, 0)
    )

    print("\nSingular values from sqrt(eigenvalues):")
    print(singular_values_from_eigenvalues)

    # --------------------------------------------------------
    # Direct SVD
    # --------------------------------------------------------

    U, singular_values, Vt = np.linalg.svd(
        X,
        full_matrices=False
    )

    print("\nU:")
    print(U)

    print("\nSingular values:")
    print(singular_values)

    print("\nSigma:")
    print(np.diag(singular_values))

    print("\nV^T:")
    print(Vt)

    # --------------------------------------------------------
    # Sigma+
    # --------------------------------------------------------

    sigma_pinv = np.array(
        [
            1.0 / value
            if value > 1e-12
            else 0.0
            for value in singular_values
        ]
    )

    Sigma_pinv = np.diag(sigma_pinv)

    print("\nSigma^+:")
    print(Sigma_pinv)

    # --------------------------------------------------------
    # U^T y
    # --------------------------------------------------------

    Uty = U.T @ y

    print("\nU^T y:")
    print(Uty)

    # --------------------------------------------------------
    # Sigma+ U^T y
    # --------------------------------------------------------

    scaled = Sigma_pinv @ Uty

    print("\nSigma^+ U^T y:")
    print(scaled)

    # --------------------------------------------------------
    # beta = V Sigma+ U^T y
    # --------------------------------------------------------

    beta = Vt.T @ scaled

    print("\nFinal beta:")
    print(beta)

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    predictions = X @ beta

    print("\nPredictions:")
    print(predictions)

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    print("\nRegression Metrics:")
    print_metrics(y, predictions)


# ============================================================
# 6. COMPARE THREE SOLUTIONS
# ============================================================

def compare_solutions():

    print("\n" + "=" * 70)
    print("NORMAL EQUATION VS PSEUDO-INVERSE VS SVD")
    print("=" * 70)

    X = np.array(
        [
            [1.0, 2.0],
            [1.0, 4.0],
            [1.0, 6.0],
            [1.0, 8.0],
        ]
    )

    y = np.array(
        [3.0, 7.0, 5.0, 10.0]
    )

    # Normal Equation
    beta_normal = normal_equation(X, y)

    # Pseudo-Inverse
    beta_pinv = pseudoinverse_solution(X, y)

    # Explicit SVD
    beta_svd = svd_solution(X, y)

    print("\nNormal Equation beta:")
    print(beta_normal)

    print("\nPseudo-Inverse beta:")
    print(beta_pinv)

    print("\nSVD beta:")
    print(beta_svd)

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    prediction_normal = X @ beta_normal
    prediction_pinv = X @ beta_pinv
    prediction_svd = X @ beta_svd

    print("\nPredictions using Normal Equation:")
    print(prediction_normal)

    print("\nPredictions using Pseudo-Inverse:")
    print(prediction_pinv)

    print("\nPredictions using SVD:")
    print(prediction_svd)

    print("\nMetrics using SVD solution:")
    print_metrics(y, prediction_svd)


# ============================================================
# 7. SINGULAR MATRIX EXAMPLE
# ============================================================

def singular_matrix_example():

    print("\n" + "=" * 70)
    print("SINGULAR MATRIX EXAMPLE")
    print("=" * 70)

    # The second column is exactly twice the first.
    # Therefore the columns are linearly dependent.

    X = np.array(
        [
            [1.0, 2.0],
            [2.0, 4.0],
            [3.0, 6.0],
        ]
    )

    y = np.array(
        [2.0, 4.0, 6.0]
    )

    print("\nX:")
    print(X)

    XtX = X.T @ X

    print("\nX^T X:")
    print(XtX)

    determinant = np.linalg.det(XtX)

    print(
        f"\ndet(X^T X): {determinant:.6f}"
    )

    print("\nAttempting Normal Equation:")

    try:

        beta_normal = normal_equation(X, y)

        print("Normal Equation solution:")
        print(beta_normal)

    except np.linalg.LinAlgError:

        print(
            "X^T X is singular."
        )

        print(
            "The ordinary inverse does not exist."
        )

    # --------------------------------------------------------
    # Pseudo-Inverse
    # --------------------------------------------------------

    beta_pinv = pseudoinverse_solution(X, y)

    print("\nPseudo-Inverse solution:")
    print(beta_pinv)

    # --------------------------------------------------------
    # SVD
    # --------------------------------------------------------

    beta_svd = svd_solution(X, y)

    print("\nSVD solution:")
    print(beta_svd)

    predictions = X @ beta_svd

    print("\nPredictions using SVD:")
    print(predictions)


# ============================================================
# 8. MAE VS SSE/MSE EXAMPLE
# ============================================================

def error_metric_example():

    print("\n" + "=" * 70)
    print("MAE VS SSE/MSE")
    print("=" * 70)

    # Lecture example:
    #
    # Errors:
    # -10, -10, -10, -10, 100

    errors = np.array(
        [-10.0, -10.0, -10.0, -10.0, 100.0]
    )

    # Construct y_true and y_pred such that:
    #
    # y_true - y_pred = errors

    y_true = np.zeros_like(errors)
    y_pred = -errors

    print("\nErrors:")
    print(errors)

    print("\nAbsolute errors:")
    print(np.abs(errors))

    print("\nSquared errors:")
    print(errors ** 2)

    metrics = regression_metrics(
        y_true,
        y_pred
    )

    print("\nMetrics:")

    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    print("\nObservation:")
    print(
        "The error of 100 contributes 10000 to SSE "
        "because the error is squared."
    )


# ============================================================
# 9. SIMPLE REGRESSION DEMO
# ============================================================

def simple_regression_demo():

    print("\n" + "=" * 70)
    print("SIMPLE MATRIX REGRESSION DEMO")
    print("=" * 70)

    # x = input feature
    # y = target

    x = np.array(
        [1.0, 2.0, 3.0, 4.0, 5.0]
    )

    y = np.array(
        [3.0, 5.0, 7.0, 9.0, 11.0]
    )

    # Add a column of ones for the intercept.
    X = np.column_stack(
        [
            np.ones(len(x)),
            x
        ]
    )

    print("\nDesign matrix X:")
    print(X)

    print("\ny:")
    print(y)

    beta = normal_equation(X, y)

    print("\nBeta:")
    print(beta)

    predictions = X @ beta

    print("\nPredictions:")
    print(predictions)

    print("\nMetrics:")
    print_metrics(y, predictions)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    lecture_svd_example()

    compare_solutions()

    singular_matrix_example()

    error_metric_example()

    simple_regression_demo()
