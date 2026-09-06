import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


def generate_dataset():
    """
    Generate a nonlinear dataset with noise.
    """
    rng = np.random.default_rng(42)

    X = np.linspace(-3, 3, 100).reshape(-1, 1)

    y = (
        0.5 * X.ravel() ** 3
        - 2 * X.ravel() ** 2
        + X.ravel()
        + rng.normal(0, 2, len(X))
    )

    return X, y


def create_polynomial_model(
    degree,
    model
):
    """
    Create a polynomial regression pipeline.

    Standardization is included before the
    regression model so coefficient penalties
    are applied on comparable feature scales.
    """
    return Pipeline([
        (
            "polynomial_features",
            PolynomialFeatures(
                degree=degree,
                include_bias=False
            )
        ),
        (
            "scaler",
            StandardScaler()
        ),
        (
            "model",
            model
        )
    ])


def evaluate_model(
    model,
    X_train,
    X_test,
    y_train,
    y_test
):
    """Train and calculate train/test MSE."""
    model.fit(
        X_train,
        y_train
    )

    train_predictions = model.predict(
        X_train
    )

    test_predictions = model.predict(
        X_test
    )

    train_mse = mean_squared_error(
        y_train,
        train_predictions
    )

    test_mse = mean_squared_error(
        y_test,
        test_predictions
    )

    return (
        train_mse,
        test_mse
    )


def compare_regularization_strengths(
    X_train,
    X_test,
    y_train,
    y_test
):
    """
    Compare Ridge models using several
    regularization strengths.
    """
    alphas = [
        0.001,
        0.01,
        0.1,
        1,
        10,
        100
    ]

    rows = []

    for alpha in alphas:
        model = create_polynomial_model(
            degree=10,
            model=Ridge(
                alpha=alpha
            )
        )

        train_mse, test_mse = evaluate_model(
            model,
            X_train,
            X_test,
            y_train,
            y_test
        )

        rows.append({
            "alpha": alpha,
            "train_mse": train_mse,
            "test_mse": test_mse
        })

    return pd.DataFrame(rows)


def compare_model_types(
    X_train,
    X_test,
    y_train,
    y_test
):
    """
    Compare an unregularized model,
    Ridge and Lasso.
    """
    models = {
        "Linear Regression": create_polynomial_model(
            degree=10,
            model=LinearRegression()
        ),
        "Ridge": create_polynomial_model(
            degree=10,
            model=Ridge(alpha=1.0)
        ),
        "Lasso": create_polynomial_model(
            degree=10,
            model=Lasso(
                alpha=0.01,
                max_iter=100000
            )
        )
    }

    rows = []

    for name, model in models.items():
        train_mse, test_mse = evaluate_model(
            model,
            X_train,
            X_test,
            y_train,
            y_test
        )

        rows.append({
            "model": name,
            "train_mse": train_mse,
            "test_mse": test_mse
        })

    return pd.DataFrame(rows)


def inspect_coefficients(model):
    """
    Display coefficients from the final regression
    estimator inside the pipeline.
    """
    estimator = model.named_steps["model"]

    coefficients = estimator.coef_

    print("\n=== COEFFICIENTS ===")

    for index, coefficient in enumerate(
        coefficients,
        start=1
    ):
        print(
            f"Coefficient {index:2d}: "
            f"{coefficient:.6f}"
        )

    print(
        "\nNumber of coefficients: "
        f"{len(coefficients)}"
    )

    print(
        "Number of exact zero coefficients: "
        f"{np.sum(np.isclose(coefficients, 0))}"
    )


def main():
    print("REGULARIZATION DEMONSTRATION")

    # 1. Generate data
    X, y = generate_dataset()

    # 2. Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42
    )

    print("\n=== DATA SPLIT ===")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples:  {len(X_test)}")

    # 3. Compare different regularization strengths
    strength_results = compare_regularization_strengths(
        X_train,
        X_test,
        y_train,
        y_test
    )

    print("\n=== RIDGE REGULARIZATION STRENGTH ===")
    print(
        strength_results.round(4).to_string(
            index=False
        )
    )

    # 4. Find the best alpha according to test MSE
    # This is only for demonstration. In a real workflow,
    # alpha should be selected using validation data or CV,
    # with the test set reserved for final evaluation.
    best_row = strength_results.loc[
        strength_results["test_mse"].idxmin()
    ]

    best_alpha = float(
        best_row["alpha"]
    )

    print(
        "\nBest alpha in this demonstration: "
        f"{best_alpha}"
    )

    # 5. Compare unregularized, Ridge and Lasso
    model_results = compare_model_types(
        X_train,
        X_test,
        y_train,
        y_test
    )

    print("\n=== MODEL COMPARISON ===")
    print(
        model_results.round(4).to_string(
            index=False
        )
    )

    # 6. Train a Ridge model using the selected alpha
    final_ridge = create_polynomial_model(
        degree=10,
        model=Ridge(
            alpha=best_alpha
        )
    )

    final_ridge.fit(
        X_train,
        y_train
    )

    print("\n=== FINAL RIDGE MODEL ===")
    print(
        f"Alpha: {best_alpha}"
    )

    inspect_coefficients(
        final_ridge
    )

    # 7. Explain the experiment
    print("\n=== INTERPRETATION ===")
    print(
        "Increasing regularization generally "
        "shrinks model coefficients."
    )

    print(
        "A suitable regularization strength can "
        "reduce overfitting."
    )

    print(
        "Too much regularization can make the "
        "model too simple and cause underfitting."
    )

    print(
        "\nFor proper model selection, use "
        "validation data or cross-validation "
        "instead of selecting alpha from the "
        "test set."
    )


if __name__ == "__main__":
    main()
