import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures


def generate_dataset():
    """
    Generate a nonlinear dataset with noise.
    """
    rng = np.random.default_rng(42)

    X = np.linspace(-3, 3, 80).reshape(-1, 1)

    true_function = (
        0.5 * X.ravel() ** 3
        - 2 * X.ravel() ** 2
        + X.ravel()
    )

    noise = rng.normal(
        loc=0.0,
        scale=2.0,
        size=len(X)
    )

    y = true_function + noise

    return X, y


def create_model(degree):
    """
    Create polynomial regression model
    of the requested degree.
    """
    return make_pipeline(
        PolynomialFeatures(
            degree=degree,
            include_bias=False
        ),
        LinearRegression()
    )


def evaluate_degree(
    degree,
    X_train,
    X_test,
    y_train,
    y_test
):
    """Train and evaluate one polynomial degree."""
    model = create_model(degree)

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

    return model, train_mse, test_mse


def classify_fit(train_mse, test_mse):
    """
    Provide a simple qualitative interpretation.

    This is a teaching heuristic, not a statistical test.
    """
    if train_mse > test_mse * 1.5:
        return "Possible underfitting"

    if test_mse > train_mse * 2:
        return "Possible overfitting"

    return "Reasonable generalization"


def run_experiment():
    """Compare models with different complexities."""
    X, y = generate_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42
    )

    degrees = [
        1,
        2,
        3,
        5,
        10,
        15
    ]

    results = []

    print("=== BIAS-VARIANCE EXPERIMENT ===")

    print(
        "\nComparing polynomial models "
        "of different degrees."
    )

    print(
        "\n"
        f"{'Degree':>8}"
        f"{'Train MSE':>15}"
        f"{'Test MSE':>15}"
        f"{'Interpretation':>25}"
    )

    print("-" * 63)

    for degree in degrees:
        _, train_mse, test_mse = evaluate_degree(
            degree,
            X_train,
            X_test,
            y_train,
            y_test
        )

        interpretation = classify_fit(
            train_mse,
            test_mse
        )

        results.append({
            "degree": degree,
            "train_mse": train_mse,
            "test_mse": test_mse
        })

        print(
            f"{degree:>8}"
            f"{train_mse:>15.4f}"
            f"{test_mse:>15.4f}"
            f"{interpretation:>25}"
        )

    return results


def find_best_test_model(results):
    """Find the model with the lowest test MSE."""
    best = min(
        results,
        key=lambda item: item["test_mse"]
    )

    print("\n=== BEST TEST PERFORMANCE ===")
    print(
        f"Polynomial degree: "
        f"{best['degree']}"
    )
    print(
        f"Test MSE: "
        f"{best['test_mse']:.4f}"
    )

    return best


def explain_results(results):
    """Print the main bias-variance interpretation."""
    print("\n=== INTERPRETATION ===")

    lowest_train = min(
        results,
        key=lambda item: item["train_mse"]
    )

    lowest_test = min(
        results,
        key=lambda item: item["test_mse"]
    )

    print(
        "Lowest training MSE:"
        f" degree {lowest_train['degree']}"
    )

    print(
        "Lowest test MSE:"
        f" degree {lowest_test['degree']}"
    )

    print(
        "\nAs model complexity increases, "
        "training error can decrease."
    )

    print(
        "However, excessive complexity can "
        "increase test error because the model "
        "may begin fitting noise."
    )

    print(
        "\nThe goal is good generalization, "
        "not simply the lowest training error."
    )


def main():
    results = run_experiment()

    find_best_test_model(results)

    explain_results(results)


if __name__ == "__main__":
    main()
