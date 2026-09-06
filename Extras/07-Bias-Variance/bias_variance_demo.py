import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures


def generate_dataset():
    rng = np.random.default_rng(42)

    X = np.linspace(-3, 3, 80).reshape(-1, 1)

    true_function = (
        0.5 * X.ravel() ** 3
        - 2 * X.ravel() ** 2
        + X.ravel()
    )

    noise = rng.normal(0, 2, len(X))

    y = true_function + noise

    return X, y


def create_model(degree):
    model = make_pipeline(
        PolynomialFeatures(
            degree=degree,
            include_bias=False
        ),
        LinearRegression()
    )

    return model


def evaluate_model(
    degree,
    X_train,
    X_test,
    y_train,
    y_test
):
    model = create_model(degree)

    model.fit(X_train, y_train)

    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)

    train_mse = mean_squared_error(
        y_train,
        train_predictions
    )

    test_mse = mean_squared_error(
        y_test,
        test_predictions
    )

    return train_mse, test_mse


def main():

    X, y = generate_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42
    )

    degrees = [1, 2, 3, 5, 10, 15]

    train_errors = []
    test_errors = []

    print("BIAS-VARIANCE EXPERIMENT")
    print()

    for degree in degrees:

        train_mse, test_mse = evaluate_model(
            degree,
            X_train,
            X_test,
            y_train,
            y_test
        )

        train_errors.append(train_mse)
        test_errors.append(test_mse)

        print(
            "Degree:",
            degree,
            "| Train MSE:",
            round(train_mse, 4),
            "| Test MSE:",
            round(test_mse, 4)
        )

    best_index = np.argmin(test_errors)
    best_degree = degrees[best_index]

    print()
    print("BEST TEST PERFORMANCE")
    print("Best polynomial degree:", best_degree)
    print(
        "Best test MSE:",
        round(test_errors[best_index], 4)
    )

    print()
    print("Creating graph...")

    plt.figure(figsize=(10, 6))

    plt.plot(
        degrees,
        train_errors,
        marker="o",
        label="Training MSE"
    )

    plt.plot(
        degrees,
        test_errors,
        marker="o",
        label="Test MSE"
    )

    plt.xlabel("Polynomial Degree")
    plt.ylabel("Mean Squared Error")

    plt.title(
        "Bias-Variance Tradeoff: "
        "Training vs Test Error"
    )

    plt.xticks(degrees)

    plt.grid(True, alpha=0.3)

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        "bias_variance_graph.png",
        dpi=300
    )

    print("Graph saved as bias_variance_graph.png")

    plt.show()


if __name__ == "__main__":
    main()
