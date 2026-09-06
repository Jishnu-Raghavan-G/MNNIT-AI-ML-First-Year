import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def load_dataset():
    """Load the student performance dataset."""
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


def inspect_dataset(df):
    """Display basic information about the dataset."""
    print("\n=== DATASET ===")
    print(df)

    print("\n=== SHAPE ===")
    print(df.shape)

    print("\n=== COLUMNS ===")
    print(list(df.columns))

    print("\n=== MISSING VALUES ===")
    print(df.isnull().sum())

    print("\n=== DUPLICATES ===")
    print(df.duplicated().sum())


def prepare_data(df):
    """Separate features and target."""
    feature_columns = [
        "hours",
        "attendance",
        "assignments"
    ]

    target_column = "score"

    X = df[feature_columns]
    y = df[target_column]

    return X, y, feature_columns


def evaluate_model(y_test, predictions):
    """Calculate and print regression metrics."""
    mae = mean_absolute_error(y_test, predictions)

    mse = mean_squared_error(y_test, predictions)

    rmse = np.sqrt(mse)

    r2 = r2_score(y_test, predictions)

    print("\n=== MODEL EVALUATION ===")
    print(f"MAE:  {mae:.4f}")
    print(f"MSE:  {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²:   {r2:.4f}")

    return mae, mse, rmse, r2


def show_predictions(y_test, predictions):
    """Display actual and predicted values."""
    results = pd.DataFrame({
        "Actual": y_test.to_numpy(),
        "Predicted": predictions
    })

    results["Error"] = (
        results["Actual"] - results["Predicted"]
    )

    print("\n=== PREDICTIONS ===")
    print(results.to_string(index=False))


def predict_new_student(model):
    """Predict the score of a new student."""
    new_student = pd.DataFrame({
        "hours": [6],
        "attendance": [90],
        "assignments": [9]
    })

    prediction = model.predict(new_student)[0]

    print("\n=== NEW STUDENT ===")
    print(new_student.to_string(index=False))

    print(f"\nPredicted score: {prediction:.2f}")


def main():
    print("STUDENT PERFORMANCE PREDICTION")

    # 1. Load dataset
    df = load_dataset()

    # 2. Inspect dataset
    inspect_dataset(df)

    # 3. Prepare features and target
    X, y, feature_columns = prepare_data(df)

    print("\n=== FEATURES ===")
    print(feature_columns)

    print("\n=== TARGET ===")
    print("score")

    # 4. Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    print("\n=== DATA SPLIT ===")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples:  {len(X_test)}")

    # 5. Create model
    model = LinearRegression()

    # 6. Train model
    model.fit(X_train, y_train)

    # 7. Display learned parameters
    print("\n=== LEARNED MODEL ===")
    print(f"Intercept: {model.intercept_:.4f}")

    for feature, coefficient in zip(
        feature_columns,
        model.coef_
    ):
        print(
            f"{feature:12s}: "
            f"{coefficient:.4f}"
        )

    # 8. Predict test data
    predictions = model.predict(X_test)

    # 9. Evaluate
    evaluate_model(y_test, predictions)

    # 10. Display predictions
    show_predictions(y_test, predictions)

    # 11. Predict a new student
    predict_new_student(model)


if __name__ == "__main__":
    main()
