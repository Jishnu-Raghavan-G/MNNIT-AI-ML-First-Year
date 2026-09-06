import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ============================================================
# LOGISTIC REGRESSION FROM SCRATCH
# ============================================================

class LogisticRegressionScratch:
    """
    Binary Logistic Regression implemented from scratch.

    Model:

        z = Xw + b

        p = sigmoid(z)

    Training:

        Binary Cross-Entropy Loss
        +
        Gradient Descent

    Optional L2 regularization is supported.
    """

    def __init__(
        self,
        learning_rate=0.01,
        epochs=5000,
        regularization_strength=0.0,
    ):
        if learning_rate <= 0:
            raise ValueError(
                "learning_rate must be greater than 0."
            )

        if epochs <= 0:
            raise ValueError(
                "epochs must be greater than 0."
            )

        if regularization_strength < 0:
            raise ValueError(
                "regularization_strength cannot be negative."
            )

        self.learning_rate = learning_rate
        self.epochs = epochs
        self.regularization_strength = (
            regularization_strength
        )

        self.weights = None
        self.bias = 0.0
        self.loss_history = []

    # --------------------------------------------------------
    # SIGMOID
    # --------------------------------------------------------

    @staticmethod
    def sigmoid(z):
        """
        Numerically stable sigmoid function.
        """

        z = np.asarray(z, dtype=float)

        result = np.empty_like(z)

        positive_mask = z >= 0
        negative_mask = ~positive_mask

        result[positive_mask] = (
            1.0
            / (
                1.0
                + np.exp(
                    -z[positive_mask]
                )
            )
        )

        exp_z = np.exp(
            z[negative_mask]
        )

        result[negative_mask] = (
            exp_z
            / (1.0 + exp_z)
        )

        return result

    # --------------------------------------------------------
    # BINARY CROSS-ENTROPY
    # --------------------------------------------------------

    def _loss(self, y, probabilities):
        epsilon = 1e-15

        probabilities = np.clip(
            probabilities,
            epsilon,
            1 - epsilon,
        )

        cross_entropy = -np.mean(
            y * np.log(probabilities)
            + (1 - y)
            * np.log(1 - probabilities)
        )

        regularization = (
            self.regularization_strength
            * np.sum(self.weights ** 2)
            / (2 * len(y))
        )

        return (
            cross_entropy
            + regularization
        )

    # --------------------------------------------------------
    # FIT
    # --------------------------------------------------------

    def fit(self, X, y):
        X = np.asarray(
            X,
            dtype=float,
        )

        y = np.asarray(
            y,
            dtype=float,
        )

        if X.ndim != 2:
            raise ValueError(
                "X must be a 2-dimensional array."
            )

        if y.ndim != 1:
            raise ValueError(
                "y must be a 1-dimensional array."
            )

        if len(X) != len(y):
            raise ValueError(
                "X and y must contain the same "
                "number of samples."
            )

        unique_classes = set(
            np.unique(y)
        )

        if not unique_classes.issubset(
            {0.0, 1.0}
        ):
            raise ValueError(
                "This implementation supports "
                "binary labels encoded as 0 and 1."
            )

        n_samples, n_features = X.shape

        self.weights = np.zeros(
            n_features,
            dtype=float,
        )

        self.bias = 0.0

        self.loss_history = []

        for epoch in range(
            self.epochs
        ):

            # ------------------------------------------------
            # Forward pass
            # ------------------------------------------------

            linear_output = (
                X @ self.weights
                + self.bias
            )

            probabilities = self.sigmoid(
                linear_output
            )

            # ------------------------------------------------
            # Loss
            # ------------------------------------------------

            loss = self._loss(
                y,
                probabilities,
            )

            self.loss_history.append(
                loss
            )

            # ------------------------------------------------
            # Gradients
            # ------------------------------------------------

            error = (
                probabilities - y
            )

            dw = (
                X.T @ error
                / n_samples
            )

            db = np.mean(
                error
            )

            # L2 regularization gradient
            dw += (
                self.regularization_strength
                * self.weights
                / n_samples
            )

            # ------------------------------------------------
            # Parameter update
            # ------------------------------------------------

            self.weights -= (
                self.learning_rate
                * dw
            )

            self.bias -= (
                self.learning_rate
                * db
            )

        return self

    # --------------------------------------------------------
    # PREDICT PROBABILITY
    # --------------------------------------------------------

    def predict_proba(self, X):
        X = np.asarray(
            X,
            dtype=float,
        )

        linear_output = (
            X @ self.weights
            + self.bias
        )

        probabilities = self.sigmoid(
            linear_output
        )

        return probabilities

    # --------------------------------------------------------
    # PREDICT
    # --------------------------------------------------------

    def predict(
        self,
        X,
        threshold=0.5,
    ):
        probabilities = (
            self.predict_proba(X)
        )

        return (
            probabilities >= threshold
        ).astype(int)


# ============================================================
# SIGMOID DEMONSTRATION
# ============================================================

print("=" * 60)
print("SIGMOID FUNCTION")
print("=" * 60)


test_values = np.array(
    [
        -5,
        -2,
        -1,
        0,
        1,
        2,
        5,
    ],
    dtype=float,
)


test_probabilities = (
    LogisticRegressionScratch.sigmoid(
        test_values
    )
)


print("\nz\t sigmoid(z)")

for z, probability in zip(
    test_values,
    test_probabilities,
):
    print(
        f"{z:>2.0f}\t {probability:.6f}"
    )


# ============================================================
# SIMPLE LOGISTIC REGRESSION EXAMPLE
# ============================================================

print("\n" + "=" * 60)
print("SIMPLE LOGISTIC REGRESSION EXAMPLE")
print("=" * 60)


X_simple = np.array(
    [
        [1.0],
        [2.0],
        [3.0],
        [4.0],
        [5.0],
        [6.0],
        [7.0],
        [8.0],
    ]
)

y_simple = np.array(
    [
        0,
        0,
        0,
        0,
        1,
        1,
        1,
        1,
    ]
)


simple_model = LogisticRegressionScratch(
    learning_rate=0.1,
    epochs=3000,
)


simple_model.fit(
    X_simple,
    y_simple,
)


simple_probabilities = (
    simple_model.predict_proba(
        X_simple
    )
)


simple_predictions = (
    simple_model.predict(
        X_simple
    )
)


print("\nLearned Weight:")
print(
    f"{simple_model.weights[0]:.6f}"
)


print("\nLearned Bias:")
print(
    f"{simple_model.bias:.6f}"
)


print("\nPredictions:")

for x, probability, prediction in zip(
    X_simple.flatten(),
    simple_probabilities,
    simple_predictions,
):
    print(
        f"x = {x:.1f} | "
        f"probability = {probability:.6f} | "
        f"class = {prediction}"
    )


# ============================================================
# STUDENT PERFORMANCE DATASET
# ============================================================

print("\n" + "=" * 60)
print("LOGISTIC REGRESSION ON STUDENT DATA")
print("=" * 60)


data = pd.read_csv(
    "datasets/student_performance.csv"
)


print("\nDataset:")
print(data)


# ------------------------------------------------------------
# Convert continuous score into binary classification
# ------------------------------------------------------------

PASS_THRESHOLD = 60

data["result"] = (
    data["score"] >= PASS_THRESHOLD
).astype(int)


features = [
    "hours",
    "attendance",
    "assignments",
]


X = data[features]

y = data["result"]


print("\nFeatures:")
print(features)


print("\nPass Threshold:")
print(PASS_THRESHOLD)


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = (
    train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
)


# ============================================================
# FEATURE SCALING
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)


# ============================================================
# TRAIN MODEL
# ============================================================

model = LogisticRegressionScratch(
    learning_rate=0.05,
    epochs=5000,
    regularization_strength=0.1,
)


model.fit(
    X_train_scaled,
    y_train.to_numpy(),
)


# ============================================================
# TEST PREDICTIONS
# ============================================================

test_probabilities = (
    model.predict_proba(
        X_test_scaled
    )
)


test_predictions = (
    model.predict(
        X_test_scaled,
        threshold=0.5,
    )
)


# ============================================================
# EVALUATION
# ============================================================

accuracy = accuracy_score(
    y_test,
    test_predictions,
)


matrix = confusion_matrix(
    y_test,
    test_predictions,
)


print("\nLearned Weights:")

for feature, weight in zip(
    features,
    model.weights,
):
    print(
        f"{feature}: "
        f"{weight:.6f}"
    )


print("\nLearned Bias:")
print(
    f"{model.bias:.6f}"
)


print("\nAccuracy:")
print(
    f"{accuracy:.4f}"
)


print("\nConfusion Matrix:")
print(matrix)


print("\nClassification Report:")

print(
    classification_report(
        y_test,
        test_predictions,
        zero_division=0,
    )
)


# ============================================================
# TEST PREDICTIONS TABLE
# ============================================================

results = X_test.copy()

results["actual"] = y_test.values

results["probability_pass"] = (
    test_probabilities
)

results["predicted"] = (
    test_predictions
)


print("\nPredictions:")

print(
    results.reset_index(
        drop=True
    )
)


# ============================================================
# NEW STUDENT PREDICTION
# ============================================================

new_student = pd.DataFrame(
    [
        {
            "hours": 6,
            "attendance": 85,
            "assignments": 8,
        }
    ]
)


new_student_scaled = (
    scaler.transform(
        new_student[features]
    )
)


new_probability = (
    model.predict_proba(
        new_student_scaled
    )[0]
)


new_prediction = (
    model.predict(
        new_student_scaled
    )[0]
)


print("\n" + "=" * 60)
print("NEW STUDENT PREDICTION")
print("=" * 60)


print("\nNew Student:")
print(new_student)


print("\nProbability of Pass:")
print(
    f"{new_probability:.6f}"
)


print("\nPrediction:")

if new_prediction == 1:
    print("Pass")
else:
    print("Fail")


# ============================================================
# THRESHOLD COMPARISON
# ============================================================

print("\n" + "=" * 60)
print("CLASSIFICATION THRESHOLD COMPARISON")
print("=" * 60)


for threshold in [
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
]:

    predictions = (
        test_probabilities
        >= threshold
    ).astype(int)

    threshold_accuracy = (
        accuracy_score(
            y_test,
            predictions,
        )
    )

    print(
        f"Threshold = {threshold:.1f} | "
        f"Accuracy = {threshold_accuracy:.4f}"
    )


# ============================================================
# TRAINING LOSS
# ============================================================

print("\n" + "=" * 60)
print("TRAINING LOSS")
print("=" * 60)


print(
    f"\nInitial Loss: "
    f"{model.loss_history[0]:.6f}"
)


print(
    f"Final Loss: "
    f"{model.loss_history[-1]:.6f}"
)


print(
    f"Total Epochs: "
    f"{model.epochs}"
)
