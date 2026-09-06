import numpy as np
import pandas as pd
from pathlib import Path


class NeuralNetwork:
    """
    A small fully connected neural network implemented from scratch.

    Architecture:

        Input -> ReLU hidden layer -> Sigmoid output

    Designed for binary classification.
    """

    def __init__(
        self,
        input_size,
        hidden_size=8,
        learning_rate=0.01,
        epochs=1000,
        random_state=42,
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.random_state = random_state

        rng = np.random.default_rng(random_state)

        # Small random initialization.
        self.W1 = rng.normal(
            0,
            np.sqrt(2 / input_size),
            size=(input_size, hidden_size),
        )

        self.b1 = np.zeros((1, hidden_size))

        self.W2 = rng.normal(
            0,
            np.sqrt(2 / hidden_size),
            size=(hidden_size, 1),
        )

        self.b2 = np.zeros((1, 1))

        self.loss_history = []

    @staticmethod
    def relu(z):
        """
        ReLU activation:
            max(0, z)
        """
        return np.maximum(0, z)

    @staticmethod
    def relu_derivative(z):
        """
        Derivative of ReLU.
        """
        return (z > 0).astype(float)

    @staticmethod
    def sigmoid(z):
        """
        Numerically stable sigmoid.
        """
        z = np.clip(z, -500, 500)

        return 1 / (1 + np.exp(-z))

    @staticmethod
    def binary_cross_entropy(y, predictions):
        """
        Binary cross-entropy loss.
        """

        predictions = np.clip(
            predictions,
            1e-9,
            1 - 1e-9,
        )

        loss = -np.mean(
            y * np.log(predictions)
            + (1 - y) * np.log(1 - predictions)
        )

        return loss

    def forward(self, X):
        """
        Forward propagation.
        """

        self.Z1 = X @ self.W1 + self.b1

        self.A1 = self.relu(self.Z1)

        self.Z2 = self.A1 @ self.W2 + self.b2

        self.A2 = self.sigmoid(self.Z2)

        return self.A2

    def backward(self, X, y):
        """
        Backpropagation for binary cross-entropy
        with sigmoid output.
        """

        m = X.shape[0]

        # Output layer gradient.
        dZ2 = self.A2 - y

        dW2 = (self.A1.T @ dZ2) / m

        db2 = np.sum(dZ2, axis=0, keepdims=True) / m

        # Propagate gradient into hidden layer.
        dA1 = dZ2 @ self.W2.T

        dZ1 = dA1 * self.relu_derivative(self.Z1)

        dW1 = (X.T @ dZ1) / m

        db1 = np.sum(
            dZ1,
            axis=0,
            keepdims=True,
        ) / m

        # Gradient descent update.
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2

        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1

    def fit(self, X, y, verbose=True):
        """
        Train the neural network.
        """

        X = np.asarray(X, dtype=float)

        y = np.asarray(y, dtype=float).reshape(-1, 1)

        for epoch in range(self.epochs):
            predictions = self.forward(X)

            loss = self.binary_cross_entropy(
                y,
                predictions,
            )

            self.backward(X, y)

            self.loss_history.append(loss)

            if verbose and (
                epoch == 0
                or (epoch + 1) % 100 == 0
                or epoch == self.epochs - 1
            ):
                print(
                    f"Epoch {epoch + 1:4d}/{self.epochs} "
                    f"| Loss: {loss:.6f}"
                )

        return self

    def predict_proba(self, X):
        """
        Return predicted probabilities.
        """

        X = np.asarray(X, dtype=float)

        return self.forward(X)

    def predict(self, X, threshold=0.5):
        """
        Convert probabilities into binary classes.
        """

        probabilities = self.predict_proba(X)

        return (
            probabilities >= threshold
        ).astype(int).ravel()


def standardize_train_test(X_train, X_test):
    """
    Standardize using statistics from training data only.
    """

    mean = np.mean(X_train, axis=0)

    std = np.std(X_train, axis=0)

    std = np.where(std == 0, 1, std)

    X_train_scaled = (
        (X_train - mean) / std
    )

    X_test_scaled = (
        (X_test - mean) / std
    )

    return (
        X_train_scaled,
        X_test_scaled,
        mean,
        std,
    )


def train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
):
    """
    Simple train/test split using NumPy.
    """

    rng = np.random.default_rng(random_state)

    indices = np.arange(len(X))

    rng.shuffle(indices)

    test_count = int(
        len(X) * test_size
    )

    test_indices = indices[:test_count]

    train_indices = indices[test_count:]

    return (
        X[train_indices],
        X[test_indices],
        y[train_indices],
        y[test_indices],
    )


def accuracy_score(y_true, y_pred):
    """
    Calculate classification accuracy.
    """

    y_true = np.asarray(y_true).ravel()

    y_pred = np.asarray(y_pred).ravel()

    return np.mean(
        y_true == y_pred
    )


def confusion_matrix_binary(y_true, y_pred):
    """
    Return a binary confusion matrix:

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


def precision_score_binary(y_true, y_pred):
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


def recall_score_binary(y_true, y_pred):
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


def f1_score_binary(y_true, y_pred):
    """
    F1 = harmonic mean of precision and recall.
    """

    precision = precision_score_binary(
        y_true,
        y_pred,
    )

    recall = recall_score_binary(
        y_true,
        y_pred,
    )

    denominator = precision + recall

    if denominator == 0:
        return 0.0

    return (
        2 * precision * recall
    ) / denominator


def load_student_data():
    """
    Load the repository student performance dataset.
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

    return pd.read_csv(dataset_path)


def prepare_student_classification(df):
    """
    Convert student score into a binary target.

    score >= 60 -> 1 (Pass)
    score < 60  -> 0 (Fail)
    """

    feature_columns = [
        "hours",
        "attendance",
        "assignments",
    ]

    X = df[feature_columns].to_numpy(
        dtype=float
    )

    y = (
        df["score"].to_numpy(dtype=float)
        >= 60
    ).astype(int)

    return X, y, feature_columns


def run_student_demo():
    """
    Train a neural network on student performance data.
    """

    df = load_student_data()

    X, y, feature_columns = (
        prepare_student_classification(df)
    )

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    (
        X_train,
        X_test,
        train_mean,
        train_std,
    ) = standardize_train_test(
        X_train,
        X_test,
    )

    print("=" * 70)
    print("NEURAL NETWORK FROM SCRATCH")
    print("=" * 70)

    print("\nFeatures:")
    print(feature_columns)

    print("\nArchitecture:")
    print(
        f"{len(feature_columns)} "
        "-> 8 -> 1"
    )

    print("\nTraining samples:")
    print(len(X_train))

    print("\nTest samples:")
    print(len(X_test))

    model = NeuralNetwork(
        input_size=X_train.shape[1],
        hidden_size=8,
        learning_rate=0.05,
        epochs=1000,
        random_state=42,
    )

    print("\nTraining...\n")

    model.fit(
        X_train,
        y_train,
        verbose=True,
    )

    train_predictions = model.predict(
        X_train
    )

    test_predictions = model.predict(
        X_test
    )

    train_accuracy = accuracy_score(
        y_train,
        train_predictions,
    )

    test_accuracy = accuracy_score(
        y_test,
        test_predictions,
    )

    test_precision = precision_score_binary(
        y_test,
        test_predictions,
    )

    test_recall = recall_score_binary(
        y_test,
        test_predictions,
    )

    test_f1 = f1_score_binary(
        y_test,
        test_predictions,
    )

    print("\n" + "=" * 70)
    print("EVALUATION")
    print("=" * 70)

    print(
        f"\nTraining accuracy: "
        f"{train_accuracy:.4f}"
    )

    print(
        f"Test accuracy: "
        f"{test_accuracy:.4f}"
    )

    print(
        f"Test precision: "
        f"{test_precision:.4f}"
    )

    print(
        f"Test recall: "
        f"{test_recall:.4f}"
    )

    print(
        f"Test F1 score: "
        f"{test_f1:.4f}"
    )

    print("\nConfusion matrix:")
    print(
        confusion_matrix_binary(
            y_test,
            test_predictions,
        )
    )

    # Example new student.
    new_student = np.array(
        [
            [8, 90, 85]
        ],
        dtype=float,
    )

    new_student_scaled = (
        new_student - train_mean
    ) / train_std

    probability = model.predict_proba(
        new_student_scaled
    )[0, 0]

    prediction = int(
        probability >= 0.5
    )

    print("\n" + "=" * 70)
    print("NEW STUDENT PREDICTION")
    print("=" * 70)

    print("\nInput:")
    print(
        "Hours = 8, "
        "Attendance = 90, "
        "Assignments = 85"
    )

    print(
        f"\nPredicted probability of Pass: "
        f"{probability:.4f}"
    )

    print(
        "Predicted class: "
        f"{'Pass' if prediction == 1 else 'Fail'}"
    )


def xor_demo():
    """
    Demonstrate that a hidden layer can learn
    a nonlinear XOR relationship.

    XOR:
        0 XOR 0 = 0
        0 XOR 1 = 1
        1 XOR 0 = 1
        1 XOR 1 = 0
    """

    X = np.array(
        [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
        ],
        dtype=float,
    )

    y = np.array(
        [0, 1, 1, 0],
        dtype=float,
    )

    model = NeuralNetwork(
        input_size=2,
        hidden_size=8,
        learning_rate=0.1,
        epochs=5000,
        random_state=42,
    )

    print("\n" + "=" * 70)
    print("XOR DEMO")
    print("=" * 70)

    model.fit(
        X,
        y,
        verbose=False,
    )

    probabilities = model.predict_proba(X).ravel()

    predictions = model.predict(X)

    print("\nInput | Probability | Prediction")

    for row, probability, prediction in zip(
        X,
        probabilities,
        predictions,
    ):
        print(
            f"{row.astype(int)} "
            f"| {probability:.4f} "
            f"| {prediction}"
        )


if __name__ == "__main__":
    xor_demo()

    run_student_demo()
