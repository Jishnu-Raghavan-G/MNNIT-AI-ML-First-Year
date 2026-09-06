from collections import Counter
from math import sqrt

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ============================================================
# K-NEAREST NEIGHBORS FROM SCRATCH
# ============================================================

class KNNClassifier:
    """
    K-Nearest Neighbors classifier implemented from scratch.

    Basic procedure:
        1. Calculate distance from the test point
           to every training point.
        2. Select the K nearest points.
        3. Use majority voting.
    """

    def __init__(self, k=5, distance_metric="euclidean"):
        if k <= 0:
            raise ValueError("k must be greater than 0.")

        if distance_metric not in {"euclidean", "manhattan"}:
            raise ValueError(
                "distance_metric must be 'euclidean' or 'manhattan'."
            )

        self.k = k
        self.distance_metric = distance_metric
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        X = list(X)
        y = list(y)

        if len(X) != len(y):
            raise ValueError(
                "X and y must contain the same number of samples."
            )

        if self.k > len(X):
            raise ValueError(
                "k cannot be greater than the number of training samples."
            )

        self.X_train = X
        self.y_train = y

        return self

    def _distance(self, point_a, point_b):
        if self.distance_metric == "euclidean":
            return sqrt(
                sum(
                    (a - b) ** 2
                    for a, b in zip(point_a, point_b)
                )
            )

        return sum(
            abs(a - b)
            for a, b in zip(point_a, point_b)
        )

    def _nearest_neighbors(self, test_point):
        distances = []

        for training_point, label in zip(
            self.X_train,
            self.y_train,
        ):
            distance = self._distance(
                test_point,
                training_point,
            )

            distances.append(
                (distance, label)
            )

        distances.sort(
            key=lambda item: item[0]
        )

        return distances[:self.k]

    def predict_one(self, test_point):
        neighbors = self._nearest_neighbors(
            test_point
        )

        labels = [
            label
            for _, label in neighbors
        ]

        votes = Counter(labels)

        return votes.most_common(1)[0][0]

    def predict(self, X):
        return [
            self.predict_one(test_point)
            for test_point in X
        ]


# ============================================================
# SIMPLE KNN EXAMPLE
# ============================================================

X_simple = [
    [1, 1],
    [2, 2],
    [3, 3],
    [6, 6],
]

y_simple = [
    "A",
    "A",
    "B",
    "B",
]


simple_model = KNNClassifier(
    k=3,
    distance_metric="euclidean",
)

simple_model.fit(
    X_simple,
    y_simple,
)


new_point = [
    [2, 3]
]


simple_prediction = simple_model.predict(
    new_point
)[0]


print("=" * 60)
print("SIMPLE KNN EXAMPLE")
print("=" * 60)

print("\nTraining Points:")

for point, label in zip(
    X_simple,
    y_simple,
):
    print(
        f"{point} -> {label}"
    )


print("\nNew Point:")
print(new_point[0])


print("\nK:")
print(simple_model.k)


print("\nDistance Metric:")
print(simple_model.distance_metric)


print("\nPrediction:")
print(simple_prediction)


print("\nNearest Neighbors:")

neighbors = simple_model._nearest_neighbors(
    new_point[0]
)

for distance, label in neighbors:
    print(
        f"Distance = {distance:.4f}, "
        f"Class = {label}"
    )


# ============================================================
# STUDENT PERFORMANCE DATASET
# ============================================================

print("\n" + "=" * 60)
print("KNN ON STUDENT PERFORMANCE DATA")
print("=" * 60)


data = pd.read_csv(
    "datasets/student_performance.csv"
)


print("\nDataset:")
print(data)


# ------------------------------------------------------------
# Create a classification target
# ------------------------------------------------------------
# The original score is continuous.
# We create a simple Pass/Fail target only for demonstrating
# KNN classification.

PASS_THRESHOLD = 60

data["result"] = (
    data["score"] >= PASS_THRESHOLD
).map(
    {
        True: "Pass",
        False: "Fail",
    }
)


features = [
    "hours",
    "attendance",
    "assignments",
]


X = data[features]
y = data["result"]


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
# TRAIN KNN FROM SCRATCH
# ============================================================

k = 5

model = KNNClassifier(
    k=k,
    distance_metric="euclidean",
)


model.fit(
    X_train_scaled.tolist(),
    y_train.tolist(),
)


# ============================================================
# PREDICTION
# ============================================================

predictions = model.predict(
    X_test_scaled.tolist()
)


# ============================================================
# EVALUATION
# ============================================================

accuracy = accuracy_score(
    y_test,
    predictions,
)


print("\nFeatures:")
print(features)


print("\nTarget:")
print("result")


print("\nPass Threshold:")
print(PASS_THRESHOLD)


print("\nK:")
print(k)


print("\nAccuracy:")
print(f"{accuracy:.4f}")


print("\nClassification Report:")

print(
    classification_report(
        y_test,
        predictions,
        zero_division=0,
    )
)


# ============================================================
# SHOW PREDICTIONS
# ============================================================

results = X_test.copy()

results["actual"] = y_test.values

results["predicted"] = predictions

print("\nPredictions:")

print(results.reset_index(drop=True))


# ============================================================
# PREDICT A NEW STUDENT
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


new_student_scaled = scaler.transform(
    new_student[features]
)


new_prediction = model.predict(
    new_student_scaled.tolist()
)[0]


print("\n" + "=" * 60)
print("NEW STUDENT PREDICTION")
print("=" * 60)


print("\nNew Student:")
print(new_student)


print("\nPrediction:")
print(new_prediction)


# ============================================================
# DISTANCE METRIC COMPARISON
# ============================================================

print("\n" + "=" * 60)
print("EUCLIDEAN vs MANHATTAN")
print("=" * 60)


euclidean_model = KNNClassifier(
    k=5,
    distance_metric="euclidean",
)

manhattan_model = KNNClassifier(
    k=5,
    distance_metric="manhattan",
)


euclidean_model.fit(
    X_train_scaled.tolist(),
    y_train.tolist(),
)


manhattan_model.fit(
    X_train_scaled.tolist(),
    y_train.tolist(),
)


euclidean_predictions = (
    euclidean_model.predict(
        X_test_scaled.tolist()
    )
)


manhattan_predictions = (
    manhattan_model.predict(
        X_test_scaled.tolist()
    )
)


euclidean_accuracy = accuracy_score(
    y_test,
    euclidean_predictions,
)


manhattan_accuracy = accuracy_score(
    y_test,
    manhattan_predictions,
)


print(
    f"\nEuclidean Accuracy: "
    f"{euclidean_accuracy:.4f}"
)


print(
    f"Manhattan Accuracy: "
    f"{manhattan_accuracy:.4f}"
)
