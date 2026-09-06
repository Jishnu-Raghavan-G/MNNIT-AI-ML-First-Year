import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# ============================================================
# SVM MARGIN DEMONSTRATION
# ============================================================

def decision_score(
    X,
    weights,
    bias,
):
    """
    Calculate the linear SVM decision score.

        f(x) = w^T x + b
    """

    X = np.asarray(
        X,
        dtype=float,
    )

    weights = np.asarray(
        weights,
        dtype=float,
    )

    return (
        X @ weights
        + bias
    )


def predict_from_score(scores):
    """
    Convert SVM decision scores into
    binary labels {-1, +1}.
    """

    scores = np.asarray(
        scores,
        dtype=float,
    )

    return np.where(
        scores >= 0,
        1,
        -1,
    )


def distance_from_boundary(
    X,
    weights,
    bias,
):
    """
    Distance from each point to the
    hyperplane:

        w^T x + b = 0

    Formula:

        |w^T x + b| / ||w||
    """

    scores = decision_score(
        X,
        weights,
        bias,
    )

    norm = np.linalg.norm(
        weights
    )

    if norm == 0:
        raise ValueError(
            "Weight vector cannot have zero magnitude."
        )

    return (
        np.abs(scores)
        / norm
    )


def hinge_loss(
    y,
    scores,
):
    """
    Calculate average hinge loss.

        L = max(0, 1 - y*f(x))
    """

    y = np.asarray(
        y,
        dtype=float,
    )

    scores = np.asarray(
        scores,
        dtype=float,
    )

    return np.mean(
        np.maximum(
            0,
            1 - y * scores,
        )
    )


# ============================================================
# SIMPLE GEOMETRIC EXAMPLE
# ============================================================

print("=" * 60)
print("SVM MARGIN DEMONSTRATION")
print("=" * 60)


# Example hyperplane:
#
#     x1 + x2 - 5 = 0
#
# Therefore:
#
#     w = [1, 1]
#     b = -5

weights = np.array(
    [1.0, 1.0]
)

bias = -5.0


X_simple = np.array(
    [
        [1.0, 1.0],
        [2.0, 2.0],
        [2.0, 4.0],
        [3.0, 3.0],
        [4.0, 4.0],
        [5.0, 5.0],
    ]
)


scores = decision_score(
    X_simple,
    weights,
    bias,
)


predictions = (
    predict_from_score(
        scores
    )
)


distances = (
    distance_from_boundary(
        X_simple,
        weights,
        bias,
    )
)


print("\nHyperplane:")

print(
    "x1 + x2 - 5 = 0"
)


print("\nWeights:")

print(weights)


print("\nBias:")

print(bias)


print("\nMargin boundaries:")

print(
    "x1 + x2 - 5 = +1"
)

print(
    "x1 + x2 - 5 = -1"
)


margin_width = (
    2 / np.linalg.norm(
        weights
    )
)


print("\nWeight magnitude:")

print(
    f"{np.linalg.norm(weights):.6f}"
)


print("\nTotal margin width:")

print(
    f"{margin_width:.6f}"
)


print("\nPoint Analysis:")

for point, score, prediction, distance in zip(
    X_simple,
    scores,
    predictions,
    distances,
):

    print(
        f"Point = {point} | "
        f"Score = {score:.4f} | "
        f"Class = {prediction:+d} | "
        f"Distance = {distance:.4f}"
    )


# ============================================================
# HINGE LOSS DEMONSTRATION
# ============================================================

print("\n" + "=" * 60)
print("HINGE LOSS")
print("=" * 60)


y_hinge = np.array(
    [
        1,
        1,
        1,
        -1,
        -1,
        -1,
    ]
)


scores_hinge = np.array(
    [
        2.0,
        0.4,
        -0.5,
        -2.0,
        -0.4,
        0.5,
    ]
)


losses = np.maximum(
    0,
    1 - y_hinge * scores_hinge,
)


for y_value, score, loss in zip(
    y_hinge,
    scores_hinge,
    losses,
):

    print(
        f"y = {y_value:+d} | "
        f"score = {score:+.2f} | "
        f"hinge loss = {loss:.4f}"
    )


print("\nAverage Hinge Loss:")

print(
    f"{hinge_loss(y_hinge, scores_hinge):.4f}"
)


# ============================================================
# STUDENT PERFORMANCE DATASET
# ============================================================

print("\n" + "=" * 60)
print("SVM ON STUDENT PERFORMANCE DATA")
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

X_train_scaled = (
    scaler.fit_transform(
        X_train
    )
)

X_test_scaled = (
    scaler.transform(
        X_test
    )
)


# ============================================================
# LINEAR SVM
# ============================================================

linear_svm = SVC(
    kernel="linear",
    C=1.0,
)


linear_svm.fit(
    X_train_scaled,
    y_train,
)


linear_predictions = (
    linear_svm.predict(
        X_test_scaled
    )
)


linear_accuracy = (
    accuracy_score(
        y_test,
        linear_predictions,
    )
)


print("\n" + "=" * 60)
print("LINEAR SVM")
print("=" * 60)


print("\nKernel:")

print("linear")


print("\nC:")

print(
    linear_svm.C
)


print("\nAccuracy:")

print(
    f"{linear_accuracy:.4f}"
)


print("\nClassification Report:")

print(
    classification_report(
        y_test,
        linear_predictions,
        zero_division=0,
    )
)


# ============================================================
# SUPPORT VECTORS
# ============================================================

print("\nNumber of Support Vectors:")

print(
    linear_svm.n_support_
)


print("\nTotal Support Vectors:")

print(
    len(
        linear_svm.support_
    )
)


# ============================================================
# LINEAR SVM COEFFICIENTS
# ============================================================

print("\nLinear SVM Coefficients:")

for feature, coefficient in zip(
    features,
    linear_svm.coef_[0],
):

    print(
        f"{feature}: "
        f"{coefficient:.6f}"
    )


print("\nLinear SVM Intercept:")

print(
    f"{linear_svm.intercept_[0]:.6f}"
)


# ============================================================
# NEW STUDENT
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


new_prediction = (
    linear_svm.predict(
        new_student_scaled
    )[0]
)


new_score = (
    linear_svm.decision_function(
        new_student_scaled
    )[0]
)


print("\n" + "=" * 60)
print("NEW STUDENT SVM PREDICTION")
print("=" * 60)


print("\nNew Student:")

print(new_student)


print("\nDecision Score:")

print(
    f"{new_score:.6f}"
)


print("\nPrediction:")

if new_prediction == 1:
    print("Pass")
else:
    print("Fail")


# ============================================================
# RBF SVM
# ============================================================

print("\n" + "=" * 60)
print("RBF SVM")
print("=" * 60)


rbf_svm = SVC(
    kernel="rbf",
    C=1.0,
    gamma="scale",
)


rbf_svm.fit(
    X_train_scaled,
    y_train,
)


rbf_predictions = (
    rbf_svm.predict(
        X_test_scaled
    )
)


rbf_accuracy = (
    accuracy_score(
        y_test,
        rbf_predictions,
    )
)


print("\nKernel:")

print("rbf")


print("\nC:")

print(
    rbf_svm.C
)


print("\nGamma:")

print(
    rbf_svm.gamma
)


print("\nAccuracy:")

print(
    f"{rbf_accuracy:.4f}"
)


print("\nClassification Report:")

print(
    classification_report(
        y_test,
        rbf_predictions,
        zero_division=0,
    )
)


# ============================================================
# C COMPARISON
# ============================================================

print("\n" + "=" * 60)
print("EFFECT OF C")
print("=" * 60)


for C in [
    0.01,
    0.1,
    1.0,
    10.0,
    100.0,
]:

    model = SVC(
        kernel="linear",
        C=C,
    )

    model.fit(
        X_train_scaled,
        y_train,
    )

    predictions = model.predict(
        X_test_scaled
    )

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    print(
        f"C = {C:<6} | "
        f"Accuracy = {accuracy:.4f} | "
        f"Support Vectors = "
        f"{len(model.support_)}"
    )


# ============================================================
# GAMMA COMPARISON
# ============================================================

print("\n" + "=" * 60)
print("EFFECT OF GAMMA")
print("=" * 60)


for gamma in [
    0.01,
    0.1,
    1.0,
    10.0,
]:

    model = SVC(
        kernel="rbf",
        C=1.0,
        gamma=gamma,
    )

    model.fit(
        X_train_scaled,
        y_train,
    )

    predictions = model.predict(
        X_test_scaled
    )

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    print(
        f"Gamma = {gamma:<5} | "
        f"Accuracy = {accuracy:.4f} | "
        f"Support Vectors = "
        f"{len(model.support_)}"
    )
