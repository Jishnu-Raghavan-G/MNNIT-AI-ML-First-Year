from math import exp, pi, sqrt

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


# ============================================================
# GAUSSIAN PROBABILITY DENSITY FUNCTION
# ============================================================

def gaussian_probability(
    x,
    mean,
    standard_deviation,
):
    """
    Calculate the Gaussian probability density.

    Formula:

    P(x | C) =
        1 / (σ * sqrt(2π))
        *
        exp(
            -1/2 * ((x - μ) / σ)^2
        )
    """

    if standard_deviation == 0:
        return 1.0 if x == mean else 0.0

    coefficient = 1 / (
        standard_deviation * sqrt(2 * pi)
    )

    exponent = -0.5 * (
        (x - mean) / standard_deviation
    ) ** 2

    return coefficient * exp(exponent)


# ============================================================
# DATASET
# ============================================================

data = pd.DataFrame(
    {
        "glucose": [
            80,
            85,
            90,
            120,
            125,
            130,
        ],
        "bmi": [
            25.0,
            26.5,
            24.0,
            30.0,
            31.5,
            29.5,
        ],
        "class": [
            "No",
            "No",
            "No",
            "Yes",
            "Yes",
            "Yes",
        ],
    }
)


features = [
    "glucose",
    "bmi",
]

X = data[features]
y = data["class"]


# ============================================================
# CLASS PRIORS
# ============================================================

classes = sorted(
    y.unique()
)


priors = {
    target: (y == target).mean()
    for target in classes
}


# ============================================================
# CLASS-WISE MEAN AND STANDARD DEVIATION
# ============================================================

statistics = {}


for target in classes:

    class_data = X[
        y == target
    ]

    statistics[target] = {}

    for feature in features:

        mean = class_data[
            feature
        ].mean()

        standard_deviation = class_data[
            feature
        ].std(ddof=0)

        statistics[target][feature] = {
            "mean": mean,
            "std": standard_deviation,
        }


# ============================================================
# NEW OBSERVATION
# ============================================================

new_patient = {
    "glucose": 120,
    "bmi": 30.0,
}


# ============================================================
# MANUAL GAUSSIAN NAIVE BAYES
# ============================================================

manual_scores = {}


for target in classes:

    probability = priors[target]

    for feature in features:

        mean = statistics[target][
            feature
        ]["mean"]

        standard_deviation = statistics[
            target
        ][feature]["std"]

        likelihood = gaussian_probability(
            new_patient[feature],
            mean,
            standard_deviation,
        )

        probability *= likelihood

    manual_scores[target] = probability


manual_prediction = max(
    manual_scores,
    key=manual_scores.get,
)


# ============================================================
# DISPLAY MANUAL CALCULATION
# ============================================================

print("=" * 60)
print("GAUSSIAN NAIVE BAYES")
print("=" * 60)


print("\nDataset:")
print(data)


print("\nClass Priors:")

for target, prior in priors.items():

    print(
        f"P({target}) = "
        f"{prior:.6f}"
    )


print("\nClass Statistics:")


for target in classes:

    print(f"\nClass: {target}")

    for feature in features:

        mean = statistics[target][
            feature
        ]["mean"]

        standard_deviation = (
            statistics[target][feature]["std"]
        )

        print(
            f"{feature}: "
            f"mean = {mean:.4f}, "
            f"std = {standard_deviation:.4f}"
        )


print("\nNew Patient:")
print(new_patient)


print("\nManual Gaussian Scores:")

for target, score in manual_scores.items():

    print(
        f"{target}: "
        f"{score:.12f}"
    )


print("\nManual Prediction:")
print(manual_prediction)


# ============================================================
# SCIKIT-LEARN GAUSSIAN NAIVE BAYES
# ============================================================

X_train, X_test, y_train, y_test = (
    train_test_split(
        X,
        y,
        test_size=0.33,
        random_state=42,
        stratify=y,
    )
)


model = GaussianNB()

model.fit(
    X_train,
    y_train,
)


# ============================================================
# PREDICT NEW PATIENT
# ============================================================

new_patient_data = [[
    new_patient["glucose"],
    new_patient["bmi"],
]]


sklearn_prediction = model.predict(
    new_patient_data
)[0]


sklearn_probabilities = (
    model.predict_proba(
        new_patient_data
    )[0]
)


print("\n" + "=" * 60)
print("SCIKIT-LEARN GAUSSIAN NAIVE BAYES")
print("=" * 60)


print("\nClasses:")
print(model.classes_)


print("\nPredicted Class:")
print(sklearn_prediction)


print("\nClass Probabilities:")

for target, probability in zip(
    model.classes_,
    sklearn_probabilities,
):

    print(
        f"P({target} | features) = "
        f"{probability:.6f}"
    )


# ============================================================
# MODEL EVALUATION
# ============================================================

test_predictions = model.predict(
    X_test
)


accuracy = accuracy_score(
    y_test,
    test_predictions,
)


print("\nTest Accuracy:")
print(f"{accuracy:.4f}")


print("\nClassification Report:")

print(
    classification_report(
        y_test,
        test_predictions,
        zero_division=0,
    )
)
