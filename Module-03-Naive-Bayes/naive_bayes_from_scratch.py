from collections import Counter, defaultdict
from math import prod


class CategoricalNaiveBayes:
    """
    Categorical Naive Bayes implemented from scratch.

    Uses:
    P(C | F) ∝ P(C) × Π P(fi | C)

    Laplace smoothing is used to avoid zero probabilities.
    """

    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.classes = []
        self.class_counts = Counter()
        self.feature_counts = defaultdict(Counter)
        self.feature_values = defaultdict(set)
        self.total_samples = 0

    def fit(self, X, y):
        self.classes = sorted(set(y))
        self.total_samples = len(y)

        for row, target in zip(X, y):
            self.class_counts[target] += 1

            for feature_index, value in enumerate(row):
                self.feature_counts[
                    (target, feature_index)
                ][value] += 1

                self.feature_values[feature_index].add(value)

        return self

    def prior_probability(self, target):
        return self.class_counts[target] / self.total_samples

    def conditional_probability(
        self,
        feature_index,
        value,
        target,
    ):
        count = self.feature_counts[
            (target, feature_index)
        ][value]

        class_count = self.class_counts[target]

        number_of_values = len(
            self.feature_values[feature_index]
        )

        numerator = count + self.alpha

        denominator = (
            class_count
            + self.alpha * number_of_values
        )

        return numerator / denominator

    def class_probability(self, row, target):
        probability = self.prior_probability(target)

        for feature_index, value in enumerate(row):
            probability *= self.conditional_probability(
                feature_index,
                value,
                target,
            )

        return probability

    def predict_proba(self, X):
        probabilities = []

        for row in X:
            class_scores = {
                target: self.class_probability(
                    row,
                    target,
                )
                for target in self.classes
            }

            total = sum(class_scores.values())

            normalized = {
                target: score / total
                for target, score in class_scores.items()
            }

            probabilities.append(normalized)

        return probabilities

    def predict(self, X):
        predictions = []

        for row in X:
            class_scores = {
                target: self.class_probability(
                    row,
                    target,
                )
                for target in self.classes
            }

            prediction = max(
                class_scores,
                key=class_scores.get,
            )

            predictions.append(prediction)

        return predictions


# ============================================================
# BUY COMPUTER EXAMPLE
# ============================================================

X = [
    ["<=30", "high", "no", "fair"],
    ["<=30", "high", "no", "excellent"],
    ["31-40", "high", "no", "fair"],
    [">40", "medium", "no", "fair"],
    [">40", "low", "yes", "fair"],
    [">40", "low", "yes", "excellent"],
    ["31-40", "low", "yes", "excellent"],
    ["<=30", "medium", "no", "fair"],
    ["<=30", "low", "yes", "fair"],
    [">40", "medium", "yes", "fair"],
    ["<=30", "medium", "yes", "excellent"],
    ["31-40", "medium", "no", "excellent"],
    ["31-40", "high", "yes", "fair"],
    [">40", "medium", "no", "excellent"],
]

y = [
    "no",
    "no",
    "yes",
    "yes",
    "yes",
    "no",
    "yes",
    "no",
    "yes",
    "yes",
    "yes",
    "yes",
    "yes",
    "no",
]


model = CategoricalNaiveBayes(alpha=1.0)

model.fit(X, y)


new_sample = [
    ["<=30", "medium", "yes", "fair"]
]


print("=" * 60)
print("CATEGORICAL NAIVE BAYES FROM SCRATCH")
print("=" * 60)


# ------------------------------------------------------------
# Class Priors
# ------------------------------------------------------------

print("\nClass Priors:")

for target in model.classes:
    print(
        f"P({target}) = "
        f"{model.prior_probability(target):.6f}"
    )


# ------------------------------------------------------------
# Class Scores
# ------------------------------------------------------------

print("\nClass Scores:")

for target in model.classes:

    score = model.class_probability(
        new_sample[0],
        target,
    )

    print(
        f"P({target} | features) ∝ "
        f"{score:.8f}"
    )


# ------------------------------------------------------------
# Posterior Probabilities
# ------------------------------------------------------------

print("\nPosterior Probabilities:")

probabilities = model.predict_proba(
    new_sample
)[0]

for target, probability in probabilities.items():
    print(
        f"P({target} | features) = "
        f"{probability:.6f}"
    )


# ------------------------------------------------------------
# Prediction
# ------------------------------------------------------------

prediction = model.predict(
    new_sample
)[0]

print("\nNew Sample:")
print(new_sample[0])

print("\nPrediction:")
print(prediction)


# ============================================================
# MANUAL PROBABILITY BREAKDOWN
# ============================================================

print("\n" + "=" * 60)
print("MANUAL PROBABILITY BREAKDOWN")
print("=" * 60)


for target in model.classes:

    print(f"\nClass: {target}")

    prior = model.prior_probability(target)

    print(f"Prior = {prior:.6f}")

    likelihoods = []

    for feature_index, value in enumerate(
        new_sample[0]
    ):

        probability = model.conditional_probability(
            feature_index,
            value,
            target,
        )

        likelihoods.append(probability)

        print(
            f"P(feature {feature_index + 1} = "
            f"{value} | {target}) = "
            f"{probability:.6f}"
        )

    score = prior * prod(likelihoods)

    print(f"Final Score = {score:.8f}")


# ============================================================
# PLAY GOLF EXAMPLE
# ============================================================

golf_X = [
    ["Rainy", "Hot", "High", "False"],
    ["Rainy", "Hot", "High", "True"],
    ["Overcast", "Hot", "High", "False"],
    ["Sunny", "Mild", "High", "False"],
    ["Sunny", "Cool", "Normal", "False"],
    ["Sunny", "Cool", "Normal", "True"],
    ["Overcast", "Cool", "Normal", "True"],
    ["Rainy", "Mild", "High", "False"],
    ["Rainy", "Cool", "Normal", "False"],
    ["Sunny", "Mild", "Normal", "False"],
    ["Rainy", "Mild", "Normal", "True"],
    ["Overcast", "Mild", "High", "True"],
    ["Overcast", "Hot", "Normal", "False"],
    ["Sunny", "Mild", "High", "True"],
]

golf_y = [
    "No",
    "No",
    "Yes",
    "Yes",
    "Yes",
    "No",
    "Yes",
    "No",
    "Yes",
    "Yes",
    "Yes",
    "Yes",
    "Yes",
    "No",
]


golf_model = CategoricalNaiveBayes(
    alpha=1.0
)

golf_model.fit(
    golf_X,
    golf_y,
)


golf_sample = [
    ["Rainy", "Cool", "High", "True"]
]


print("\n" + "=" * 60)
print("PLAY GOLF NAIVE BAYES EXAMPLE")
print("=" * 60)


golf_probabilities = (
    golf_model.predict_proba(
        golf_sample
    )[0]
)


print("\nPosterior Probabilities:")

for target, probability in (
    golf_probabilities.items()
):
    print(
        f"P({target} | features) = "
        f"{probability:.6f}"
    )


golf_prediction = golf_model.predict(
    golf_sample
)[0]


print("\nNew Sample:")
print(golf_sample[0])

print("\nPrediction:")
print(golf_prediction)
