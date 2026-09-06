from collections import Counter
from dataclasses import dataclass

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
)
from sklearn.model_selection import train_test_split


# ============================================================
# TREE NODE
# ============================================================

@dataclass
class Node:
    feature_index: int = None
    threshold: float = None
    left: object = None
    right: object = None
    prediction: object = None

    @property
    def is_leaf(self):
        return self.prediction is not None


# ============================================================
# DECISION TREE CLASSIFIER FROM SCRATCH
# ============================================================

class DecisionTreeClassifierScratch:
    """
    A simple binary Decision Tree classifier implemented
    from scratch using Gini impurity.

    Supported input:
        Numerical features

    Main idea:
        Find the split that produces the lowest weighted
        Gini impurity and recursively build the tree.
    """

    def __init__(
        self,
        max_depth=4,
        min_samples_split=2,
        min_samples_leaf=1,
    ):
        if max_depth < 1:
            raise ValueError(
                "max_depth must be at least 1."
            )

        if min_samples_split < 2:
            raise ValueError(
                "min_samples_split must be at least 2."
            )

        if min_samples_leaf < 1:
            raise ValueError(
                "min_samples_leaf must be at least 1."
            )

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.root = None

    # --------------------------------------------------------
    # GINI IMPURITY
    # --------------------------------------------------------

    def _gini(self, y):
        if len(y) == 0:
            return 0.0

        counts = Counter(y)
        total = len(y)

        impurity = 1.0

        for count in counts.values():
            probability = count / total
            impurity -= probability ** 2

        return impurity

    # --------------------------------------------------------
    # MAJORITY CLASS
    # --------------------------------------------------------

    def _majority_class(self, y):
        return Counter(y).most_common(1)[0][0]

    # --------------------------------------------------------
    # SPLIT DATA
    # --------------------------------------------------------

    def _split(
        self,
        X,
        y,
        feature_index,
        threshold,
    ):
        left_mask = X[:, feature_index] <= threshold
        right_mask = ~left_mask

        left_X = X[left_mask]
        right_X = X[right_mask]

        left_y = y[left_mask]
        right_y = y[right_mask]

        return (
            left_X,
            left_y,
            right_X,
            right_y,
        )

    # --------------------------------------------------------
    # WEIGHTED GINI
    # --------------------------------------------------------

    def _weighted_gini(
        self,
        left_y,
        right_y,
    ):
        total = len(left_y) + len(right_y)

        if total == 0:
            return 0.0

        left_weight = len(left_y) / total
        right_weight = len(right_y) / total

        return (
            left_weight * self._gini(left_y)
            + right_weight * self._gini(right_y)
        )

    # --------------------------------------------------------
    # FIND BEST SPLIT
    # --------------------------------------------------------

    def _best_split(self, X, y):
        n_samples, n_features = X.shape

        best_feature = None
        best_threshold = None
        best_impurity = float("inf")

        for feature_index in range(n_features):

            values = sorted(
                set(
                    X[:, feature_index]
                )
            )

            if len(values) <= 1:
                continue

            thresholds = [
                (values[i] + values[i + 1]) / 2
                for i in range(len(values) - 1)
            ]

            for threshold in thresholds:

                (
                    left_X,
                    left_y,
                    right_X,
                    right_y,
                ) = self._split(
                    X,
                    y,
                    feature_index,
                    threshold,
                )

                if (
                    len(left_y)
                    < self.min_samples_leaf
                    or len(right_y)
                    < self.min_samples_leaf
                ):
                    continue

                impurity = self._weighted_gini(
                    left_y,
                    right_y,
                )

                if impurity < best_impurity:
                    best_impurity = impurity
                    best_feature = feature_index
                    best_threshold = threshold

        return (
            best_feature,
            best_threshold,
            best_impurity,
        )

    # --------------------------------------------------------
    # BUILD TREE
    # --------------------------------------------------------

    def _build_tree(
        self,
        X,
        y,
        depth,
    ):
        n_samples = len(y)

        majority_class = self._majority_class(
            y
        )

        # Stop if:
        # - all samples belong to one class
        # - maximum depth has been reached
        # - too few samples remain for a split
        if (
            len(set(y)) == 1
            or depth >= self.max_depth
            or n_samples < self.min_samples_split
        ):
            return Node(
                prediction=majority_class
            )

        (
            feature_index,
            threshold,
            impurity,
        ) = self._best_split(
            X,
            y,
        )

        if feature_index is None:
            return Node(
                prediction=majority_class
            )

        (
            left_X,
            left_y,
            right_X,
            right_y,
        ) = self._split(
            X,
            y,
            feature_index,
            threshold,
        )

        if (
            len(left_y) < self.min_samples_leaf
            or len(right_y) < self.min_samples_leaf
        ):
            return Node(
                prediction=majority_class
            )

        left_child = self._build_tree(
            left_X,
            left_y,
            depth + 1,
        )

        right_child = self._build_tree(
            right_X,
            right_y,
            depth + 1,
        )

        return Node(
            feature_index=feature_index,
            threshold=threshold,
            left=left_child,
            right=right_child,
        )

    # --------------------------------------------------------
    # FIT
    # --------------------------------------------------------

    def fit(self, X, y):
        X = pd.DataFrame(X).to_numpy(
            dtype=float
        )

        y = pd.Series(y).to_numpy()

        if len(X) != len(y):
            raise ValueError(
                "X and y must have the same number "
                "of samples."
            )

        if len(X) == 0:
            raise ValueError(
                "Training data cannot be empty."
            )

        self.root = self._build_tree(
            X,
            y,
            depth=0,
        )

        return self

    # --------------------------------------------------------
    # PREDICT ONE
    # --------------------------------------------------------

    def _predict_one(self, row, node):
        if node.is_leaf:
            return node.prediction

        if row[node.feature_index] <= node.threshold:
            return self._predict_one(
                row,
                node.left,
            )

        return self._predict_one(
            row,
            node.right,
        )

    # --------------------------------------------------------
    # PREDICT
    # --------------------------------------------------------

    def predict(self, X):
        X = pd.DataFrame(X).to_numpy(
            dtype=float
        )

        return [
            self._predict_one(
                row,
                self.root,
            )
            for row in X
        ]

    # --------------------------------------------------------
    # PRINT TREE
    # --------------------------------------------------------

    def _print_tree(self, node, depth=0):
        indentation = "    " * depth

        if node.is_leaf:
            print(
                f"{indentation}Leaf: "
                f"predict = {node.prediction}"
            )
            return

        print(
            f"{indentation}Feature[{node.feature_index}] "
            f"<= {node.threshold:.4f}"
        )

        print(
            f"{indentation}├── True:"
        )

        self._print_tree(
            node.left,
            depth + 1,
        )

        print(
            f"{indentation}└── False:"
        )

        self._print_tree(
            node.right,
            depth + 1,
        )

    def print_tree(self):
        if self.root is None:
            print("Tree has not been trained.")
            return

        self._print_tree(self.root)


# ============================================================
# GINI DEMONSTRATION
# ============================================================

print("=" * 60)
print("GINI IMPURITY EXAMPLE")
print("=" * 60)

example_labels = [
    "Yes",
    "Yes",
    "Yes",
    "Yes",
    "Yes",
    "Yes",
    "No",
    "No",
    "No",
    "No",
]

example_tree = DecisionTreeClassifierScratch()

gini_value = example_tree._gini(
    example_labels
)

print("\nClass counts:")
print(Counter(example_labels))

print("\nGini impurity:")
print(f"{gini_value:.4f}")


# ============================================================
# SIMPLE DECISION TREE EXAMPLE
# ============================================================

print("\n" + "=" * 60)
print("SIMPLE DECISION TREE EXAMPLE")
print("=" * 60)


X_simple = [
    [1, 1],
    [2, 1],
    [2, 2],
    [3, 2],
    [6, 6],
    [7, 7],
    [8, 7],
    [9, 8],
]

y_simple = [
    "A",
    "A",
    "A",
    "A",
    "B",
    "B",
    "B",
    "B",
]


simple_model = DecisionTreeClassifierScratch(
    max_depth=3,
    min_samples_split=2,
    min_samples_leaf=1,
)


simple_model.fit(
    X_simple,
    y_simple,
)


print("\nLearned Tree:")

simple_model.print_tree()


simple_points = [
    [2.5, 2],
    [7.5, 7],
]


simple_predictions = simple_model.predict(
    simple_points
)


print("\nPredictions:")

for point, prediction in zip(
    simple_points,
    simple_predictions,
):
    print(
        f"{point} -> {prediction}"
    )


# ============================================================
# STUDENT PERFORMANCE DATASET
# ============================================================

print("\n" + "=" * 60)
print("DECISION TREE ON STUDENT PERFORMANCE DATA")
print("=" * 60)


data = pd.read_csv(
    "datasets/student_performance.csv"
)


print("\nDataset:")
print(data)


# ------------------------------------------------------------
# Convert continuous score into a classification target
# ------------------------------------------------------------

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
# TRAIN FROM SCRATCH
# ============================================================

model = DecisionTreeClassifierScratch(
    max_depth=4,
    min_samples_split=2,
    min_samples_leaf=1,
)


model.fit(
    X_train,
    y_train,
)


print("\nLearned Decision Tree:")

model.print_tree()


# ============================================================
# PREDICTION
# ============================================================

predictions = model.predict(
    X_test
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


print("\nMaximum Depth:")
print(model.max_depth)


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
# PREDICTION TABLE
# ============================================================

results = X_test.copy()

results["actual"] = y_test.values

results["predicted"] = predictions


print("\nPredictions:")

print(
    results.reset_index(
        drop=True
    )
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


new_prediction = model.predict(
    new_student[features]
)[0]


print("\n" + "=" * 60)
print("NEW STUDENT PREDICTION")
print("=" * 60)


print("\nNew Student:")
print(new_student)


print("\nPrediction:")
print(new_prediction)


# ============================================================
# DEPTH COMPARISON
# ============================================================

print("\n" + "=" * 60)
print("TREE DEPTH COMPARISON")
print("=" * 60)


for depth in [1, 2, 3, 4, 5]:

    depth_model = (
        DecisionTreeClassifierScratch(
            max_depth=depth,
            min_samples_split=2,
            min_samples_leaf=1,
        )
    )

    depth_model.fit(
        X_train,
        y_train,
    )

    depth_predictions = (
        depth_model.predict(X_test)
    )

    depth_accuracy = accuracy_score(
        y_test,
        depth_predictions,
    )

    print(
        f"max_depth = {depth}: "
        f"accuracy = {depth_accuracy:.4f}"
    )
