import numpy as np
import pandas as pd
from pathlib import Path


class KMeansScratch:
    """
    Basic K-Means implementation from scratch.

    Supports:
    - Euclidean distance
    - Manhattan distance
    - Random centroid initialization
    - Convergence based on centroid movement
    """

    def __init__(
        self,
        n_clusters=3,
        max_iter=100,
        tol=1e-4,
        distance="euclidean",
        random_state=42,
    ):
        if n_clusters < 1:
            raise ValueError("n_clusters must be at least 1.")

        if distance not in {"euclidean", "manhattan"}:
            raise ValueError("distance must be 'euclidean' or 'manhattan'.")

        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.distance = distance
        self.random_state = random_state

        self.centroids = None
        self.labels_ = None
        self.n_iter_ = 0
        self.inertia_ = None

    def _distance_matrix(self, X, centroids):
        """
        Calculate distance from every point to every centroid.
        """
        if self.distance == "euclidean":
            distances = np.sqrt(
                np.sum((X[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
            )
        else:
            distances = np.sum(
                np.abs(X[:, None, :] - centroids[None, :, :]),
                axis=2,
            )

        return distances

    def _initialize_centroids(self, X):
        rng = np.random.default_rng(self.random_state)

        if self.n_clusters > len(X):
            raise ValueError(
                "n_clusters cannot be greater than the number of samples."
            )

        indices = rng.choice(len(X), size=self.n_clusters, replace=False)

        return X[indices].astype(float).copy()

    def fit(self, X):
        """
        Fit K-Means to X.
        """
        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError("X must be a 2-dimensional array.")

        if len(X) == 0:
            raise ValueError("X cannot be empty.")

        self.centroids = self._initialize_centroids(X)

        for iteration in range(1, self.max_iter + 1):
            distances = self._distance_matrix(X, self.centroids)

            labels = np.argmin(distances, axis=1)

            new_centroids = np.zeros_like(self.centroids)

            for cluster_id in range(self.n_clusters):
                cluster_points = X[labels == cluster_id]

                if len(cluster_points) == 0:
                    # Reinitialize an empty cluster using a random point.
                    rng = np.random.default_rng(
                        self.random_state + iteration + cluster_id
                    )
                    random_index = rng.integers(0, len(X))
                    new_centroids[cluster_id] = X[random_index]
                else:
                    new_centroids[cluster_id] = np.mean(
                        cluster_points,
                        axis=0,
                    )

            centroid_shift = np.max(
                np.linalg.norm(
                    new_centroids - self.centroids,
                    axis=1,
                )
            )

            self.centroids = new_centroids
            self.labels_ = labels
            self.n_iter_ = iteration

            if centroid_shift <= self.tol:
                break

        # WCSS uses squared Euclidean distance.
        final_distances = np.sqrt(
            np.sum(
                (X - self.centroids[self.labels_]) ** 2,
                axis=1,
            )
        )

        self.inertia_ = np.sum(final_distances ** 2)

        return self

    def predict(self, X):
        """
        Assign new points to the nearest learned centroid.
        """
        if self.centroids is None:
            raise ValueError("Fit the model before calling predict().")

        X = np.asarray(X, dtype=float)

        distances = self._distance_matrix(X, self.centroids)

        return np.argmin(distances, axis=1)

    def fit_predict(self, X):
        """
        Fit the model and return cluster assignments.
        """
        self.fit(X)
        return self.labels_


def euclidean_distance(point_a, point_b):
    """
    Calculate Euclidean distance between two points.
    """
    point_a = np.asarray(point_a, dtype=float)
    point_b = np.asarray(point_b, dtype=float)

    return np.sqrt(np.sum((point_a - point_b) ** 2))


def manhattan_distance(point_a, point_b):
    """
    Calculate Manhattan distance between two points.
    """
    point_a = np.asarray(point_a, dtype=float)
    point_b = np.asarray(point_b, dtype=float)

    return np.sum(np.abs(point_a - point_b))


def lecture_euclidean_example():
    """
    Reproduce the K=2 Euclidean example from the lecture.

    Points:
        (2,10), (2,5), (8,4), (5,8),
        (7,5), (6,4), (1,2), (4,9)

    Initial centroids:
        C1 = (2,10)
        C2 = (2,5)
    """

    X = np.array(
        [
            [2, 10],
            [2, 5],
            [8, 4],
            [5, 8],
            [7, 5],
            [6, 4],
            [1, 2],
            [4, 9],
        ],
        dtype=float,
    )

    centroids = np.array(
        [
            [2, 10],
            [2, 5],
        ],
        dtype=float,
    )

    print("\n" + "=" * 70)
    print("LECTURE EXAMPLE: K=2, EUCLIDEAN DISTANCE")
    print("=" * 70)

    for iteration in range(2):
        distances = np.sqrt(
            np.sum((X[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
        )

        labels = np.argmin(distances, axis=1)

        print(f"\nIteration {iteration + 1}")

        for cluster_id in range(2):
            cluster_points = X[labels == cluster_id]
            print(f"Cluster {cluster_id + 1}:")
            print(cluster_points)

        new_centroids = np.array(
            [
                X[labels == cluster_id].mean(axis=0)
                for cluster_id in range(2)
            ]
        )

        print("Centroids:")
        print(new_centroids)

        centroids = new_centroids


def lecture_manhattan_example():
    """
    Reproduce the K=3 Manhattan example from the lecture.

    Initial centroids:
        C1 = (2,10)
        C2 = (5,8)
        C3 = (1,2)
    """

    X = np.array(
        [
            [2, 10],
            [2, 5],
            [8, 4],
            [5, 8],
            [7, 5],
            [6, 4],
            [1, 2],
            [4, 9],
        ],
        dtype=float,
    )

    centroids = np.array(
        [
            [2, 10],
            [5, 8],
            [1, 2],
        ],
        dtype=float,
    )

    distances = np.sum(
        np.abs(X[:, None, :] - centroids[None, :, :]),
        axis=2,
    )

    labels = np.argmin(distances, axis=1)

    print("\n" + "=" * 70)
    print("LECTURE EXAMPLE: K=3, MANHATTAN DISTANCE")
    print("=" * 70)

    for cluster_id in range(3):
        print(f"\nCluster {cluster_id + 1}:")
        print(X[labels == cluster_id])

    new_centroids = np.array(
        [
            X[labels == cluster_id].mean(axis=0)
            for cluster_id in range(3)
        ]
    )

    print("\nUpdated centroids:")
    print(new_centroids)


def load_customer_data():
    """
    Load the repository customer dataset.
    """

    repo_root = Path(__file__).resolve().parents[2]

    dataset_path = repo_root / "datasets" / "customers.csv"

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}"
        )

    return pd.read_csv(dataset_path)


def scale_features(X):
    """
    Standardize features manually.
    """
    X = np.asarray(X, dtype=float)

    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)

    # Prevent division by zero for constant features.
    std = np.where(std == 0, 1, std)

    return (X - mean) / std


def customer_clustering_demo():
    """
    Apply K-Means from scratch to customers.csv.
    """

    df = load_customer_data()

    feature_columns = [
        "age",
        "income",
        "spending_score",
    ]

    X = df[feature_columns].to_numpy(dtype=float)

    X_scaled = scale_features(X)

    model = KMeansScratch(
        n_clusters=3,
        max_iter=100,
        tol=1e-4,
        distance="euclidean",
        random_state=42,
    )

    labels = model.fit_predict(X_scaled)

    df["cluster"] = labels

    print("\n" + "=" * 70)
    print("CUSTOMER CLUSTERING")
    print("=" * 70)

    print("\nFinal centroids:")
    print(model.centroids)

    print("\nIterations:")
    print(model.n_iter_)

    print("\nWCSS:")
    print(model.inertia_)

    print("\nCluster counts:")
    print(df["cluster"].value_counts().sort_index())

    print("\nCluster assignments:")
    print(df)

    print("\nCluster profiles:")
    print(
        df.groupby("cluster")[feature_columns]
        .mean()
        .round(2)
    )


if __name__ == "__main__":
    lecture_euclidean_example()

    lecture_manhattan_example()

    customer_clustering_demo()
