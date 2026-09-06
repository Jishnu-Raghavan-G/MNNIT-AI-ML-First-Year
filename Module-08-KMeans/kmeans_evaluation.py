import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def load_customer_data():
    """
    Load customers.csv from the repository datasets folder.
    """

    repo_root = Path(__file__).resolve().parents[2]

    dataset_path = repo_root / "datasets" / "customers.csv"

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}"
        )

    return pd.read_csv(dataset_path)


def calculate_wcss(X, k, random_state=42):
    """
    Fit K-Means for a given K and return WCSS.

    sklearn's inertia_ is the within-cluster sum of squared
    Euclidean distances from points to their centroids.
    """

    model = KMeans(
        n_clusters=k,
        random_state=random_state,
        n_init=10,
    )

    model.fit(X)

    return model.inertia_, model


def calculate_silhouette(X, labels):
    """
    Calculate the average silhouette score.
    """

    number_of_clusters = len(np.unique(labels))

    if number_of_clusters < 2:
        return np.nan

    return silhouette_score(X, labels)


def evaluate_k_values(X, min_k=2, max_k=8):
    """
    Calculate WCSS and silhouette score for several K values.
    """

    results = []

    max_valid_k = min(max_k, len(X) - 1)

    for k in range(min_k, max_valid_k + 1):
        wcss, model = calculate_wcss(X, k)

        labels = model.labels_

        silhouette = calculate_silhouette(
            X,
            labels,
        )

        results.append(
            {
                "k": k,
                "wcss": wcss,
                "silhouette": silhouette,
            }
        )

    return pd.DataFrame(results)


def print_elbow_results(results):
    """
    Print WCSS values for the elbow method.
    """

    print("\n" + "=" * 70)
    print("ELBOW METHOD — WCSS")
    print("=" * 70)

    print(
        results[
            ["k", "wcss"]
        ].to_string(index=False)
    )


def print_silhouette_results(results):
    """
    Print silhouette scores.
    """

    print("\n" + "=" * 70)
    print("SILHOUETTE SCORES")
    print("=" * 70)

    print(
        results[
            ["k", "silhouette"]
        ].to_string(index=False)
    )


def choose_best_k_by_silhouette(results):
    """
    Select K with the highest silhouette score.
    """

    valid_results = results.dropna(
        subset=["silhouette"]
    )

    if valid_results.empty:
        raise ValueError(
            "No valid silhouette scores available."
        )

    best_row = valid_results.loc[
        valid_results["silhouette"].idxmax()
    ]

    return int(best_row["k"])


def final_clustering(X, df, feature_columns, k):
    """
    Train final K-Means model and display cluster information.
    """

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10,
    )

    labels = model.fit_predict(X)

    result_df = df.copy()

    result_df["cluster"] = labels

    print("\n" + "=" * 70)
    print(f"FINAL K-MEANS MODEL — K={k}")
    print("=" * 70)

    print("\nFinal WCSS:")
    print(model.inertia_)

    print("\nCluster counts:")
    print(
        result_df["cluster"]
        .value_counts()
        .sort_index()
    )

    print("\nCluster profiles:")
    print(
        result_df.groupby("cluster")[feature_columns]
        .mean()
        .round(2)
    )

    print("\nCluster centers in scaled feature space:")
    print(
        pd.DataFrame(
            model.cluster_centers_,
            columns=feature_columns,
        ).round(3)
    )

    print("\nFirst rows with cluster assignments:")
    print(result_df.head(10))

    return model, result_df


def main():
    df = load_customer_data()

    feature_columns = [
        "age",
        "income",
        "spending_score",
    ]

    X = df[feature_columns].to_numpy(dtype=float)

    print("=" * 70)
    print("CUSTOMER K-MEANS EVALUATION")
    print("=" * 70)

    print("\nOriginal data:")
    print(df.head())

    # Scale features because K-Means is distance-based.
    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    print("\nFeature means after scaling:")
    print(X_scaled.mean(axis=0).round(6))

    print("\nFeature standard deviations after scaling:")
    print(X_scaled.std(axis=0).round(6))

    results = evaluate_k_values(
        X_scaled,
        min_k=2,
        max_k=8,
    )

    print_elbow_results(results)

    print_silhouette_results(results)

    best_k = choose_best_k_by_silhouette(results)

    print("\nBest K according to silhouette score:")
    print(best_k)

    final_clustering(
        X_scaled,
        df,
        feature_columns,
        best_k,
    )


if __name__ == "__main__":
    main()
