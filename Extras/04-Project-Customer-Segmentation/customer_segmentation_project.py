import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def load_dataset():
    """Load the customer dataset."""
    dataset_path = (
        Path(__file__).resolve().parents[2]
        / "datasets"
        / "customers.csv"
    )

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}"
        )

    return pd.read_csv(dataset_path)


def inspect_dataset(df):
    """Display basic dataset information."""
    print("\n=== DATASET ===")
    print(df)

    print("\n=== SHAPE ===")
    print(df.shape)

    print("\n=== MISSING VALUES ===")
    print(df.isnull().sum())

    print("\n=== DUPLICATES ===")
    print(df.duplicated().sum())

    print("\n=== SUMMARY ===")
    print(df.describe())


def prepare_features(df):
    """Select numerical features for clustering."""
    feature_columns = [
        "age",
        "income",
        "spending_score"
    ]

    X = df[feature_columns].to_numpy(dtype=float)

    return X, feature_columns


def scale_features(X):
    """Standardize clustering features."""
    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    return X_scaled, scaler


def calculate_wcss(X, k):
    """Fit K-Means and return WCSS."""
    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    model.fit(X)

    return model.inertia_, model


def calculate_silhouette(X, labels):
    """Calculate silhouette score."""
    unique_labels = np.unique(labels)

    if len(unique_labels) < 2:
        return float("nan")

    return silhouette_score(X, labels)


def evaluate_k_values(X, min_k=2, max_k=8):
    """Evaluate several possible values of K."""
    results = []

    max_valid_k = min(max_k, len(X) - 1)

    for k in range(min_k, max_valid_k + 1):
        wcss, model = calculate_wcss(X, k)

        silhouette = calculate_silhouette(
            X,
            model.labels_
        )

        results.append({
            "k": k,
            "wcss": wcss,
            "silhouette": silhouette
        })

    return pd.DataFrame(results)


def choose_k(results):
    """Choose K using the highest silhouette score."""
    best_row = results.loc[
        results["silhouette"].idxmax()
    ]

    return int(best_row["k"])


def fit_final_model(X, k):
    """Fit the final K-Means model."""
    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(X)

    return model, labels


def create_cluster_profile(df, labels):
    """Create average feature values for each cluster."""
    result = df.copy()

    result["cluster"] = labels

    profile = (
        result
        .groupby("cluster")
        .agg({
            "age": "mean",
            "income": "mean",
            "spending_score": "mean"
        })
        .round(2)
    )

    counts = (
        result["cluster"]
        .value_counts()
        .sort_index()
        .rename("customer_count")
    )

    profile = profile.join(counts)

    return result, profile


def print_cluster_assignments(df, labels):
    """Print every customer's assigned cluster."""
    output = df.copy()
    output["cluster"] = labels

    print("\n=== CUSTOMER CLUSTERS ===")
    print(output.to_string(index=False))


def print_centroids(model, scaler, feature_columns):
    """Print centroids in original feature units."""
    original_centroids = scaler.inverse_transform(
        model.cluster_centers_
    )

    centroid_df = pd.DataFrame(
        original_centroids,
        columns=feature_columns
    )

    centroid_df.index.name = "cluster"

    print("\n=== CLUSTER CENTROIDS ===")
    print(centroid_df.round(2))


def main():
    print("CUSTOMER SEGMENTATION USING K-MEANS")

    # 1. Load dataset
    df = load_dataset()

    # 2. Inspect dataset
    inspect_dataset(df)

    # 3. Select features
    X, feature_columns = prepare_features(df)

    print("\n=== FEATURES ===")
    print(feature_columns)

    # 4. Scale features
    X_scaled, scaler = scale_features(X)

    print("\n=== SCALED DATA ===")
    print(
        pd.DataFrame(
            X_scaled,
            columns=feature_columns
        ).head()
    )

    # 5. Evaluate several K values
    results = evaluate_k_values(
        X_scaled,
        min_k=2,
        max_k=8
    )

    print("\n=== K EVALUATION ===")
    print(results.round(4).to_string(index=False))

    # 6. Select K
    best_k = choose_k(results)

    print(f"\nSelected K: {best_k}")
    print(
        "Selection criterion: "
        "highest silhouette score"
    )

    # 7. Fit final model
    model, labels = fit_final_model(
        X_scaled,
        best_k
    )

    # 8. Print assignments
    print_cluster_assignments(
        df,
        labels
    )

    # 9. Print centroids
    print_centroids(
        model,
        scaler,
        feature_columns
    )

    # 10. Create cluster profiles
    clustered_df, profile = create_cluster_profile(
        df,
        labels
    )

    print("\n=== CLUSTER PROFILE ===")
    print(profile)

    # 11. Final evaluation
    final_silhouette = silhouette_score(
        X_scaled,
        labels
    )

    print("\n=== FINAL MODEL ===")
    print(f"Clusters: {best_k}")
    print(f"WCSS: {model.inertia_:.4f}")
    print(
        f"Silhouette score: "
        f"{final_silhouette:.4f}"
    )


if __name__ == "__main__":
    main()
