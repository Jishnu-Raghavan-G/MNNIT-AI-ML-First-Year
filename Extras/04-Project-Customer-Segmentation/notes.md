# Customer Segmentation

## 1. Project Overview

This project applies K-Means Clustering to divide customers into groups with similar characteristics.

Unlike supervised learning, clustering does not require a target label.

The model discovers groups based on the similarity between customers.

---

## 2. Dataset

The project uses:

    datasets/customers.csv

Columns:

| Column | Description |
|---|---|
| age | Customer age |
| income | Customer income |
| spending_score | Customer spending score |

All three columns are used as features.

There is no target variable.

---

## 3. Why K-Means?

K-Means is an unsupervised clustering algorithm.

Its objective is to divide observations into K clusters.

Each cluster has a centroid.

The algorithm assigns each customer to the nearest centroid and repeatedly updates the centroids.

---

## 4. Project Workflow

    Load Data
        ↓
    Inspect Data
        ↓
    Select Features
        ↓
    Standardize Features
        ↓
    Test Different K Values
        ↓
    Calculate WCSS
        ↓
    Calculate Silhouette Score
        ↓
    Select K
        ↓
    Train Final K-Means Model
        ↓
    Analyze Customer Clusters

---

## 5. Feature Scaling

Age, income and spending score may have different numerical ranges.

Without scaling, a feature with a much larger numerical range can dominate distance calculations.

Standardization:

    x' = (x - μ) / σ

The mean and standard deviation are calculated from the dataset used for scaling.

---

## 6. K-Means Algorithm

The basic procedure is:

1. Choose K.
2. Initialize K centroids.
3. Calculate the distance between every point and every centroid.
4. Assign each point to its nearest centroid.
5. Recalculate each centroid using the mean of its assigned points.
6. Repeat the assignment and update steps.
7. Stop when the centroids stabilize or the maximum number of iterations is reached.

---

## 7. Distance

Euclidean distance between two points is:

    d = √Σ(xᵢ - yᵢ)²

For two dimensions:

    d = √((x₁-y₁)² + (x₂-y₂)²)

Distance determines which centroid is closest to a customer.

---

## 8. Centroid

A centroid is the mean position of the points assigned to a cluster.

For one feature:

    centroid = mean(points)

For multiple features, the mean is calculated independently for every feature.

---

## 9. Choosing K

The number of clusters K must be selected.

Two useful approaches are:

- Elbow Method
- Silhouette Score

---

## 10. WCSS

WCSS means Within-Cluster Sum of Squares.

It measures the total squared distance between observations and their assigned cluster centroids.

    WCSS = Σ Σ ||xᵢ - μₖ||²

where:

- xᵢ = data point
- μₖ = centroid of its cluster

Lower WCSS means points are closer to their centroids.

---

## 11. Elbow Method

Run K-Means for several values of K.

For example:

    K = 2
    K = 3
    K = 4
    K = 5
    K = 6

Calculate WCSS for every K.

WCSS generally decreases as K increases.

Plotting WCSS against K can reveal an elbow.

The elbow represents a point where increasing K further gives diminishing improvement.

---

## 12. Silhouette Score

The silhouette score measures how well each point fits within its cluster compared with other clusters.

For point i:

    s(i) = (b(i) - a(i)) / max(a(i), b(i))

where:

- a(i) = average distance to points in the same cluster
- b(i) = average distance to points in the nearest other cluster

The score ranges from:

    -1 to +1

Interpretation:

- Close to +1 → good clustering
- Around 0 → point may lie near a cluster boundary
- Negative → point may be assigned to an inappropriate cluster

---

## 13. WCSS vs Silhouette Score

WCSS focuses on compactness.

Silhouette score considers both:

- Within-cluster similarity
- Separation from other clusters

Therefore, it is useful to consider both measures rather than relying on only one.

---

## 14. Cluster Profiles

After clustering, calculate statistics for each cluster.

For example:

    average age
    average income
    average spending score
    number of customers

This helps interpret what each cluster represents.

Example interpretations might include:

- Younger customers with lower income.
- Higher-income customers with high spending.
- Customers with moderate spending.
- Older customers with different spending behavior.

The actual interpretation must be based on the resulting cluster statistics.

---

## 15. K-Means and Unsupervised Learning

K-Means is unsupervised because the dataset does not provide a target label indicating the correct cluster.

The algorithm discovers structure from the feature values.

This differs from classification.

### Classification

Input → known label

### Clustering

Input → discovered group

---

## 16. Advantages

- Simple to understand.
- Easy to implement.
- Efficient for many datasets.
- Useful for customer segmentation.
- Works well when clusters are reasonably compact.

---

## 17. Limitations

- K must be selected.
- Sensitive to initialization.
- Sensitive to feature scaling.
- Sensitive to outliers.
- Works best for certain cluster shapes.
- Different initial centroids can produce different results.

K-Means++ initialization can improve centroid initialization.

---

## 18. Common Mistakes

### Mistake 1: Forgetting Feature Scaling

Features with larger numerical ranges can dominate distance calculations.

### Mistake 2: Treating Cluster Numbers as Labels

Cluster 0 is not inherently better or worse than Cluster 1.

The numbers are identifiers, not meaningful rankings.

### Mistake 3: Choosing K Arbitrarily

Use methods such as the elbow method and silhouette score.

### Mistake 4: Assuming Clusters Have Meaning Automatically

Clusters need to be analyzed and interpreted using their feature statistics.

---

## 19. Learning Objectives

After completing this project, you should understand:

- What unsupervised learning means.
- How K-Means works.
- How distance is used in clustering.
- Why feature scaling matters.
- What centroids represent.
- How WCSS is calculated.
- How the elbow method helps choose K.
- How silhouette score evaluates clustering.
- How to interpret customer segments.

---

## 20. How to Run

From the repository root:

    python Extras/04-Project-Customer-Segmentation/customer_segmentation_project.py

Required libraries:

    numpy
    pandas
    scikit-learn
