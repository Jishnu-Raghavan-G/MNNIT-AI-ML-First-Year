# Module 08 — K-Means Clustering

## 1. Introduction

K-Means is an unsupervised machine learning algorithm used for clustering.

Unlike supervised learning, there is no target label provided to the model. The algorithm tries to divide data into groups based on similarity.

The main idea is:

> Similar data points should belong to the same cluster.

K-Means is commonly used for:
- Customer segmentation
- Grouping similar products
- Image compression
- Pattern discovery
- Market segmentation
- Exploratory data analysis

---

## 2. What Is Clustering?

Clustering means dividing a dataset into groups called clusters.

For example, suppose we have customers described by:

- Age
- Income
- Spending Score

We may want to discover groups such as:

- Young, high-spending customers
- Older, low-spending customers
- High-income customers with moderate spending

There are no predefined labels such as "Group 1" or "Group 2".

The algorithm discovers the groups from the data.

---

## 3. Why Is K-Means Called K-Means?

The algorithm has two important ideas:

### K

`K` represents the number of clusters we want.

For example:

- K = 2 → two clusters
- K = 3 → three clusters
- K = 5 → five clusters

### Means

Each cluster is represented by its centroid.

The centroid is calculated using the mean of the points assigned to that cluster.

Therefore:

> K-Means = K clusters + mean-based centroids

---

## 4. Important Terms

### Data Point

One observation in the dataset.

Example:

`(2, 10)`

### Cluster

A group of similar data points.

### Centroid

The center of a cluster.

For a two-dimensional cluster:

`Centroid = (mean of x values, mean of y values)`

### K

The number of clusters.

### Distance

A measure of how far a point is from a centroid.

---

## 5. Basic Idea of K-Means

K-Means repeatedly performs four main operations:

1. Choose K initial centroids.
2. Assign every point to its nearest centroid.
3. Recalculate the centroids using the mean of assigned points.
4. Repeat until the clusters stop changing or the maximum number of iterations is reached.

In simple form:

`Initialize → Assign → Update → Repeat`

---

## 6. K-Means Algorithm

Suppose we have a dataset X and want K clusters.

### Step 1 — Choose K

Decide how many clusters are required.

Example:

`K = 3`

### Step 2 — Initialize Centroids

Select K points as initial centroids.

These may be selected randomly.

### Step 3 — Calculate Distances

For every data point, calculate its distance from every centroid.

### Step 4 — Assign Points

Assign each point to the nearest centroid.

### Step 5 — Update Centroids

For every cluster, calculate the mean of all points belonging to that cluster.

### Step 6 — Repeat

Repeat the assignment and update steps.

Stop when:

- Centroids barely change,
- Cluster assignments stop changing, or
- Maximum iterations are reached.

---

## 7. Euclidean Distance

The most common distance measure in K-Means is Euclidean distance.

For two points:

`P = (x1, y1)`

`Q = (x2, y2)`

the Euclidean distance is:

`d(P,Q) = sqrt((x1-x2)^2 + (y1-y2)^2)`

For n-dimensional data:

`d(P,Q) = sqrt(sum((Pi-Qi)^2))`

### Example

Consider:

`P = (2, 5)`

`Q = (5, 9)`

Then:

`d(P,Q) = sqrt((2-5)^2 + (5-9)^2)`

`= sqrt(9 + 16)`

`= sqrt(25)`

`= 5`

---

## 8. Manhattan Distance

Another distance measure is Manhattan distance.

For:

`P = (x1, y1)`

`Q = (x2, y2)`

the Manhattan distance is:

`d(P,Q) = |x1-x2| + |y1-y2|`

Example:

`P = (2, 5)`

`Q = (5, 9)`

`d(P,Q) = |2-5| + |5-9|`

`= 3 + 4`

`= 7`

---

## 9. Lecture Example — K = 2 Using Euclidean Distance

Consider the following points:

`(2,10), (2,5), (8,4), (5,8), (7,5), (6,4), (1,2), (4,9)`

Let:

`K = 2`

Initial centroids:

`C1 = (2,10)`

`C2 = (2,5)`

Each point is assigned to whichever centroid is closer.

After the first assignment:

### Cluster 1

`C1 = {(2,10), (5,8), (4,9)}`

### Cluster 2

`C2 = {(2,5), (8,4), (7,5), (6,4), (1,2)}`

---

## 10. Updating the Centroids

For Cluster 1:

Points:

`(2,10), (5,8), (4,9)`

Mean x:

`(2 + 5 + 4) / 3 = 3.67`

Mean y:

`(10 + 8 + 9) / 3 = 9`

Therefore:

`C1 = (3.67, 9.0)`

For Cluster 2:

Points:

`(2,5), (8,4), (7,5), (6,4), (1,2)`

Mean x:

`(2 + 8 + 7 + 6 + 1) / 5 = 4.8`

Mean y:

`(5 + 4 + 5 + 4 + 2) / 5 = 4`

Therefore:

`C2 = (4.8, 4.0)`

---

## 11. Second Iteration

Using:

`C1 = (3.67, 9.0)`

`C2 = (4.8, 4.0)`

the points are assigned again.

The assignments remain unchanged:

### Cluster 1

`{(2,10), (5,8), (4,9)}`

### Cluster 2

`{(2,5), (8,4), (7,5), (6,4), (1,2)}`

Therefore, the algorithm has converged for this example.

Final centroids:

`C1 = (3.67, 9.0)`

`C2 = (4.8, 4.0)`

---

## 12. Lecture Example — K = 3 Using Manhattan Distance

Using the same points:

`(2,10), (2,5), (8,4), (5,8), (7,5), (6,4), (1,2), (4,9)`

Let:

`K = 3`

Initial centroids:

`C1 = (2,10)`

`C2 = (5,8)`

`C3 = (1,2)`

Using Manhattan distance, the first assignment gives:

### Cluster 1

`{(2,10)}`

### Cluster 2

`{(8,4), (5,8), (7,5), (6,4), (4,9)}`

### Cluster 3

`{(2,5), (1,2)}`

---

## 13. Updating the K = 3 Centroids

Cluster 1:

`C1 = (2,10)`

Cluster 2:

Points:

`(8,4), (5,8), (7,5), (6,4), (4,9)`

Mean x:

`(8 + 5 + 7 + 6 + 4) / 5 = 6`

Mean y:

`(4 + 8 + 5 + 4 + 9) / 5 = 6`

Therefore:

`C2 = (6,6)`

Cluster 3:

Points:

`(2,5), (1,2)`

Mean x:

`(2 + 1) / 2 = 1.5`

Mean y:

`(5 + 2) / 2 = 3.5`

Therefore:

`C3 = (1.5,3.5)`

The lecture example therefore obtains:

`C1 = (2,10)`

`C2 = (6,6)`

`C3 = (1.5,3.5)`

---

## 14. Centroid Update

The centroid of a cluster is calculated using the arithmetic mean.

For a cluster containing points:

`(x1,y1), (x2,y2), ..., (xn,yn)`

the centroid is:

`Cx = (x1+x2+...+xn)/n`

`Cy = (y1+y2+...+yn)/n`

For higher-dimensional data, calculate the mean independently for every feature.

---

## 15. When Does K-Means Stop?

K-Means can stop when:

### Condition 1 — Assignments Stop Changing

If every point remains in the same cluster, the algorithm has converged.

### Condition 2 — Centroids Stop Changing

If the centroids change by less than a small tolerance, the algorithm can stop.

### Condition 3 — Maximum Iterations

A maximum number of iterations can be specified to prevent endless computation.

---

## 16. WCSS

WCSS stands for:

> Within-Cluster Sum of Squares

It measures how close the points are to their cluster centroids.

The formula is:

`WCSS = sum over clusters [sum over points in cluster (distance from point to centroid)^2]`

For cluster k:

`WCSS_k = sum ||xi - Ck||^2`

Overall:

`WCSS = sum(k=1 to K) sum(xi in cluster k) ||xi-Ck||^2`

A smaller WCSS means the points are generally closer to their centroids.

---

## 17. Why Does WCSS Decrease as K Increases?

If we increase K, we allow the algorithm to create more clusters.

With more centroids, points can generally be placed closer to a centroid.

Therefore:

`K increases → WCSS decreases`

For example:

`K=1 → high WCSS`

`K=2 → lower WCSS`

`K=3 → lower WCSS`

`K=4 → lower WCSS`

However, choosing a very large K is not automatically better.

We need a method to choose a suitable K.

---

## 18. Elbow Method

The Elbow Method is commonly used to select K.

### Procedure

1. Run K-Means for several values of K.
2. Calculate WCSS for each K.
3. Plot K against WCSS.
4. Look for the point where the decrease in WCSS starts becoming much smaller.

This point is called the elbow.

The intuition is:

> Choose K where adding another cluster provides significantly less improvement.

---

## 19. Elbow Example

Suppose:

`K = 1 → WCSS = 1000`

`K = 2 → WCSS = 500`

`K = 3 → WCSS = 250`

`K = 4 → WCSS = 210`

`K = 5 → WCSS = 190`

The improvement is very large initially.

After K = 3, the improvement becomes much smaller.

Therefore, K = 3 may be a reasonable choice.

The elbow method is a heuristic, not an absolute rule.

---

## 20. Silhouette Score

Another method for evaluating clustering is the silhouette score.

For a data point i:

`s(i) = (b(i)-a(i)) / max(a(i),b(i))`

where:

`a(i) = average distance from i to other points in its own cluster`

`b(i) = average distance from i to points in the nearest other cluster`

The silhouette score ranges from:

`-1 to +1`

---

## 21. Interpreting Silhouette Score

### Close to +1

The point is well matched to its own cluster and far from other clusters.

This generally indicates good clustering.

### Around 0

The point lies near a cluster boundary.

### Negative

The point may have been assigned to the wrong cluster.

Therefore:

> Higher average silhouette score generally indicates better-separated clusters.

---

## 22. WCSS vs Silhouette Score

### WCSS

Measures compactness.

Lower WCSS is better.

### Silhouette Score

Measures both:

- How close points are to their own cluster
- How separated they are from other clusters

Higher silhouette score is generally better.

Important:

WCSS almost always decreases when K increases, so WCSS alone cannot simply tell us to choose the largest K.

---

## 23. Feature Scaling

Distance-based algorithms are sensitive to feature scales.

Suppose a dataset contains:

- Age: 18–70
- Income: 20,000–2,00,000

Income has a much larger numerical scale than age.

Without scaling, income can dominate distance calculations.

Therefore, feature scaling is often important before K-Means.

---

## 24. Standardization

Standardization transforms a feature using:

`z = (x - mean) / standard deviation`

After standardization:

- Mean is approximately 0
- Standard deviation is approximately 1

This helps features contribute more comparably to distance calculations.

---

## 25. Dataset Used in This Module

The repository contains:

`datasets/customers.csv`

with features:

- `age`
- `income`
- `spending_score`

There is no target label.

This makes it suitable for demonstrating unsupervised clustering.

The workflow is:

`Load data`

`→ Select numerical features`

`→ Scale features`

`→ Apply K-Means`

`→ Inspect clusters`

`→ Evaluate clustering`

---

## 26. K-Means From Scratch

A basic implementation can be written using NumPy.

The main components are:

1. Initialize centroids.
2. Calculate distances.
3. Assign clusters.
4. Calculate new centroids.
5. Repeat.
6. Store final centroids and labels.

A basic pseudocode representation is:

Initialize K centroids

Repeat:

    Calculate distance from every point to every centroid

    Assign every point to the nearest centroid

    Recalculate each centroid as the mean of its assigned points

    Check convergence

Return clusters and centroids

---

## 27. Initialization

The initial centroids can affect the final solution.

Random initialization can produce different results on different runs.

Therefore, setting a `random_state` makes experiments reproducible.

A common improved initialization technique is K-Means++.

K-Means++ attempts to select initial centroids that are spread apart.

---

## 28. Empty Clusters

Sometimes no data points are assigned to a centroid.

This creates an empty cluster.

A practical implementation needs to handle this case.

Possible strategies include:

- Reinitialize the centroid
- Choose a data point as the new centroid
- Use another initialization strategy

---

## 29. Advantages of K-Means

- Simple to understand
- Easy to implement
- Relatively fast
- Works well for many numerical datasets
- Useful for exploratory analysis
- Easy to visualize in low dimensions

---

## 30. Limitations of K-Means

- K must usually be selected beforehand
- Sensitive to initialization
- Sensitive to feature scaling
- Sensitive to outliers
- Works best with roughly compact/spherical clusters
- Different runs can produce different results
- Choosing K may not always be obvious

---

## 31. Common Mistakes

### Mistake 1 — Forgetting Scaling

Features with large numerical ranges can dominate distance calculations.

### Mistake 2 — Choosing K Arbitrarily

Use methods such as:

- Elbow Method
- Silhouette Score
- Domain knowledge

### Mistake 3 — Confusing Clustering With Classification

K-Means does not learn predefined labels.

Classification uses labelled data.

Clustering discovers groups without target labels.

### Mistake 4 — Thinking Lower WCSS Alone Is Always Better

WCSS decreases as K increases.

A very large K can create unnecessary clusters.

### Mistake 5 — Assuming One Run Is Always Enough

Different initializations can produce different results.

---

## 32. K-Means vs Classification

| Property | K-Means | Classification |
|---|---|---|
| Learning type | Unsupervised | Supervised |
| Labels required | No | Yes |
| Main goal | Discover groups | Predict class |
| Example | Customer segmentation | Spam detection |
| Output | Cluster assignment | Class label |

---

## 33. Important Formulas

### Euclidean Distance

`d(P,Q) = sqrt(sum((Pi-Qi)^2))`

### Manhattan Distance

`d(P,Q) = sum(|Pi-Qi|)`

### Centroid

`Cj = mean of all points assigned to cluster j`

### WCSS

`WCSS = sum ||xi-Ccluster(i)||^2`

### Silhouette Score

`s(i) = (b(i)-a(i)) / max(a(i),b(i))`

---

## 34. Important Terms for Exams

Remember these terms:

- Clustering
- K
- Cluster
- Centroid
- Euclidean distance
- Manhattan distance
- Initialization
- Assignment
- Centroid update
- Convergence
- WCSS
- Elbow Method
- Silhouette Score
- Feature scaling
- K-Means++
- Unsupervised learning

---

## 35. Complete K-Means Workflow

A practical K-Means workflow is:

`Dataset`

`↓`

`Select numerical features`

`↓`

`Check and clean data`

`↓`

`Scale features if necessary`

`↓`

`Choose candidate values of K`

`↓`

`Run K-Means`

`↓`

`Calculate WCSS / Silhouette Score`

`↓`

`Select a suitable K`

`↓`

`Fit final K-Means model`

`↓`

`Analyze clusters`

---

## 36. Final Mental Model

The easiest way to remember K-Means is:

> K-Means repeatedly assigns points to the nearest centroid and then moves each centroid to the mean of its assigned points.

The complete cycle is:

`Choose K`

`↓`

`Initialize centroids`

`↓`

`Calculate distances`

`↓`

`Assign points`

`↓`

`Calculate means`

`↓`

`Move centroids`

`↓`

`Repeat`

`↓`

`Converge`

For choosing K:

`Run multiple K values`

`↓`

`Compare WCSS / Silhouette`

`↓`

`Choose a sensible K`

That is the core idea behind K-Means clustering.
