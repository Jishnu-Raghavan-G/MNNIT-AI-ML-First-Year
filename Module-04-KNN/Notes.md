# Module 04 — K-Nearest Neighbors (KNN)

> The lecture deck lists KNN as one of the machine-learning classification algorithms, but it does not provide a detailed KNN section or worked KNN algorithm in the supplied slides.
> The notes below therefore expand KNN as an implementation-oriented module while keeping the course's classification and distance-based framing.

---

# 1. What is K-Nearest Neighbors?

K-Nearest Neighbors (KNN) is a supervised machine-learning algorithm used mainly for:

- Classification
- Regression

The basic idea is simple:

"Similar data points tend to have similar outputs."

Instead of learning an explicit mathematical model during training, KNN stores the training data.

When a new data point arrives:

1. Calculate its distance from the training points.
2. Find the K closest points.
3. For classification, use the majority class among those neighbors.
4. For regression, use the average target value of those neighbors.

KNN is therefore called a:

- Instance-based learning algorithm
- Lazy learning algorithm
- Non-parametric algorithm

---

# 2. Why is KNN Called a Lazy Learner?

Many machine-learning algorithms perform substantial computation during training.

For example:

Linear Regression learns coefficients.

Decision Trees construct a tree.

Naive Bayes calculates probability distributions.

KNN does not build a complicated model during training.

Its training phase is mainly:

    Store the training data.

Most of the computational work happens when making a prediction.

Therefore:

Training:
    Very little computation

Prediction:
    More computation because distances must be calculated

---

# 3. KNN for Classification

Suppose we have training points belonging to different classes.

For a new point:

    New point → Calculate distances → Find K nearest points → Majority vote → Prediction

Example:

Suppose K = 3.

The three nearest neighbors are:

    Neighbor 1 → Class A
    Neighbor 2 → Class A
    Neighbor 3 → Class B

Votes:

    Class A = 2
    Class B = 1

Therefore:

    Prediction = Class A

---

# 4. Meaning of K

K represents the number of nearest neighbors considered when making a prediction.

For example:

    K = 1
    → Look at only the nearest point

    K = 3
    → Look at the three nearest points

    K = 5
    → Look at the five nearest points

The value of K strongly affects the behavior of the model.

---

# 5. Small K vs Large K

## Small K

Example:

    K = 1

The model considers very few neighbors.

Advantages:

- Can capture local patterns
- More flexible

Disadvantages:

- Sensitive to noise
- Can overfit the training data

For K = 1, a single unusual training point can determine the prediction.

---

## Large K

Example:

    K = 50

The model considers many neighbors.

Advantages:

- More stable
- Less sensitive to individual noisy points

Disadvantages:

- Can ignore local patterns
- Can underfit

Therefore, K should be selected carefully.

---

# 6. Choosing K

There is no single universally correct value of K.

A practical approach is:

1. Try several K values.
2. Evaluate their validation performance.
3. Select a suitable value.

Common choices include:

    K = 3
    K = 5
    K = 7
    K = 9

For binary classification, an odd value of K is often useful because it reduces the chance of a tie.

The best K depends on the dataset.

---

# 7. Distance in KNN

KNN depends heavily on distance.

A common distance measure is Euclidean distance.

For two points:

    A = (x1, y1)
    B = (x2, y2)

Euclidean distance is:

    d(A, B) = sqrt((x2 - x1)^2 + (y2 - y1)^2)

For n-dimensional data:

    d(x, y) =
        sqrt(
            Σ (xi - yi)^2
        )

---

# 8. Example of Euclidean Distance

Suppose:

    A = (2, 3)
    B = (5, 7)

Then:

    d(A, B)
    = sqrt((5 - 2)^2 + (7 - 3)^2)

    = sqrt(3^2 + 4^2)

    = sqrt(9 + 16)

    = sqrt(25)

    = 5

Therefore:

    Distance = 5

---

# 9. Manhattan Distance

Another distance metric is Manhattan distance.

For two points:

    A = (x1, y1)
    B = (x2, y2)

Manhattan distance is:

    d(A, B)
    = |x2 - x1| + |y2 - y1|

Example:

    A = (2, 3)
    B = (5, 7)

    d(A, B)
    = |5 - 2| + |7 - 3|

    = 3 + 4

    = 7

Therefore:

    Manhattan distance = 7

The lecture deck also uses Euclidean and Manhattan distance in its distance-based machine-learning examples, particularly in the K-Means section.

---

# 10. KNN Algorithm

For classification:

Step 1:
    Store the training dataset.

Step 2:
    Select K.

Step 3:
    For a new observation, calculate its distance from every training observation.

Step 4:
    Sort the observations by distance.

Step 5:
    Select the K closest observations.

Step 6:
    Count the class labels among those K neighbors.

Step 7:
    Assign the class with the highest number of votes.

---

# 11. Complete Example

Suppose we have:

    Point       Class
    ------------------
    (1, 1)      A
    (2, 2)      A
    (3, 3)      B
    (6, 6)      B

New point:

    X = (2, 3)

Use:

    K = 3

Calculate distances.

Distance to (1, 1):

    sqrt((2 - 1)^2 + (3 - 1)^2)
    = sqrt(1 + 4)
    = sqrt(5)

Distance to (2, 2):

    sqrt((2 - 2)^2 + (3 - 2)^2)
    = 1

Distance to (3, 3):

    sqrt((2 - 3)^2 + (3 - 3)^2)
    = 1

Distance to (6, 6):

    sqrt((2 - 6)^2 + (3 - 6)^2)
    = 5

Nearest three:

    (2, 2) → A
    (3, 3) → B
    (1, 1) → A

Votes:

    A → 2
    B → 1

Therefore:

    Prediction = A

---

# 12. KNN Regression

KNN can also be used for regression.

Instead of voting for a class, calculate the average target value of the K nearest neighbors.

Example:

    Neighbor target values:

    80
    85
    90

Then:

    Prediction
    = (80 + 85 + 90) / 3
    = 85

Therefore:

    Predicted value = 85

---

# 13. Distance-Weighted KNN

Basic KNN gives every selected neighbor equal importance.

However, a closer point may be more relevant than a farther point.

Distance-weighted KNN gives greater importance to closer neighbors.

A common idea is:

    Weight ∝ 1 / distance

Therefore:

    Smaller distance → Larger weight
    Larger distance  → Smaller weight

This can be useful when nearby observations should have stronger influence.

---

# 14. Feature Scaling is Important

KNN is distance-based.

Therefore, feature scales can strongly affect the result.

Suppose we have:

    Age:
        18 to 60

    Income:
        20,000 to 500,000

Income has much larger numerical values.

Without scaling, income can dominate the distance calculation.

This can cause KNN to pay too little attention to other features.

Therefore, feature scaling is usually important before applying KNN.

---

# 15. Standardization

Standardization transforms a feature approximately to:

    Mean = 0
    Standard deviation = 1

Formula:

    z = (x - μ) / σ

where:

    x = original value
    μ = mean
    σ = standard deviation

This prevents features with larger numerical scales from dominating distance calculations.

---

# 16. Min-Max Scaling

Another option is Min-Max scaling.

Formula:

    x_scaled =
        (x - xmin) / (xmax - xmin)

This generally transforms values to:

    [0, 1]

Both standardization and Min-Max scaling can be useful for KNN.

The appropriate choice depends on the dataset.

---

# 17. KNN and the Curse of Dimensionality

KNN relies on meaningful distances.

As the number of features becomes very large, distance-based methods can become less effective.

This phenomenon is related to the:

    Curse of Dimensionality

With many dimensions:

- Points can become difficult to distinguish by distance.
- Computation increases.
- Nearest and farthest points may become less meaningfully separated.

Therefore, feature selection or dimensionality reduction may become useful for high-dimensional datasets.

---

# 18. Advantages of KNN

### 1. Simple

The algorithm is conceptually easy to understand.

### 2. Easy to implement

The core algorithm requires:

- Distance calculation
- Sorting
- Voting or averaging

### 3. No complicated training phase

The algorithm mainly stores the training data.

### 4. Can model non-linear decision boundaries

Because predictions depend on local neighborhoods, KNN can represent complex patterns.

### 5. Supports classification and regression

The same basic neighborhood idea can be used for both tasks.

---

# 19. Disadvantages of KNN

### 1. Prediction can be slow

For a new point, distances to many training observations may need to be calculated.

### 2. Sensitive to feature scaling

Different feature scales can distort distances.

### 3. Sensitive to irrelevant features

Irrelevant features can affect distance calculations.

### 4. Sensitive to K

A poor K value can cause overfitting or underfitting.

### 5. Memory requirement

The training data generally needs to be retained for prediction.

### 6. Can struggle with high-dimensional data

Distance relationships can become less useful as dimensionality increases.

---

# 20. KNN and Overfitting

KNN demonstrates the bias-variance trade-off clearly.

Very small K:

    Low bias
    High variance

Very large K:

    Higher bias
    Lower variance

Therefore:

    Small K → may overfit
    Large K → may underfit

This connects KNN to the lecture's discussion of overfitting, underfitting, bias, and variance.

---

# 21. KNN Workflow

A practical KNN classification workflow is:

    Dataset
       ↓
    Inspect data
       ↓
    Clean data
       ↓
    Separate features and target
       ↓
    Train/Test split
       ↓
    Feature scaling
       ↓
    Select K
       ↓
    Calculate distances
       ↓
    Find K nearest neighbors
       ↓
    Majority voting
       ↓
    Prediction
       ↓
    Evaluation

---

# 22. Important Implementation Detail: Scale After Splitting

When using a scaler:

1. Split the dataset into training and testing data.
2. Fit the scaler only on the training data.
3. Transform the training data.
4. Transform the test data using the same scaler.

Do not calculate scaling parameters using the complete dataset before the split.

Otherwise, information from the test set can leak into the training process.

This is called:

    Data Leakage

---

# 23. KNN Classification Using the Student Dataset

The repository contains:

    datasets/student_performance.csv

Relevant features include:

    hours
    attendance
    assignments

Target:

    score

Because score is continuous, this dataset naturally fits regression.

To demonstrate KNN classification, a categorical target can be created from the score, for example:

    score >= threshold → Pass
    score < threshold  → Fail

This converts the problem into a classification task.

The transformation should be clearly documented rather than treating the original continuous score as a categorical label.

---

# 24. KNN from Scratch

A basic KNN classifier from scratch needs only a few major components:

    1. Store X_train and y_train
    2. Calculate distances
    3. Sort distances
    4. Select K nearest observations
    5. Count class votes
    6. Return the majority class

Conceptually:

    distances = []

    for every training point:
        calculate distance
        store distance and label

    sort by distance

    nearest = first K points

    prediction = majority label

---

# 25. Pseudocode

    KNN(X_train, y_train, X_test, K):

        predictions = []

        for each test_point in X_test:

            distances = []

            for each training_point in X_train:

                distance =
                    Euclidean(test_point, training_point)

                store(distance, training_label)

            sort distances

            neighbors = first K observations

            prediction =
                most common class among neighbors

            append prediction

        return predictions

---

# 26. Computational Cost

Suppose:

    n = number of training samples
    d = number of features
    m = number of test samples

For a straightforward implementation, prediction requires approximately:

    O(m × n × d)

distance calculations.

This is one reason KNN can become expensive when the dataset is large.

---

# 27. Important Terms

### Neighbor

A training observation close to the new observation.

### K

Number of neighbors considered.

### Distance Metric

Method used to measure similarity/distance.

Examples:

    Euclidean
    Manhattan

### Majority Voting

The most frequent class among the K neighbors.

### Lazy Learning

Little model-building work during training; computation occurs mainly during prediction.

### Feature Scaling

Transforming features so that numerical magnitude does not unfairly dominate distance.

---

# 28. KNN vs Other Algorithms

| Algorithm | Main Idea |
|---|---|
| Naive Bayes | Probability and Bayes theorem |
| KNN | Similarity to nearby observations |
| Decision Tree | Series of decision rules |
| Logistic Regression | Predict class probability using a mathematical model |
| SVM | Find a separating decision boundary |

KNN is fundamentally different from Naive Bayes.

Naive Bayes estimates probabilities.

KNN compares distances between observations.

---

# 29. Exam-Oriented Algorithm

For KNN classification:

1. Choose K.
2. Calculate the distance between the test point and every training point.
3. Sort the distances.
4. Select the K nearest points.
5. Count their class labels.
6. Assign the majority class.
7. Repeat for every test point.

Remember:

    KNN = Distance + Neighbors + Voting

---

# 30. Key Formulas

## Euclidean Distance

    d(x, y)
    = sqrt(Σ(xi - yi)^2)

For two dimensions:

    d
    = sqrt((x2 - x1)^2 + (y2 - y1)^2)

---

## Manhattan Distance

    d(x, y)
    = Σ|xi - yi|

For two dimensions:

    d
    = |x2 - x1| + |y2 - y1|

---

## Standardization

    z = (x - μ) / σ

---

## Min-Max Scaling

    x_scaled =
        (x - xmin) / (xmax - xmin)

---

# 31. Final Summary

K-Nearest Neighbors is a supervised, distance-based learning algorithm.

Core idea:

    Nearby points → Similar predictions

For classification:

    K nearest neighbors
             ↓
       Majority voting
             ↓
         Class label

For regression:

    K nearest neighbors
             ↓
          Average
             ↓
       Numeric prediction

The most important things to remember are:

- K determines how many neighbors are considered.
- Small K can overfit.
- Large K can underfit.
- Distance metrics determine which points are considered neighbors.
- Euclidean distance is commonly used.
- Manhattan distance is another option.
- Feature scaling is important.
- KNN has little training computation.
- KNN can be computationally expensive during prediction.
- Irrelevant features can negatively affect distance.
- High-dimensional data can make nearest-neighbor relationships less useful.

The central idea is:

    "Predict a new observation using the behavior of its nearest training observations."
