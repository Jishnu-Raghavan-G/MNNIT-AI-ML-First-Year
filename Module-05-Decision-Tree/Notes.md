# Module 05 — Decision Tree

> The supplied MNNIT AI-ML lecture deck lists Decision Tree as one of the classification algorithms, but it does not contain a detailed Decision Tree section or a worked Decision Tree construction example.
> This module therefore provides implementation-oriented Decision Tree notes while preserving the course's classification, supervised-learning, overfitting, and evaluation framing.

---

# 1. What is a Decision Tree?

A Decision Tree is a supervised machine-learning algorithm used for:

- Classification
- Regression

It makes predictions by repeatedly asking questions about the input features.

The resulting model looks like a tree.

Basic structure:

    Root
     |
     +---- Decision
     |       |
     |       +---- Branch
     |       |
     |       +---- Branch
     |
     +---- Decision
             |
             +---- Prediction

The final prediction is made at a leaf node.

---

# 2. Why is it Called a Tree?

A Decision Tree resembles an upside-down tree.

It contains:

### Root Node

The first and most important decision.

### Internal Node

A point where another decision is made.

### Branch

The outcome of a decision.

### Leaf Node

The final prediction.

Example:

    Outlook?
       |
       +---- Sunny
       |      |
       |      +---- Humidity > 70 → No
       |      |
       |      +---- Humidity <= 70 → Yes
       |
       +---- Overcast → Yes
       |
       +---- Rainy
              |
              +---- Windy → No
              |
              +---- Not Windy → Yes

---

# 3. Decision Trees for Classification

For classification, the tree predicts a discrete class.

Examples:

    Spam / Ham

    Pass / Fail

    Disease / No Disease

    Cat / Dog

The model follows a path from the root to a leaf.

Example:

    Attendance > 75?
          |
        Yes
          |
    Assignments > 6?
       /          \
     Yes           No
      |             |
    Pass           Fail

For a new observation, the appropriate path is followed until a leaf is reached.

---

# 4. Decision Trees for Regression

Decision Trees can also predict continuous values.

Instead of a class label, a leaf contains a numerical prediction.

Example:

    House area > 1500?
          |
       Yes
          |
    Bedrooms > 3?
       /        \
     Yes         No
      |           |
    85 lakh     70 lakh

The exact leaf prediction depends on the training observations reaching that leaf.

---

# 5. Main Idea of a Decision Tree

The goal is to divide the dataset into groups that become increasingly pure.

For classification:

    Mixed classes
         ↓
    Choose useful feature/split
         ↓
    Divide data
         ↓
    More homogeneous groups
         ↓
    Repeat
         ↓
    Leaf predictions

A good split separates different classes effectively.

---

# 6. What is a Split?

A split divides the training data according to a feature.

For a numerical feature:

    Age <= 30
    Age > 30

For another feature:

    Income <= 50000
    Income > 50000

For a categorical feature:

    Weather = Sunny
    Weather = Rainy
    Weather = Overcast

The algorithm searches for useful splits.

---

# 7. Impurity

A Decision Tree needs a way to measure how mixed the classes are inside a node.

This is called impurity.

A node containing only one class is pure.

Example:

    10 Yes
     0 No

This node is completely pure.

A node containing:

     5 Yes
     5 No

is highly mixed.

Common impurity measures include:

- Gini impurity
- Entropy

---

# 8. Gini Impurity

Gini impurity is commonly used for classification trees.

Formula:

    Gini = 1 - Σ pi²

where:

    pi = proportion of class i in the node

For two classes:

    Gini = 1 - (p1² + p2²)

---

# 9. Gini Example

Suppose a node contains:

    6 Yes
    4 No

Total:

    10

Therefore:

    P(Yes) = 6/10 = 0.6

    P(No) = 4/10 = 0.4

Gini:

    = 1 - (0.6² + 0.4²)

    = 1 - (0.36 + 0.16)

    = 1 - 0.52

    = 0.48

Therefore:

    Gini impurity = 0.48

---

# 10. Pure Node

Suppose:

    10 Yes
     0 No

Then:

    P(Yes) = 1
    P(No) = 0

Gini:

    = 1 - (1² + 0²)

    = 0

Therefore:

    Gini = 0

A Gini value of zero means the node is perfectly pure.

---

# 11. Maximum Gini for Binary Classification

For two classes, the maximum Gini impurity occurs when the classes are equally represented.

For:

    P(Yes) = 0.5
    P(No) = 0.5

Gini:

    = 1 - (0.5² + 0.5²)

    = 1 - (0.25 + 0.25)

    = 0.5

Therefore, for binary classification:

    Minimum Gini = 0
    Maximum Gini = 0.5

---

# 12. Entropy

Entropy is another impurity measure.

Formula:

    Entropy = -Σ pi log2(pi)

For two classes:

    Entropy =
        -p1 log2(p1)
        -p2 log2(p2)

Entropy measures uncertainty.

---

# 13. Entropy Example

Suppose:

    6 Yes
    4 No

Then:

    P(Yes) = 0.6
    P(No) = 0.4

Entropy:

    = -(0.6 log2(0.6))
      -(0.4 log2(0.4))

Approximately:

    Entropy ≈ 0.971

A pure node has:

    Entropy = 0

---

# 14. Information Gain

Information Gain measures how much a split reduces uncertainty.

Basic idea:

    Information Gain
    =
    Parent Impurity
    -
    Weighted Child Impurity

For entropy:

    IG =
        Entropy(parent)
        -
        Weighted Entropy(children)

A split with higher information gain is generally preferred.

---

# 15. Weighted Child Impurity

Suppose a split creates two child nodes.

Child 1:

    6 samples

Child 2:

    4 samples

Total:

    10 samples

Then the weighted impurity is:

    (6/10) × Impurity(child 1)
    +
    (4/10) × Impurity(child 2)

The weighting matters because larger child nodes contain more observations.

---

# 16. Gini-Based Split Selection

When using Gini impurity, the algorithm can select the split that minimizes weighted Gini impurity.

For a split:

    Weighted Gini
    =
    (Nleft / N) × Gini(left)
    +
    (Nright / N) × Gini(right)

Choose the split with the lowest weighted impurity.

---

# 17. Entropy-Based Split Selection

When using entropy:

    Information Gain
    =
    Parent Entropy
    -
    Weighted Child Entropy

Choose a split that produces high information gain.

Therefore:

    High Information Gain
          ↓
    Large reduction in uncertainty
          ↓
    Good split

---

# 18. Example of Split Selection

Suppose we have two candidate splits:

    Split A → weighted Gini = 0.20

    Split B → weighted Gini = 0.35

For a Gini-based tree:

    Split A is preferred

because:

    0.20 < 0.35

The algorithm attempts to create purer child nodes.

---

# 19. Recursive Tree Construction

A Decision Tree is built recursively.

Basic process:

    Start with entire dataset
             ↓
    Find best split
             ↓
    Divide dataset
          /     \
       Left     Right
        ↓         ↓
    Find best   Find best
      split       split
        ↓         ↓
      ...       ...
        ↓         ↓
      Leaves    Leaves

The process continues until stopping conditions are reached.

---

# 20. Stopping Conditions

A tree does not necessarily grow forever.

Common stopping conditions include:

### Maximum Depth

Stop after reaching a specified depth.

### Minimum Samples per Split

Do not split a node unless it contains enough samples.

### Minimum Samples per Leaf

Require a minimum number of samples in each leaf.

### Pure Node

Stop when all observations in a node belong to the same class.

### No Useful Split

Stop if no meaningful split can improve the tree.

---

# 21. Maximum Depth

Maximum depth controls how deep the tree can become.

Example:

    max_depth = 2

means the tree cannot grow beyond depth 2.

A small depth:

    → simpler tree
    → potentially higher bias
    → potentially lower variance

A large depth:

    → more complex tree
    → potentially lower bias
    → potentially higher variance

---

# 22. Overfitting in Decision Trees

Decision Trees can easily overfit.

An excessively deep tree may memorize the training dataset.

Example:

    Training accuracy = 100%

but:

    Test accuracy = 70%

This indicates poor generalization.

The tree has learned training-specific patterns rather than only the general relationship.

---

# 23. Controlling Overfitting

Common methods include:

- Limit maximum depth.
- Increase minimum samples required for splitting.
- Increase minimum samples required in leaves.
- Prune unnecessary branches.
- Use validation data or cross-validation.
- Remove irrelevant features.

These ideas connect directly to the lecture's discussion of overfitting and methods for improving generalization.

---

# 24. Decision Tree Pruning

Pruning means removing unnecessary parts of a tree.

Instead of allowing:

    Very deep tree
         ↓
    Many tiny leaves

we simplify it:

    Smaller tree
         ↓
    Better generalization

Pruning can reduce overfitting.

---

# 25. Feature Importance

Decision Trees can provide a measure of feature importance.

A feature is considered important if it contributes significantly to reducing impurity across the tree.

For example:

    Feature A → high importance
    Feature B → medium importance
    Feature C → low importance

Feature importance can help understand which variables are most useful to the model.

---

# 26. Numerical Features

Decision Trees can naturally handle numerical features.

For example:

    hours <= 5.5
    hours > 5.5

The algorithm can test different thresholds and choose an effective split.

Unlike KNN, Decision Trees do not require distance calculations.

---

# 27. Categorical Features

Conceptually, Decision Trees can split categorical features into groups.

For example:

    Weather:

    Sunny
    Rainy
    Overcast

However, the exact handling of categorical variables depends on the implementation/library.

In a simple scratch implementation, numerical features are easier to support correctly.

---

# 28. Decision Tree Workflow

A typical classification workflow is:

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
    Build Decision Tree
       ↓
    Select useful splits
       ↓
    Grow tree recursively
       ↓
    Stop according to constraints
       ↓
    Predict
       ↓
    Evaluate

---

# 29. Decision Tree From Scratch

A basic Decision Tree implementation requires:

1. Calculate class impurity.
2. Try candidate splits.
3. Calculate the quality of each split.
4. Select the best split.
5. Divide the dataset.
6. Recursively build child nodes.
7. Stop when a stopping condition is reached.
8. Store a class prediction in leaf nodes.

For this repository implementation, Gini impurity is used.

---

# 30. Simplified Pseudocode

    build_tree(data):

        if stopping_condition:
            return leaf

        best_split = find_best_split(data)

        left_data, right_data = split(data)

        left_child = build_tree(left_data)

        right_child = build_tree(right_data)

        return node(
            best_split,
            left_child,
            right_child
        )

---

# 31. Prediction

After the tree has been trained, prediction is straightforward.

For a new observation:

    Start at root
          ↓
    Evaluate split
          ↓
    Choose left/right branch
          ↓
    Evaluate next split
          ↓
    Continue
          ↓
    Reach leaf
          ↓
    Return leaf prediction

---

# 32. Example Prediction

Suppose the tree contains:

    hours <= 5.5?
       |
       +---- Yes → Fail
       |
       +---- No
              |
              attendance <= 70?
                 |
                 +---- Yes → Fail
                 |
                 +---- No → Pass

For:

    hours = 7
    attendance = 85

First:

    7 <= 5.5 → False

Move right.

Then:

    85 <= 70 → False

Move to:

    Pass

Therefore:

    Prediction = Pass

---

# 33. Advantages of Decision Trees

### 1. Easy to Understand

The model can be represented as a sequence of decisions.

### 2. Interpretable

The path from root to leaf explains how a prediction was reached.

### 3. Handles Non-Linear Relationships

Decision boundaries do not need to be linear.

### 4. Little Mathematical Preprocessing

Decision Trees generally do not require feature scaling.

### 5. Works for Classification and Regression

The same general tree structure supports both tasks.

---

# 34. Disadvantages of Decision Trees

### 1. Can Overfit

Deep trees can memorize training data.

### 2. Unstable

Small changes in training data can produce a different tree.

### 3. Greedy Construction

The standard tree-building process chooses locally good splits rather than searching every possible complete tree.

### 4. Very Deep Trees Become Difficult to Interpret

Although trees are interpretable, extremely large trees lose this advantage.

---

# 35. Decision Tree vs KNN

| Property | Decision Tree | KNN |
|---|---|---|
| Main idea | Decision rules | Nearby observations |
| Training | Builds a tree | Stores data |
| Prediction | Follow tree path | Calculate distances |
| Scaling required | Usually no | Usually important |
| Overfitting | Deep trees can overfit | Small K can overfit |
| Interpretability | High | Lower |
| Classification | Yes | Yes |
| Regression | Yes | Yes |

---

# 36. Decision Tree vs Naive Bayes

| Property | Decision Tree | Naive Bayes |
|---|---|---|
| Main idea | Recursive decisions | Probabilities |
| Assumption | No independence assumption like NB | Conditional independence |
| Output | Class/value | Probabilities/class |
| Interpretability | High | Moderate |
| Training | Build tree | Estimate probabilities |
| Prediction | Traverse tree | Calculate class scores |

---

# 37. Important Hyperparameters

Common Decision Tree hyperparameters include:

    max_depth

    min_samples_split

    min_samples_leaf

    criterion

For classification, criterion can commonly be:

    gini
    entropy

The choice of hyperparameters affects model complexity and generalization.

---

# 38. Bias-Variance Perspective

Decision Trees demonstrate the bias-variance trade-off.

Very shallow tree:

    High bias
    Lower variance

Very deep tree:

    Lower training bias
    Higher variance

Therefore, tree depth should be controlled based on validation performance.

---

# 39. Classification Evaluation

After training a Decision Tree classifier, predictions can be evaluated using classification metrics.

Important metrics include:

    Accuracy
    Precision
    Recall
    F1-score

These metrics are covered in the lecture's classification evaluation section.

---

# 40. Accuracy

Accuracy measures the fraction of correct predictions.

Formula:

    Accuracy =
        Correct Predictions
        --------------------
        Total Predictions

Example:

    90 correct predictions
    100 total predictions

    Accuracy = 90 / 100
             = 0.90

Therefore:

    Accuracy = 90%

---

# 41. Decision Tree Exam Checklist

When solving a Decision Tree problem, remember:

1. Identify the target classes.
2. Calculate class counts.
3. Calculate impurity.
4. Consider candidate splits.
5. Calculate child impurities.
6. Calculate weighted impurity or information gain.
7. Select the best split.
8. Repeat recursively.
9. Apply stopping conditions.
10. Predict by following the root-to-leaf path.

---

# 42. Core Formulas

## Gini Impurity

    Gini = 1 - Σ pi²

---

## Entropy

    Entropy = -Σ pi log2(pi)

---

## Information Gain

    IG =
        Parent Entropy
        -
        Weighted Child Entropy

---

## Weighted Impurity

    Weighted Impurity =
        (Nleft / N) × Impurity(left)
        +
        (Nright / N) × Impurity(right)

---

# 43. Key Points to Remember

- Decision Tree is a supervised-learning algorithm.
- It can perform classification and regression.
- A tree consists of root nodes, internal nodes, branches, and leaves.
- Splits divide the dataset.
- Impurity measures how mixed the classes are.
- Gini impurity and entropy are common measures.
- Information gain measures reduction in entropy.
- Recursive splitting builds the tree.
- Maximum depth and other constraints control complexity.
- Very deep trees can overfit.
- Pruning can reduce unnecessary complexity.
- Feature scaling is generally not required.
- Decision Trees can represent non-linear decision boundaries.
- Predictions are made by traversing the tree from root to leaf.

The central idea is:

    "Keep splitting the data into purer groups until a useful prediction can be made at the leaves."
