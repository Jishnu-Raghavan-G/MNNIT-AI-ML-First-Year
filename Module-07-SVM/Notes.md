# Module 07 — Support Vector Machine (SVM)

> The supplied MNNIT AI-ML lecture deck lists Support Vector Machine (SVM) among the classification algorithms, but it does not contain a detailed SVM section or a worked SVM derivation.
>
> Therefore, this module expands SVM as an implementation-oriented classification topic while keeping the course's supervised-learning, classification, overfitting, bias-variance, and evaluation framework.

---

# 1. What is Support Vector Machine?

Support Vector Machine (SVM) is a supervised machine-learning algorithm used mainly for classification.

The central idea of SVM is:

    Find a decision boundary
    that separates classes
    with the largest possible margin.

For binary classification:

    Class +1
    Class -1

SVM attempts to find a boundary that separates these classes while keeping the boundary as far as possible from the closest training observations.

---

# 2. Classification with a Decision Boundary

For two-dimensional data, a linear decision boundary can be written as:

    w1x1 + w2x2 + b = 0

or:

    wᵀx + b = 0

where:

    w = weight vector
    x = feature vector
    b = bias/intercept

The boundary separates the feature space into two regions.

For example:

    wᵀx + b > 0
        → Class +1

    wᵀx + b < 0
        → Class -1

---

# 3. Why SVM is Different

Many classification algorithms attempt to find a boundary that separates the classes.

SVM goes further.

It tries to find a boundary with the:

    Maximum Margin

The margin is the distance between the decision boundary and the closest training points.

The closest points are especially important.

These are called:

    Support Vectors

---

# 4. What is a Support Vector?

A Support Vector is a training observation that lies closest to the decision boundary.

These points determine the position and orientation of the optimal separating hyperplane.

Conceptually:

    Class +1

       ●
        \
         ●  ← Support Vector
          \
-----------\----------------
            \  Decision
             \ Boundary
          ●  ← Support Vector
         /
        ●

    Class -1

The observations far away from the boundary generally do not determine the margin directly.

---

# 5. Margin

The margin is the distance between the decision boundary and the closest observations from either class.

SVM attempts to maximize this distance.

Conceptually:

    Class +1
       ●
       |
       |   Margin
       |
-------|---------------- Decision Boundary
       |
       |   Margin
       |
       ●
    Class -1

A larger margin generally provides better separation between classes.

---

# 6. Maximum Margin Principle

Suppose several lines can separate two classes.

Example:

    Boundary A → small margin
    Boundary B → medium margin
    Boundary C → large margin

SVM prefers:

    Boundary C

because it maximizes the margin.

The intuition is:

    Larger margin
        ↓
    More separation
        ↓
    Potentially better generalization

---

# 7. Hyperplane

In two dimensions, the decision boundary is a line.

In three dimensions, it is a plane.

In higher dimensions, it is called a:

    Hyperplane

General equation:

    wᵀx + b = 0

where:

    w = normal vector to the hyperplane
    b = bias

---

# 8. Classification Rule

For binary SVM, labels are commonly represented as:

    y ∈ {-1, +1}

The decision function is:

    f(x) = wᵀx + b

Prediction:

    if f(x) >= 0:
        +1

    else:
        -1

Therefore:

    Sign(wᵀx + b)

determines the predicted class.

---

# 9. Canonical Margin Equations

For a linearly separable dataset, the SVM can be represented using:

    wᵀx + b = +1

and:

    wᵀx + b = -1

These two lines define the margin boundaries.

The central decision boundary is:

    wᵀx + b = 0

Therefore:

    +1 margin boundary
    → wᵀx + b = +1

    Decision boundary
    → wᵀx + b = 0

    -1 margin boundary
    → wᵀx + b = -1

---

# 10. Geometric Margin

The distance between:

    wᵀx + b = +1

and:

    wᵀx + b = -1

is:

    2 / ||w||

where:

    ||w|| = magnitude of w

Therefore:

    Margin = 2 / ||w||

To maximize the margin:

    maximize 2 / ||w||

which is equivalent to minimizing:

    ||w||

or more conveniently:

    1/2 ||w||²

---

# 11. Hard-Margin SVM

Hard-margin SVM assumes the training data can be perfectly separated by a hyperplane.

The constraints are:

    yi(wᵀxi + b) >= 1

for every training observation.

The optimization problem is:

    minimize:

        1/2 ||w||²

subject to:

    yi(wᵀxi + b) >= 1

The objective encourages a large margin.

---

# 12. Why Use 1/2 ||w||²?

The margin is:

    2 / ||w||

Maximizing the margin is equivalent to minimizing:

    ||w||

Since the square function preserves the ordering for non-negative values, we can minimize:

    ||w||²

Using:

    1/2 ||w||²

makes derivatives simpler.

Therefore, the standard optimization objective is:

    minimize 1/2 ||w||²

---

# 13. Problem with Hard Margin

Real-world data is often not perfectly separable.

Examples:

- Noise
- Outliers
- Overlapping classes
- Measurement errors

A hard-margin SVM cannot tolerate classification errors.

Therefore, we introduce:

    Soft-Margin SVM

---

# 14. Soft-Margin SVM

Soft-margin SVM allows some observations to violate the margin.

This is controlled using:

    Slack variables

Usually represented by:

    ξi

The constraints become:

    yi(wᵀxi + b)
    >= 1 - ξi

where:

    ξi >= 0

The slack variables allow the model to tolerate violations.

---

# 15. SVM Objective with Slack Variables

A common soft-margin objective is:

    minimize:

        1/2 ||w||²
        + C Σξi

where:

    C = regularization parameter
    ξi = slack variables

There is a trade-off between:

    Large margin

and:

    Classification violations

---

# 16. Meaning of C

C controls how strongly the model penalizes margin violations.

### Large C

Large penalty for errors.

The model tries harder to classify training observations correctly.

This can produce:

    Smaller margin
    Less tolerance for errors
    Potentially higher variance

### Small C

Smaller penalty for errors.

The model allows more violations in exchange for a wider margin.

This can produce:

    Larger margin
    More tolerance for errors
    Potentially better generalization

---

# 17. C and Overfitting

The parameter C affects model complexity.

Very large C:

    Strongly prioritize training correctness
        ↓
    Can produce a more complex boundary
        ↓
    Potential overfitting

Smaller C:

    Allow more training violations
        ↓
    Prefer a wider margin
        ↓
    More regularization

Therefore, C should be selected using validation or cross-validation.

---

# 18. Hinge Loss

SVM classification is closely associated with hinge loss.

For one observation:

    Loss = max(0, 1 - y f(x))

where:

    y ∈ {-1, +1}

and:

    f(x) = wᵀx + b

If:

    y f(x) >= 1

then:

    Loss = 0

The observation is correctly classified with sufficient margin.

---

# 19. Hinge Loss Example

Suppose:

    y = +1

and:

    f(x) = 2

Then:

    y f(x) = 2

Hinge loss:

    max(0, 1 - 2)
    = max(0, -1)
    = 0

The point satisfies the margin.

---

Suppose:

    y = +1

and:

    f(x) = 0.4

Then:

    y f(x) = 0.4

Hinge loss:

    max(0, 1 - 0.4)
    = 0.6

The point is correctly classified but lies inside the desired margin.

---

Suppose:

    y = +1

and:

    f(x) = -0.5

Then:

    y f(x) = -0.5

Hinge loss:

    max(0, 1 - (-0.5))
    = 1.5

The point is misclassified.

---

# 20. SVM Optimization Intuition

A soft-margin SVM can be understood as balancing:

    Large margin
          +
    Few classification violations

The objective combines:

    Model complexity
          +
    Training violations

Conceptually:

    Minimize:

        margin penalty
        +
        error penalty

The parameter C controls the relative importance of these terms.

---

# 21. Linear SVM

A linear SVM uses:

    wᵀx + b

as its decision function.

It is appropriate when the classes can be separated reasonably well by a linear boundary.

Example:

    Class A        Class B

       ● ●            ○ ○
       ● ●            ○ ○

             |
             |
             |
          Boundary

The boundary is a straight line in two dimensions.

---

# 22. Non-Linear Classification

Some datasets cannot be separated by a straight line.

Example:

          ○ ○
       ○       ○
      ○    ●    ○
       ○       ○
          ○ ○

A linear boundary may not separate the classes effectively.

SVM can handle such problems using:

    Kernel functions

---

# 23. Kernel Trick

The Kernel Trick allows SVM to operate as if the data were transformed into a higher-dimensional feature space without explicitly computing that transformation.

The kernel measures relationships between observations.

Common kernels include:

    Linear
    Polynomial
    RBF
    Sigmoid

The most commonly encountered non-linear kernel is:

    RBF
    (Radial Basis Function)

---

# 24. Linear Kernel

The linear kernel is:

    K(x, z) = xᵀz

It corresponds to a linear decision boundary in the original feature space.

Use it when:

    Data is approximately linearly separable
    or
    A linear model is sufficient

---

# 25. Polynomial Kernel

The polynomial kernel can model more complex relationships.

General form:

    K(x, z)
    = (γ xᵀz + r)^d

where:

    γ = scaling parameter
    r = coefficient
    d = polynomial degree

Increasing the degree can make the decision boundary more flexible.

---

# 26. RBF Kernel

The Radial Basis Function kernel is:

    K(x, z)
    =
    exp(
        -γ ||x - z||²
    )

where:

    γ = kernel coefficient

The RBF kernel can create non-linear decision boundaries.

---

# 27. Gamma in RBF SVM

Gamma controls how strongly individual training points influence the decision boundary.

High gamma:

    Narrow influence
        ↓
    More local behavior
        ↓
    More complex boundary
        ↓
    Potential overfitting

Low gamma:

    Wider influence
        ↓
    Smoother boundary
        ↓
    Simpler model
        ↓
    Potential underfitting

Therefore:

    C and gamma

are both important SVM hyperparameters when using an RBF kernel.

---

# 28. SVM Feature Scaling

Feature scaling is generally important for SVM.

Suppose:

    Feature 1:
        0 to 10

    Feature 2:
        0 to 1,000,000

A feature with a much larger numerical scale can dominate the geometry of the optimization.

Standardization is commonly used:

    z = (x - μ) / σ

Scaling is especially important for distance/kernel-based calculations and helps optimization behave more consistently.

---

# 29. SVM and Support Vectors

Only certain observations directly determine the optimal margin.

These are:

    Support Vectors

Points far away from the boundary generally have less direct influence on the final separating hyperplane.

Therefore:

    SVM
      ↓
    Find separating boundary
      ↓
    Identify critical boundary points
      ↓
    Maximize margin

---

# 30. Example of Support Vectors

Suppose:

    Class +1:

        ●
        ●
             ●  ← support vector

    ---------------- decision boundary

             ○  ← support vector
        ○
        ○

The points closest to the boundary are the most important for defining the margin.

---

# 31. Decision Boundary and Margin

For a linear SVM:

    Decision boundary:

        wᵀx + b = 0

    Positive margin:

        wᵀx + b = +1

    Negative margin:

        wᵀx + b = -1

    Margin width:

        2 / ||w||

The objective is to maximize the margin while controlling classification violations.

---

# 32. SVM Classification Workflow

A typical workflow is:

    Dataset
       ↓
    Clean data
       ↓
    Separate X and y
       ↓
    Encode binary labels if necessary
       ↓
    Train/Test split
       ↓
    Feature scaling
       ↓
    Select kernel
       ↓
    Select C
       ↓
    Train SVM
       ↓
    Predict
       ↓
    Evaluate

---

# 33. SVM From Scratch vs Library Implementation

A complete SVM optimizer requires solving a constrained optimization problem.

Therefore, a basic educational demonstration can focus on:

- Linear decision boundary
- Margin boundaries
- Support vectors
- Hinge loss
- Effect of C
- Effect of feature scaling

Production implementations should normally use a well-tested optimization library.

The repository code for this module therefore focuses on an interpretable margin demonstration rather than pretending to implement a complete industrial-strength SVM solver from scratch.

---

# 34. SVM Margin Demonstration

For a simple two-dimensional dataset, we can define a separating hyperplane:

    wᵀx + b = 0

Then calculate:

    score = wᵀx + b

and classify using:

    score >= 0 → +1
    score < 0  → -1

The margin boundaries are:

    score = +1
    score = -1

The distance from a point to the decision boundary is related to:

    |wᵀx + b| / ||w||

This makes the geometry of SVM easier to understand.

---

# 35. Distance from a Point to a Hyperplane

For:

    wᵀx + b = 0

the perpendicular distance of point x from the hyperplane is:

    distance =
        |wᵀx + b| / ||w||

This formula is useful for understanding the SVM margin.

For points satisfying:

    wᵀx + b = ±1

the distance to the decision boundary is:

    1 / ||w||

Therefore, the total margin width is:

    2 / ||w||

---

# 36. SVM and Generalization

The maximum-margin idea is related to generalization.

Instead of simply trying to classify training data correctly, SVM seeks a boundary that separates the classes with a substantial margin.

The intuition is:

    Better separation
        ↓
    Less sensitivity to small changes
        ↓
    Potentially better performance on unseen data

This does not guarantee good generalization; hyperparameters, data quality, and feature representation still matter.

---

# 37. SVM Advantages

- Effective for classification.
- Maximum-margin principle provides a strong geometric interpretation.
- Works well in high-dimensional feature spaces.
- Can model linear and non-linear decision boundaries.
- Kernel methods allow flexible non-linear classification.
- Support vectors provide an intuitive explanation of important training observations.
- Regularization through C helps control model complexity.

---

# 38. SVM Disadvantages

- Can be computationally expensive for very large datasets.
- Requires careful hyperparameter selection.
- Feature scaling is generally important.
- Kernel selection can be difficult.
- Non-linear SVMs can become difficult to interpret.
- Probability estimates are not the primary output of the basic SVM formulation.

---

# 39. SVM vs Logistic Regression

| Property | SVM | Logistic Regression |
|---|---|---|
| Main idea | Maximum-margin separation | Probability through sigmoid |
| Basic boundary | Linear | Linear |
| Non-linear extension | Kernels | Feature engineering / transformations |
| Output | Class/decision score | Probability |
| Loss | Hinge loss | Log loss |
| Regularization | C | Regularization strength |
| Important points | Support vectors | Training observations contribute to optimization |

Both can produce linear decision boundaries, but they optimize different objectives.

---

# 40. SVM vs KNN

| Property | SVM | KNN |
|---|---|---|
| Main idea | Maximum margin | Nearest neighbors |
| Training | Optimization | Mainly stores data |
| Prediction | Evaluate decision function | Calculate distances |
| Scaling | Important | Important |
| Non-linear modeling | Kernels | Naturally local |
| Hyperparameter | C, kernel, gamma | K, distance metric |
| Large datasets | Can be expensive | Prediction can be expensive |

---

# 41. SVM vs Decision Tree

| Property | SVM | Decision Tree |
|---|---|---|
| Main idea | Maximum-margin boundary | Recursive decisions |
| Scaling | Usually important | Usually unnecessary |
| Boundary | Linear or kernel-based | Piecewise decision regions |
| Interpretability | Moderate/low | High |
| Overfitting control | C, kernel parameters | Depth/pruning constraints |
| Non-linear patterns | Kernels | Naturally supported |

---

# 42. Important Hyperparameters

### C

Controls the penalty for classification and margin violations.

### Kernel

Determines the type of decision function.

Common choices:

    linear
    polynomial
    rbf
    sigmoid

### Gamma

Important for kernels such as RBF.

Controls the influence range of individual observations.

### Degree

Important for polynomial kernels.

Controls polynomial complexity.

---

# 43. Common Mistakes

### Mistake 1: Thinking SVM simply finds any separating line

SVM specifically seeks a maximum-margin solution.

### Mistake 2: Ignoring scaling

SVM is sensitive to feature scales.

### Mistake 3: Thinking support vectors are all training points

Support vectors are the critical observations associated with the margin.

### Mistake 4: Using a huge C automatically

Large C is not always better.

It can encourage fitting training data more aggressively.

### Mistake 5: Using large gamma without considering overfitting

Large gamma can create highly local and complex decision boundaries.

---

# 44. Exam-Oriented Summary

Remember:

    SVM
     ↓
    Find separating hyperplane
     ↓
    Maximize margin
     ↓
    Support vectors define the margin
     ↓
    Use soft margin when perfect separation
    is not possible

Core decision function:

    f(x) = wᵀx + b

Prediction:

    sign(f(x))

Margin boundaries:

    wᵀx + b = +1

    wᵀx + b = -1

Decision boundary:

    wᵀx + b = 0

Margin:

    2 / ||w||

Soft-margin objective:

    1/2 ||w||² + CΣξi

Hinge loss:

    max(0, 1 - y f(x))

---

# 45. Key Points to Remember

- SVM is a supervised-learning classification algorithm.
- It attempts to find a maximum-margin decision boundary.
- The closest training observations are called support vectors.
- A linear SVM uses a hyperplane.
- Hard-margin SVM assumes perfect separability.
- Soft-margin SVM allows margin violations.
- C controls the penalty for violations.
- Hinge loss is associated with SVM classification.
- Kernel functions allow non-linear decision boundaries.
- RBF is a commonly used non-linear kernel.
- Gamma controls the locality of an RBF kernel.
- Feature scaling is generally important.
- SVM can work well in high-dimensional spaces.
- Hyperparameter selection is important for good generalization.

The central idea is:

    "Find the decision boundary that separates the classes with the largest possible margin, while allowing controlled violations when necessary."
