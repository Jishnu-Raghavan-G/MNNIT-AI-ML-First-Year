# Module 06 — Logistic Regression

> The supplied MNNIT AI-ML lecture deck lists Logistic Regression among the classification algorithms and discusses classification as a supervised-learning task. However, the supplied slides do not contain a detailed Logistic Regression derivation or worked implementation.
>
> The notes below therefore expand Logistic Regression as an implementation-oriented module while keeping the course's classification, model evaluation, overfitting, and regularization framing.

---

# 1. What is Logistic Regression?

Logistic Regression is a supervised machine-learning algorithm mainly used for classification.

Despite its name, Logistic Regression is primarily a classification algorithm.

It estimates the probability that an observation belongs to a particular class.

For binary classification:

    Class 0
    or
    Class 1

Example:

    Student → Pass / Fail

    Email → Spam / Ham

    Patient → Disease / No Disease

The model first produces a probability and then converts that probability into a class using a threshold.

---

# 2. Logistic Regression and Classification

The lecture describes classification as assigning input data to discrete class values.

Examples include:

- Spam classification
- Offensive content classification
- Image classification

Classification can be:

- Binary classification
- Multi-class classification
- Multi-label classification

Logistic Regression is especially natural for binary classification.

Example:

    P(Pass) = 0.82

If the classification threshold is 0.5:

    0.82 >= 0.5

Therefore:

    Prediction = Pass

---

# 3. Why Not Use Linear Regression Directly?

Suppose we use a linear equation:

    z = β0 + β1x1 + β2x2 + ... + βnxn

The value of z can be any number:

    -10
    -2
     0
     4
    100

But a probability should lie between:

    0 and 1

Therefore, Logistic Regression transforms the linear output into a probability.

This transformation is performed using the sigmoid function.

---

# 4. Linear Combination

Logistic Regression first calculates:

    z = β0 + β1x1 + β2x2 + ... + βnxn

where:

    β0 = intercept
    β1, β2, ..., βn = model coefficients
    x1, x2, ..., xn = input features

The value z is sometimes called the:

    logit
    or
    linear score

The sigmoid function then converts z into a value between 0 and 1.

---

# 5. Sigmoid Function

The sigmoid function is:

    σ(z) = 1 / (1 + e^(-z))

Its output always lies between 0 and 1.

Therefore it is useful for representing a probability.

Examples:

    z = 0

    σ(0) = 0.5


    z > 0

    σ(z) > 0.5


    z < 0

    σ(z) < 0.5

As z becomes very large:

    σ(z) → 1

As z becomes very negative:

    σ(z) → 0

---

# 6. Sigmoid Intuition

The sigmoid function has an S-shaped curve.

Conceptually:

    Probability
       1 |                 ______
         |              __/
         |            _/
     0.5 |-----------/
         |         _/
         |      __/
       0 |_____/
         +----------------------
                  z

The important point is:

    Any real-valued z
          ↓
       sigmoid
          ↓
    Probability between 0 and 1

---

# 7. From Features to Probability

The complete process is:

    Input features
          ↓
    Linear combination
          ↓
    z = β0 + β1x1 + ... + βnxn
          ↓
       Sigmoid
          ↓
    Probability
          ↓
       Threshold
          ↓
      Class label

For example:

    z = 2

    P(y = 1 | x)
    = sigmoid(2)
    ≈ 0.881

Therefore:

    Probability of class 1 ≈ 88.1%

---

# 8. Classification Threshold

A common threshold is:

    0.5

Decision rule:

    if probability >= 0.5:
        class = 1

    else:
        class = 0

Example:

    probability = 0.73

    0.73 >= 0.5

    prediction = 1

Another example:

    probability = 0.31

    0.31 < 0.5

    prediction = 0

The threshold can be changed depending on the application.

---

# 9. Binary Logistic Regression

For binary classification:

    y ∈ {0, 1}

The model estimates:

    P(y = 1 | x)

Then:

    P(y = 0 | x)
    = 1 - P(y = 1 | x)

Example:

    P(y = 1) = 0.80

Therefore:

    P(y = 0) = 0.20

---

# 10. Log-Odds Interpretation

Logistic Regression can also be expressed using odds.

Odds are:

    odds = p / (1 - p)

The log-odds are:

    log(p / (1 - p))

Logistic Regression models the log-odds as a linear function:

    log(p / (1 - p))
    =
    β0 + β1x1 + ... + βnxn

This is one of the most important mathematical interpretations of Logistic Regression.

---

# 11. Interpreting Coefficients

Consider:

    log(p / (1 - p))
    =
    β0 + β1x1

If β1 is positive:

    Increasing x1
        ↓
    increases the log-odds
        ↓
    generally increases probability of class 1

If β1 is negative:

    Increasing x1
        ↓
    decreases the log-odds
        ↓
    generally decreases probability of class 1

The size of a coefficient indicates how strongly the corresponding feature affects the log-odds, with the usual caveat that feature scale matters.

---

# 12. Example of Coefficient Interpretation

Suppose:

    z = -3 + 0.8x

The coefficient is:

    β1 = 0.8

This means that increasing x by one unit increases the log-odds by 0.8.

It does NOT mean that the probability increases by exactly 0.8.

The sigmoid transformation makes the relationship between a feature and probability non-linear.

---

# 13. Decision Boundary

The decision boundary separates different predicted classes.

With a threshold of 0.5:

    sigmoid(z) = 0.5

This occurs when:

    z = 0

Therefore the decision boundary is:

    β0 + β1x1 + ... + βnxn = 0

For two features:

    β0 + β1x1 + β2x2 = 0

This produces a linear decision boundary.

---

# 14. Logistic Regression as a Linear Classifier

Although Logistic Regression uses the sigmoid function, its decision boundary is linear in the original feature space.

For example:

    β0 + β1x1 + β2x2 = 0

defines a line in two dimensions.

For more features, it becomes a hyperplane.

Therefore:

    Logistic Regression
          ↓
    Linear decision boundary
          ↓
    Probability through sigmoid

---

# 15. Cost Function

The model needs a way to measure how well its predicted probabilities match the true labels.

For Logistic Regression, the common loss is:

    Binary Cross-Entropy

For one observation:

    Loss =
        -[y log(p) + (1-y) log(1-p)]

where:

    y = actual class
    p = predicted probability of class 1

For the entire dataset:

    J(β)
    =
    -(1/n) Σ[
        yi log(pi)
        +
        (1-yi)log(1-pi)
    ]

The goal is to minimize this loss.

---

# 16. Why Not Use Mean Squared Error?

Mean Squared Error can be used in some classification contexts, but Logistic Regression is normally trained using log loss / binary cross-entropy.

Cross-entropy is particularly appropriate for probabilistic classification.

It strongly penalizes confident incorrect predictions.

Example:

    Actual class = 1

Prediction:

    p = 0.99

Very good.

Prediction:

    p = 0.01

Very bad.

The loss reflects this difference.

---

# 17. Gradient Descent

One way to train Logistic Regression from scratch is Gradient Descent.

Basic idea:

    Start with initial coefficients
             ↓
    Calculate predictions
             ↓
    Calculate loss
             ↓
    Calculate gradients
             ↓
    Update coefficients
             ↓
    Repeat

The parameters gradually move toward values that minimize the loss.

---

# 18. Parameter Update

A generic Gradient Descent update is:

    β := β - learning_rate × gradient

where:

    β = model parameters
    learning_rate = step size
    gradient = direction of increasing loss

Therefore, subtracting the gradient moves the parameters toward lower loss.

---

# 19. Learning Rate

The learning rate controls how large each update is.

Small learning rate:

    Small updates
    → slower learning
    → potentially stable

Large learning rate:

    Large updates
    → faster movement
    → may overshoot the minimum

A suitable learning rate is important for successful optimization.

---

# 20. Logistic Regression Training From Scratch

A basic implementation can follow:

    Initialize weights
    Initialize bias

    Repeat for each iteration:

        1. Calculate z
        2. Apply sigmoid
        3. Calculate loss
        4. Calculate gradients
        5. Update weights
        6. Update bias

At the end:

    learned weights
    learned bias

can be used to make predictions.

---

# 21. Gradient for Logistic Regression

For binary Logistic Regression with cross-entropy loss, the gradient has a particularly simple form.

Let:

    p = sigmoid(Xβ + β0)

Then:

    gradient of weights
    =
    (1/n) Xᵀ(p - y)

and:

    gradient of bias
    =
    (1/n) Σ(p - y)

The parameters are then updated using Gradient Descent.

---

# 22. Numerical Stability

The sigmoid function involves:

    e^(-z)

For extremely large positive or negative values, direct computation can create numerical problems.

A stable implementation handles positive and negative z separately.

Similarly, when calculating logarithms in cross-entropy, probabilities should be clipped away from exactly:

    0
    and
    1

This avoids:

    log(0)

which is undefined.

---

# 23. Feature Scaling

Logistic Regression can benefit from feature scaling, especially when Gradient Descent is used.

Suppose:

    Feature 1 → 0 to 10
    Feature 2 → 0 to 1,000,000

The large-scale feature can affect optimization disproportionately.

Standardization can make optimization more well-behaved.

Standardization:

    z = (x - μ) / σ

However, scaling is not mathematically required in every Logistic Regression implementation.

---

# 24. Regularization

Logistic Regression can use regularization to control model complexity.

Two common forms are:

    L1 regularization
    L2 regularization

Regularization discourages excessively large coefficients.

This can improve generalization and reduce overfitting.

The lecture specifically discusses L1 and L2 regularization as methods for controlling overfitting.

---

# 25. L1 Regularization

L1 regularization adds a penalty related to the absolute values of coefficients.

Conceptually:

    Loss
    +
    λ Σ|βj|

where:

    λ = regularization strength

L1 regularization can encourage some coefficients to become exactly zero.

This can effectively perform feature selection.

---

# 26. L2 Regularization

L2 regularization adds a penalty related to squared coefficients.

Conceptually:

    Loss
    +
    λ Σβj²

L2 encourages coefficients to remain smaller.

It generally does not force coefficients exactly to zero as strongly as L1.

---

# 27. Regularization Strength

The parameter controlling regularization is often represented by:

    λ

Large λ:

    Stronger penalty
    → simpler model
    → may underfit if excessive

Small λ:

    Weaker penalty
    → more flexible model
    → may overfit if too weak

Therefore regularization must be chosen carefully.

---

# 28. Overfitting and Underfitting

The lecture discusses:

    Overfitting
    Underfitting
    Bias
    Variance

These concepts also apply to Logistic Regression.

Overfitting:

    Model performs well on training data
    but poorly on unseen data.

Underfitting:

    Model performs poorly even on training data.

Regularization can help reduce overfitting.

---

# 29. Multiclass Logistic Regression

Binary Logistic Regression distinguishes between two classes.

For multiple classes, Logistic Regression can be extended using approaches such as:

    One-vs-Rest
    or
    Multinomial Logistic Regression

Example:

    Class 0
    Class 1
    Class 2

The model can produce probabilities for each class.

The predicted class is typically the class with the highest probability.

---

# 30. Logistic Regression Example

Suppose a model predicts whether a student passes.

Features:

    x1 = study hours
    x2 = attendance

Model:

    z = -8 + 0.9x1 + 0.05x2

For:

    study hours = 7
    attendance = 85

Calculate:

    z
    = -8 + 0.9(7) + 0.05(85)

    = -8 + 6.3 + 4.25

    = 2.55

Apply sigmoid:

    p = 1 / (1 + e^(-2.55))

    p ≈ 0.928

Therefore:

    P(Pass) ≈ 92.8%

Using threshold 0.5:

    Prediction = Pass

---

# 31. Logistic Regression Workflow

A practical workflow is:

    Dataset
       ↓
    Inspect data
       ↓
    Clean data
       ↓
    Separate X and y
       ↓
    Train/Test split
       ↓
    Scale features if appropriate
       ↓
    Initialize parameters
       ↓
    Train using optimization
       ↓
    Calculate probabilities
       ↓
    Apply classification threshold
       ↓
    Evaluate model

---

# 32. Evaluation

For classification, useful metrics include:

    Accuracy
    Precision
    Recall
    F1-score

These metrics are part of the classification evaluation material in the lecture.

Accuracy:

    Correct / Total

Precision:

    TP / (TP + FP)

Recall:

    TP / (TP + FN)

F1-score:

    2 × Precision × Recall
    -----------------------
    Precision + Recall

The appropriate metric depends on the application.

---

# 33. Confusion Matrix

For binary classification:

                    Actual
                  1       0

Predicted 1      TP      FP

Predicted 0      FN      TN

where:

    TP = True Positive
    TN = True Negative
    FP = False Positive
    FN = False Negative

These four values form the basis of several classification metrics.

---

# 34. Advantages of Logistic Regression

- Simple and efficient.
- Good baseline for classification.
- Produces probabilities.
- Easy to interpret compared with many complex models.
- Works well when the classes are reasonably separable by a linear boundary.
- Regularization can control model complexity.
- Efficient for many practical classification problems.

---

# 35. Disadvantages of Logistic Regression

- Basic Logistic Regression produces a linear decision boundary.
- It may struggle with strongly non-linear relationships unless features are transformed.
- Performance can be affected by irrelevant or poorly represented features.
- Strongly correlated features can complicate coefficient interpretation.
- Feature engineering may be needed for complex patterns.

---

# 36. Logistic Regression vs Naive Bayes

| Property | Logistic Regression | Naive Bayes |
|---|---|---|
| Main idea | Learn decision function/probability | Estimate class probabilities |
| Classification | Yes | Yes |
| Output probability | Yes | Yes |
| Independence assumption | No Naive Bayes assumption | Conditional independence |
| Decision boundary | Linear in basic form | Determined by likelihoods |
| Training | Optimization | Probability estimation |
| Regularization | Common | Not the same mechanism |

---

# 37. Logistic Regression vs Decision Tree

| Property | Logistic Regression | Decision Tree |
|---|---|---|
| Main idea | Linear probability model | Recursive decisions |
| Decision boundary | Linear | Can be non-linear |
| Scaling | Can help optimization | Usually unnecessary |
| Interpretability | Coefficients | Decision rules |
| Overfitting | Controlled using regularization | Controlled using tree constraints/pruning |
| Probability output | Yes | Can provide class probabilities |

---

# 38. Important Hyperparameters

Common Logistic Regression settings include:

    learning rate
    number of iterations
    regularization strength
    regularization type
    classification threshold

When using a library implementation, the exact parameter names can differ.

For a scratch implementation, the important optimization parameters are:

    learning_rate
    epochs

and the regularization setting can be added when required.

---

# 39. Common Mistakes

### Mistake 1: Calling it a regression prediction

Logistic Regression is mainly used for classification.

### Mistake 2: Treating sigmoid output as the final class

The sigmoid output is a probability.

A threshold is then used to obtain a class.

### Mistake 3: Using the test set during training

The test set should be reserved for final evaluation.

### Mistake 4: Ignoring numerical stability

Extreme values can cause overflow or log-of-zero problems.

### Mistake 5: Assuming 0.5 is always the best threshold

0.5 is common, but the best threshold depends on the application and the costs of false positives and false negatives.

---

# 40. Exam-Oriented Summary

Remember the sequence:

    Linear score
         ↓
    z = β0 + β1x1 + ... + βnxn
         ↓
       Sigmoid
         ↓
    Probability
         ↓
     Threshold
         ↓
      Class

Core sigmoid:

    σ(z) = 1 / (1 + e^(-z))

Core decision boundary:

    β0 + β1x1 + ... + βnxn = 0

Core loss:

    Binary Cross-Entropy

Core optimization idea:

    Gradient Descent

---

# 41. Key Points to Remember

- Logistic Regression is a supervised classification algorithm.
- It is especially useful for binary classification.
- It models the probability of a class.
- The linear score is passed through the sigmoid function.
- Sigmoid outputs a value between 0 and 1.
- A threshold converts probability into a class.
- The standard binary loss is cross-entropy.
- Gradient Descent can be used to train the model from scratch.
- Feature scaling can help Gradient Descent.
- L1 and L2 regularization can help control overfitting.
- The basic Logistic Regression decision boundary is linear.
- Accuracy, precision, recall, and F1-score can be used for evaluation.

The central idea is:

    "Learn a linear score, convert it into a probability with the sigmoid function, and use that probability to classify the observation."
