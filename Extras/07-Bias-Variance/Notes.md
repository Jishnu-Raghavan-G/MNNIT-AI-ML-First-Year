# Bias and Variance

## 1. Introduction

Bias and Variance are two important concepts used to understand model errors and generalization.

They help explain why a machine learning model may perform poorly on:

- Training data
- Unseen data

A useful mental model is:

    High Bias   → model too simple
    High Variance → model too sensitive to training data

---

## 2. Bias

Bias describes the error caused by overly simplistic assumptions in a model.

A high-bias model may fail to capture important relationships in the training data.

This commonly leads to underfitting.

Example:

A straight line used to model a strongly nonlinear relationship may have high bias.

---

## 3. Variance

Variance describes how sensitive a model is to changes in the training data.

A high-variance model can learn very specific patterns, including noise.

This commonly leads to overfitting.

Example:

A very high-degree polynomial can closely fit training points but perform poorly on new data.

---

## 4. Underfitting

Underfitting occurs when the model is too simple to represent the underlying relationship.

Typical behavior:

    Training error → high
    Test error     → high

The model performs poorly even on the training data.

---

## 5. Overfitting

Overfitting occurs when a model learns the training data too closely, including random fluctuations or noise.

Typical behavior:

    Training error → very low
    Test error     → high

The model does not generalize well to unseen data.

---

## 6. Good Fit

A well-generalized model should perform reasonably well on both training and unseen data.

Typical behavior:

    Training error → low
    Test error     → low

The goal is not simply to minimize training error.

The goal is to achieve good generalization.

---

## 7. Bias-Variance Tradeoff

Increasing model complexity can change the balance between bias and variance.

Very simple model:

    Bias high
    Variance low

Very complex model:

    Bias low
    Variance high

A useful model aims for a suitable balance.

Conceptually:

    Model Complexity
          →
    Bias tends to decrease
          +
    Variance tends to increase

---

## 8. Model Complexity

Examples of increasing complexity include:

- Increasing polynomial degree.
- Adding more features.
- Increasing tree depth.
- Increasing neural-network capacity.

More complexity is not automatically better.

A model should have enough capacity to learn meaningful patterns without memorizing noise.

---

## 9. Diagnosing Underfitting

Signs of underfitting:

- Training performance is poor.
- Test performance is also poor.
- Model is too simple.
- Important relationships are not captured.

Possible solutions:

- Increase model complexity.
- Add useful features.
- Improve feature engineering.
- Reduce excessive regularization.
- Train longer when optimization is incomplete.
- Try a more suitable algorithm.

---

## 10. Diagnosing Overfitting

Signs of overfitting:

- Training performance is very good.
- Test performance is much worse.
- Model is excessively complex.
- Model captures noise.

Possible solutions:

- Simplify the model.
- Use regularization.
- Increase training data.
- Use data augmentation when appropriate.
- Use cross-validation.
- Use early stopping.
- Reduce feature redundancy.
- Prune overly complex models.

---

## 11. Training Error vs Test Error

Suppose different models have the following errors:

| Model | Training Error | Test Error |
|---|---:|---:|
| A | High | High |
| B | Low | Low |
| C | Very Low | High |

Interpretation:

Model A → likely underfitting

Model B → likely good generalization

Model C → likely overfitting

---

## 12. Polynomial Regression Example

Consider a dataset where the true relationship is curved.

A degree-1 polynomial:

    y = β₀ + β₁x

may be too simple.

A moderate-degree polynomial can capture the relationship better.

A very high-degree polynomial may pass extremely closely through training points and begin modeling noise.

Therefore:

    Low complexity → possible underfitting
    Moderate complexity → possible good fit
    High complexity → possible overfitting

---

## 13. Generalization

Generalization is the ability of a model to perform well on data it has not seen during training.

A model that memorizes the training set but performs poorly on new data has poor generalization.

This is why an unseen test set is important.

---

## 14. Validation Set

A validation set can be used to compare model configurations.

For example:

    Degree 1 → validation error
    Degree 2 → validation error
    Degree 3 → validation error
    Degree 4 → validation error

The validation results can help select an appropriate model complexity.

The final test set should be kept separate for final evaluation.

---

## 15. Cross-Validation

Cross-validation repeatedly divides the available training data into training and validation portions.

This gives a more robust estimate of how a model performs across different subsets.

K-fold cross-validation commonly divides the data into K folds.

Each fold is used as validation once while the remaining folds are used for training.

---

## 16. Relationship with Regularization

Regularization discourages unnecessarily complex models.

For example:

### L2 Regularization

Adds a penalty related to squared coefficient magnitudes.

### L1 Regularization

Adds a penalty related to absolute coefficient magnitudes.

Regularization can help reduce overfitting.

---

## 17. Bias and Variance in Common Models

### Simple Model

Usually:

    Higher bias
    Lower variance

### Highly Flexible Model

Usually:

    Lower bias
    Higher variance

These are tendencies rather than absolute rules.

The actual behavior depends on the dataset, model, features and training procedure.

---

## 18. Practical Workflow

When evaluating a model:

1. Train the model.
2. Measure training performance.
3. Measure validation performance if applicable.
4. Compare the gap between training and validation performance.
5. Evaluate the final model on an untouched test set.
6. Determine whether the model appears underfit, well-generalized or overfit.
7. Modify model complexity or regularization if necessary.

---

## 19. Common Mistakes

### Mistake 1: Choosing the Model with the Lowest Training Error

The lowest training error can correspond to overfitting.

### Mistake 2: Using the Test Set for Model Selection

Repeatedly tuning the model based on test performance compromises the purpose of the test set.

### Mistake 3: Assuming More Complexity Is Always Better

More complexity can increase variance and overfitting.

### Mistake 4: Assuming High Training Error Always Means Overfitting

High training error together with high test error is more consistent with underfitting.

---

## 20. Key Takeaway

Remember:

    High Bias
        ↓
    Model too simple
        ↓
    Underfitting

    High Variance
        ↓
    Model too sensitive
        ↓
    Overfitting

The objective is good generalization rather than perfect memorization of the training data.
