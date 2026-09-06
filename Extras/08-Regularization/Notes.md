# Regularization

## 1. Introduction

Regularization is a technique used to reduce overfitting by discouraging unnecessarily large or complex model parameters.

A model can fit training data extremely well but perform poorly on unseen data.

Regularization adds a penalty to the training objective so that the model is encouraged to use simpler parameter values.

---

## 2. Why Regularization Is Needed

Consider a model with many features or a highly flexible structure.

Without regularization:

    Model may fit training data very closely
             ↓
        Training error ↓
             ↓
        Complexity ↑
             ↓
        Possible overfitting

Regularization adds a constraint on model parameters.

The goal is better generalization.

---

## 3. Regularized Objective

A general regularized objective can be written as:

    Loss = Data Loss + λ × Penalty

Where:

- Data Loss measures prediction error.
- λ controls regularization strength.
- Penalty discourages large coefficients.

If λ = 0:

    No regularization penalty

If λ becomes larger:

    Stronger penalty

---

# 4. L1 Regularization

L1 regularization adds a penalty based on the absolute values of coefficients.

The objective can be written as:

    Loss = Data Loss + λ Σ|βⱼ|

For Linear Regression with squared error:

    Loss = Σ(yᵢ - ŷᵢ)² + λΣ|βⱼ|

L1 regularization is associated with Lasso Regression.

---

## 5. Effect of L1 Regularization

L1 regularization can drive some coefficients exactly to zero.

Therefore, it can perform a form of feature selection.

Example:

Before regularization:

    β₁ = 2.4
    β₂ = 1.8
    β₃ = 0.2
    β₄ = 3.1

After strong L1 regularization:

    β₁ = 2.0
    β₂ = 1.2
    β₃ = 0
    β₄ = 2.6

The coefficient becoming zero means the corresponding feature does not contribute to the model under that fitted solution.

---

# 6. L2 Regularization

L2 regularization adds a penalty based on squared coefficient values.

The objective can be written as:

    Loss = Data Loss + λ Σβⱼ²

For Linear Regression:

    Loss = Σ(yᵢ - ŷᵢ)² + λΣβⱼ²

L2 regularization is associated with Ridge Regression.

---

## 7. Effect of L2 Regularization

L2 generally shrinks coefficients toward zero rather than forcing many of them exactly to zero.

Example:

Before regularization:

    β₁ = 5.0
    β₂ = 3.0
    β₃ = 1.5

After L2 regularization:

    β₁ = 3.8
    β₂ = 2.2
    β₃ = 1.1

The exact values depend on the dataset and regularization strength.

---

# 8. L1 vs L2

| Property | L1 | L2 |
|---|---|---|
| Penalty | Σ|βⱼ| | Σβⱼ² |
| Common model | Lasso | Ridge |
| Can produce exact zero coefficients | Yes | Usually no |
| Feature selection | Can occur | Usually no |
| Main effect | Sparsity and shrinkage | Smooth shrinkage |

---

# 9. Regularization Strength

The parameter λ controls the strength of regularization.

Small λ:

    Weak penalty
    More model flexibility

Large λ:

    Strong penalty
    Smaller coefficients
    Less flexibility

If regularization is too weak, overfitting may remain.

If regularization is too strong, the model can become too simple and underfit.

---

# 10. Standardization and Regularization

Feature scaling is especially important for regularized linear models.

Suppose one feature ranges from:

    0 to 1

and another ranges from:

    0 to 100000

A coefficient penalty can affect the features differently because their scales are different.

Standardization puts features on comparable scales:

    x' = (x - μ) / σ

The scaler should be fitted using training data only.

---

# 11. Ridge Regression

Ridge Regression uses L2 regularization.

Conceptually:

    Minimize:
    
    prediction_error + λ × coefficient_squared_sum

A common implementation is:

    Ridge(alpha=λ)

Larger alpha generally means stronger regularization.

---

# 12. Lasso Regression

Lasso Regression uses L1 regularization.

Conceptually:

    Minimize:

    prediction_error + λ × coefficient_absolute_sum

A common implementation is:

    Lasso(alpha=λ)

Lasso can produce sparse models by setting some coefficients to zero.

---

# 13. Polynomial Regression and Regularization

Polynomial regression can generate many features.

For example:

    x
    x²
    x³
    x⁴
    ...

A high-degree polynomial can overfit.

Regularization can reduce this problem by penalizing large coefficients.

A useful experiment is:

    Polynomial Regression
            ↓
       No Regularization

compared with:

    Polynomial Regression
            ↓
          Ridge
            ↓
          Lasso

---

# 14. Regularization and Overfitting

Regularization is primarily useful when the model has enough flexibility to overfit.

Without regularization:

    Training error → very low
    Test error → potentially high

With appropriate regularization:

    Training error → may increase slightly
    Test error → may decrease

The goal is not the lowest possible training error.

The goal is better performance on unseen data.

---

# 15. Regularization and Underfitting

Too much regularization can make a model excessively constrained.

Then:

    Model flexibility ↓
    Training error ↑
    Test error ↑

This can produce underfitting.

Therefore, regularization strength must be selected carefully.

---

# 16. Choosing Regularization Strength

The regularization parameter can be selected using validation data or cross-validation.

Example:

    α = 0.001
    α = 0.01
    α = 0.1
    α = 1
    α = 10
    α = 100

Train and evaluate each model.

Choose a value that provides good validation performance.

The final test set should remain untouched until final evaluation.

---

# 17. Regularization in Other Models

Regularization is not limited to Linear Regression.

Examples:

### Logistic Regression

Can use L1 or L2 penalties.

### Neural Networks

Can use:

- L1 regularization
- L2 regularization
- Weight decay
- Dropout
- Early stopping

### Decision Trees

Tree complexity can be controlled through:

- Maximum depth
- Minimum samples per split
- Minimum samples per leaf
- Pruning

---

# 18. Advantages

- Helps reduce overfitting.
- Can improve generalization.
- Controls model complexity.
- L1 can produce sparse models.
- L2 can stabilize coefficient values.
- Useful when many features are present.

---

# 19. Limitations

- Requires choosing regularization strength.
- Too much regularization can cause underfitting.
- L1 and L2 behave differently.
- Feature scaling is important.
- Regularization does not automatically solve every source of overfitting.

---

# 20. Common Mistakes

### Mistake 1: Using the Test Set to Select Alpha

The test set should be reserved for final evaluation.

### Mistake 2: Forgetting Feature Scaling

Regularization penalties depend on coefficient magnitudes, so feature scales matter.

### Mistake 3: Assuming Larger Alpha Is Always Better

Excessive regularization can cause underfitting.

### Mistake 4: Confusing L1 and L2

L1:

    Σ|β|

L2:

    Σβ²

### Mistake 5: Assuming L2 Performs Feature Selection

L2 usually shrinks coefficients without making many of them exactly zero.

---

# 21. Practical Workflow

1. Prepare the dataset.
2. Split into training and test sets.
3. Scale features using training data.
4. Train an unregularized baseline.
5. Train models with different regularization strengths.
6. Compare validation performance.
7. Select the appropriate regularization strength.
8. Evaluate the final selected model on the test set.
9. Inspect coefficient magnitudes.
10. Check for underfitting or overfitting.

---

# 22. Important Formulas

L1:

    Loss = Data Loss + λΣ|βⱼ|

L2:

    Loss = Data Loss + λΣβⱼ²

Standardization:

    x' = (x - μ) / σ

The key idea:

    Regularization
          ↓
    Penalize complexity
          ↓
    Reduce excessive coefficient magnitude
          ↓
    Improve generalization
