# Regression: Matrix Approach and SVD

## 1. Overview

Linear Regression can be solved mathematically using matrices instead of updating parameters one step at a time.

The main ideas in this topic are:

- Matrix representation of Linear Regression
- Residuals and least squares
- Residual Sum of Squares (RSS)
- Normal Equation
- Singular matrices
- Moore-Penrose Pseudo-Inverse
- Singular Value Decomposition (SVD)
- SVD-based regression
- MAE, SAE, SSE, MSE, and RMSE
- Why MAE and squared-error metrics behave differently

The lecture specifically develops the matrix and SVD approach to regression.

---

## 2. Linear Regression in Matrix Form

A linear regression model can be written as:

**y_hat = X beta**

where:

- `X` = design matrix containing the input features
- `beta` = vector of regression coefficients
- `y_hat` = predicted target vector

For example:

```text
X =
[ x11  x12 ]
[ x21  x22 ]
[ x31  x32 ]
```

and:

```text
beta =
[ beta1 ]
[ beta2 ]
```

Then:

```text
y_hat = X beta
```

If an intercept is required, a column of ones is included in `X`.

---

# 3. Residuals

The difference between the actual and predicted values is called the residual.

```text
e = y - X beta
```

where:

- `y` = actual target values
- `X beta` = predicted values
- `e` = residual/error vector

A good regression model attempts to make these residuals small.

---

# 4. Residual Sum of Squares (RSS)

The lecture defines:

```text
RSS = e^T e
```

Since:

```text
e = y - X beta
```

we get:

```text
RSS = (y - X beta)^T (y - X beta)
```

### Meaning

RSS measures the total squared error made by the regression model.

A smaller RSS means the predictions are closer to the observed values in terms of squared error.

---

# 5. Expanding RSS

Starting with:

```text
RSS = (y - X beta)^T (y - X beta)
```

Expanding:

```text
RSS =
y^T y
- y^T X beta
- beta^T X^T y
+ beta^T X^T X beta
```

Because `y^T X beta` is a scalar:

```text
y^T X beta = beta^T X^T y
```

Therefore:

```text
RSS =
y^T y
- 2 beta^T X^T y
+ beta^T X^T X beta
```

This expression is minimized to obtain the least-squares regression coefficients.

---

# 6. Least Squares

The objective of ordinary least squares is to find `beta` that minimizes:

```text
RSS = (y - X beta)^T (y - X beta)
```

In simple words:

> Choose the regression coefficients that make the total squared prediction error as small as possible.

---

# 7. Normal Equation

Minimizing RSS with respect to `beta` produces:

```text
X^T X beta = X^T y
```

This is called the **normal equation**.

If `X^T X` is invertible:

```text
beta = (X^T X)^(-1) X^T y
```

### Meaning

The formula directly calculates the regression coefficients without iterative gradient descent.

---

# 8. Understanding the Matrices

Suppose:

```text
X ∈ R^(n × p)
```

where:

- `n` = number of observations
- `p` = number of features

Then:

```text
y ∈ R^(n × 1)
```

and:

```text
beta ∈ R^(p × 1)
```

Therefore:

```text
X beta
```

has dimensions:

```text
(n × p)(p × 1) = n × 1
```

which matches `y`.

---

# 9. Why the Normal Equation Can Fail

The normal equation requires:

```text
(X^T X)^(-1)
```

But an inverse does not exist for every matrix.

A matrix is singular when its determinant is zero:

```text
det(X^T X) = 0
```

In such a case, the ordinary inverse cannot be calculated.

---

# 10. Example of a Singular Matrix

Consider:

```text
X =
[ 1  2 ]
[ 2  4 ]
[ 3  6 ]
```

The second column is exactly twice the first column.

Therefore the features are linearly dependent.

Calculate:

```text
X^T X =
[ 14  28 ]
[ 28  56 ]
```

Its determinant is:

```text
14(56) - 28(28) = 0
```

Therefore:

```text
det(X^T X) = 0
```

and:

```text
(X^T X)^(-1)
```

does not exist.

---

# 11. Pseudo-Inverse

To deal with matrices where the ordinary inverse cannot be calculated, we can use the **Moore-Penrose pseudo-inverse**.

The regression solution becomes:

```text
beta = X^+ y
```

where:

```text
X^+
```

is the pseudo-inverse of `X`.

### Meaning

The pseudo-inverse provides a generalized inverse and allows us to obtain a least-squares solution even when the usual inverse is unavailable.

---

# 12. Singular Value Decomposition (SVD)

SVD decomposes a matrix into three matrices:

```text
X = U Sigma V^T
```

where:

- `U` = matrix of left singular vectors
- `Sigma` = diagonal matrix containing singular values
- `V` = matrix of right singular vectors
- `V^T` = transpose of `V`

SVD is particularly useful for understanding matrix structure and obtaining a stable pseudo-inverse solution.

---

# 13. Singular Values

The singular values of `X` are related to the eigenvalues of `X^T X`:

```text
sigma_i = sqrt(lambda_i)
```

where:

- `sigma_i` = singular value
- `lambda_i` = corresponding eigenvalue of `X^T X`

Large singular values represent strong directions in the data.

Very small singular values indicate directions that may be poorly determined or nearly dependent.

---

# 14. Pseudo-Inverse Using SVD

Given:

```text
X = U Sigma V^T
```

the pseudo-inverse is:

```text
X^+ = V Sigma^+ U^T
```

Therefore:

```text
beta = X^+ y
```

becomes:

```text
beta = V Sigma^+ U^T y
```

This is the SVD-based regression solution.

---

# 15. Meaning of Each SVD Step

The expression:

```text
beta = V Sigma^+ U^T y
```

can be understood in three stages.

### Step 1: Projection

```text
U^T y
```

projects the target vector `y` onto the directions represented by `U`.

### Step 2: Scaling

```text
Sigma^+ U^T y
```

scales those projected values using the reciprocal singular values.

### Step 3: Reconstruction

```text
V (Sigma^+ U^T y)
```

maps the result back into the regression parameter space.

Therefore:

```text
beta = V (Sigma^+ U^T y)
```

---

# 16. Constructing Sigma^+

Suppose:

```text
Sigma =
[ sigma1   0    ]
[ 0      sigma2 ]
```

Then:

```text
Sigma^+ =
[ 1/sigma1    0       ]
[ 0         1/sigma2  ]
```

for non-zero singular values.

If a singular value is zero, its reciprocal is not used; the corresponding pseudo-inverse value is set to zero.

Thus:

```text
sigma_i^+ =
    1 / sigma_i,   if sigma_i != 0
    0,              if sigma_i = 0
```

---

# 17. Lecture SVD Example

The lecture gives:

```text
X =
[ 1  2 ]
[ 1  4 ]
[ 1  6 ]
[ 1  8 ]
```

and:

```text
y =
[ 3 ]
[ 7 ]
[ 5 ]
[ 10 ]
```

First:

```text
X^T X =
[ 4   20 ]
[ 20  120 ]
```

The eigenvalues are approximately:

```text
lambda1 ≈ 123.36
lambda2 ≈ 0.64
```

Therefore the singular values are approximately:

```text
sigma1 ≈ 11.11
sigma2 ≈ 0.80
```

The lecture gives approximately:

```text
V =
[ 0.160  -0.987 ]
[ 0.987   0.160 ]
```

and corresponding `U` vectors.

The pseudo-inverse diagonal matrix is approximately:

```text
Sigma^+ =
[ 0.090   0    ]
[ 0       1.25 ]
```

Then:

```text
U^T y ≈
[ 13.16 ]
[  1.21 ]
```

and:

```text
Sigma^+ U^T y ≈
[ 1.184 ]
[ 1.512 ]
```

Finally:

```text
beta = V Sigma^+ U^T y
```

giving the lecture's final result:

```text
beta ≈
[ 1.5  ]
[ 0.95 ]
```

---

# 18. Normal Equation vs Pseudo-Inverse vs SVD

## Normal Equation

```text
beta = (X^T X)^(-1) X^T y
```

Use when `X^T X` is invertible.

## Pseudo-Inverse

```text
beta = X^+ y
```

Use when an ordinary inverse is unavailable or a generalized least-squares solution is desired.

## SVD

```text
X = U Sigma V^T
```

and:

```text
beta = V Sigma^+ U^T y
```

SVD provides a direct way to construct the pseudo-inverse.

---

# 19. Regression Error Metrics

After finding regression coefficients, we need to measure how well the model predicts.

Let:

```text
e_i = y_i - y_hat_i
```

be the prediction error for observation `i`.

Different metrics measure error differently.

---

# 20. Absolute Error

For one observation:

```text
AE_i = |y_i - y_hat_i|
```

### Meaning

Absolute error tells us how far the prediction is from the actual value without considering the direction of the error.

For example:

```text
y = 100
y_hat = 90
```

gives:

```text
AE = |100 - 90| = 10
```

---

# 21. Sum of Absolute Errors (SAE)

```text
SAE = sum |y_i - y_hat_i|
```

### Meaning

SAE adds all absolute prediction errors.

It measures the total magnitude of errors without allowing positive and negative errors to cancel each other.

---

# 22. Mean Absolute Error (MAE)

```text
MAE = (1/n) sum |y_i - y_hat_i|
```

### Meaning

MAE is the average absolute prediction error.

If:

```text
MAE = 5
```

then the model's predictions are, on average, about 5 target units away from the actual values in absolute terms.

### Important property

MAE treats errors linearly.

An error of 10 is twice as large as an error of 5.

It does not square the error, so it is less strongly affected by extreme errors than MSE.

---

# 23. Squared Error

For one observation:

```text
SE_i = (y_i - y_hat_i)^2
```

### Meaning

The error is squared so that:

1. Positive and negative errors do not cancel.
2. Large errors receive much greater weight.

For example:

```text
e = 5
```

gives:

```text
e^2 = 25
```

while:

```text
e = 10
```

gives:

```text
e^2 = 100
```

The second error is twice as large, but its squared error is four times as large.

---

# 24. Sum of Squared Errors (SSE)

```text
SSE = sum (y_i - y_hat_i)^2
```

### Meaning

SSE is the total squared prediction error.

It strongly emphasizes large prediction errors.

---

# 25. Mean Squared Error (MSE)

```text
MSE = (1/n) sum (y_i - y_hat_i)^2
```

### Meaning

MSE is the average squared prediction error.

Because errors are squared, large errors have a disproportionately large effect.

---

# 26. Root Mean Squared Error (RMSE)

```text
RMSE = sqrt(MSE)
```

### Meaning

RMSE is the square root of the average squared error.

Unlike MSE, RMSE is expressed in the same units as the target variable.

---

# 27. MAE vs MSE

The main difference is how they treat large errors.

### MAE

```text
MAE = (1/n) sum |e_i|
```

Error contribution grows linearly.

### MSE

```text
MSE = (1/n) sum e_i^2
```

Error contribution grows quadratically.

Therefore:

- MAE is more robust to large outliers.
- MSE heavily penalizes large errors.
- RMSE also strongly reflects large errors because it is based on squared errors.

---

# 28. Lecture Example: Why Large Errors Matter

Consider the errors:

```text
-10, -10, -10, -10, 100
```

Absolute errors are:

```text
10, 10, 10, 10, 100
```

Therefore:

```text
SAE = 10 + 10 + 10 + 10 + 100
```

```text
SAE = 140
```

Now square the errors:

```text
100, 100, 100, 100, 10000
```

Therefore:

```text
SSE = 10400
```

The single error of `100` contributes:

```text
10000
```

to SSE.

This demonstrates how strongly squared-error metrics emphasize very large errors.

---

# 29. MAE vs SSE/MSE: Intuition

Suppose two predictions have errors:

```text
e1 = 2
```

and:

```text
e2 = 20
```

For MAE:

```text
2, 20
```

The second error is 10 times larger.

For squared error:

```text
2^2 = 4
```

```text
20^2 = 400
```

The second error is 100 times larger.

Therefore, MSE/SSE punish large errors much more aggressively.

---

# 30. When MAE Can Be Preferred

The lecture discusses situations where MAE is preferred because it is more robust to outliers.

Examples include:

- Robust versions of Linear Regression.
- Quantile Regression.
- L1-regularized models.
- Some forecasting applications involving extreme fluctuations.

The key idea is:

```text
MAE is less sensitive to extreme errors than squared-error metrics.
```

---

# 31. Relationship Between Metrics

For errors:

```text
e_i = y_i - y_hat_i
```

we have:

```text
SAE = sum |e_i|
```

```text
MAE = SAE / n
```

```text
SSE = sum e_i^2
```

```text
MSE = SSE / n
```

```text
RMSE = sqrt(SSE / n)
```

So:

```text
MAE = SAE / n
```

and:

```text
RMSE = sqrt(SSE / n)
```

---

# 32. Complete Matrix Regression Workflow

A complete matrix-based regression workflow is:

### Step 1: Prepare X and y

```text
X = feature matrix
```

```text
y = target vector
```

### Step 2: Construct the model

```text
y_hat = X beta
```

### Step 3: Calculate residuals

```text
e = y - X beta
```

### Step 4: Define RSS

```text
RSS = e^T e
```

### Step 5: Solve for coefficients

Using the normal equation:

```text
beta = (X^T X)^(-1) X^T y
```

or using the pseudo-inverse:

```text
beta = X^+ y
```

or using SVD:

```text
beta = V Sigma^+ U^T y
```

### Step 6: Predict

```text
y_hat = X beta
```

### Step 7: Evaluate

Calculate:

- MAE
- SSE
- MSE
- RMSE

---

# 33. Common Mistakes

## Mistake 1: Forgetting the transpose

The normal equation is:

```text
X^T X beta = X^T y
```

not:

```text
X X beta = X y
```

---

## Mistake 2: Assuming X^T X is always invertible

It may be singular.

Always remember:

```text
det(X^T X) = 0
```

means the ordinary inverse does not exist.

---

## Mistake 3: Confusing eigenvalues and singular values

For `X^T X`:

```text
sigma_i = sqrt(lambda_i)
```

---

## Mistake 4: Forgetting the transpose in SVD

Correct:

```text
X = U Sigma V^T
```

---

## Mistake 5: Using the wrong pseudo-inverse expression

Correct:

```text
X^+ = V Sigma^+ U^T
```

---

## Mistake 6: Confusing MAE and MSE

MAE:

```text
(1/n) sum |e_i|
```

MSE:

```text
(1/n) sum e_i^2
```

MAE is linear in error magnitude.

MSE is quadratic.

---

# 34. Exam-Oriented Formula Sheet

### Prediction

```text
y_hat = X beta
```

### Residual

```text
e = y - X beta
```

### RSS

```text
RSS = e^T e
```

```text
RSS = (y - X beta)^T (y - X beta)
```

### Normal Equation

```text
X^T X beta = X^T y
```

### Closed-form solution

```text
beta = (X^T X)^(-1) X^T y
```

### Pseudo-inverse

```text
beta = X^+ y
```

### SVD

```text
X = U Sigma V^T
```

### SVD pseudo-inverse

```text
X^+ = V Sigma^+ U^T
```

### SVD regression

```text
beta = V Sigma^+ U^T y
```

### Singular values

```text
sigma_i = sqrt(lambda_i)
```

### Absolute Error

```text
AE_i = |y_i - y_hat_i|
```

### SAE

```text
SAE = sum |y_i - y_hat_i|
```

### MAE

```text
MAE = (1/n) sum |y_i - y_hat_i|
```

### SSE

```text
SSE = sum (y_i - y_hat_i)^2
```

### MSE

```text
MSE = (1/n) sum (y_i - y_hat_i)^2
```

### RMSE

```text
RMSE = sqrt(MSE)
```

---

# 35. Final Concept Summary

Remember the central chain:

```text
X, y
  ↓
y_hat = X beta
  ↓
e = y - X beta
  ↓
RSS = e^T e
  ↓
Minimize RSS
  ↓
X^T X beta = X^T y
```

If invertible:

```text
beta = (X^T X)^(-1) X^T y
```

If not:

```text
beta = X^+ y
```

Using SVD:

```text
X = U Sigma V^T
```

and:

```text
X^+ = V Sigma^+ U^T
```

therefore:

```text
beta = V Sigma^+ U^T y
```

Finally evaluate predictions using:

```text
MAE, SSE, MSE, RMSE
```

The most important distinction to remember is:

```text
MAE penalizes errors linearly
```

while:

```text
MSE/SSE penalize errors quadratically
```

Therefore, large errors have a much stronger effect on SSE and MSE.
