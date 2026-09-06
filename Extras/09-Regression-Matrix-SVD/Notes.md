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

\[
\hat{y}=X\beta
\]

where:

- \(X\) = design matrix containing the input features
- \(\beta\) = vector of regression coefficients
- \(\hat{y}\) = predicted target vector

For example:

\[
X=
\begin{bmatrix}
x_{11}&x_{12}\\
x_{21}&x_{22}\\
x_{31}&x_{32}
\end{bmatrix}
\]

and:

\[
\beta=
\begin{bmatrix}
\beta_1\\
\beta_2
\end{bmatrix}
\]

Then:

\[
\hat y=X\beta
\]

If an intercept is required, a column of ones is included in \(X\).

---

# 3. Residuals

The difference between the actual and predicted values is called the residual.

\[
\boxed{e=y-X\beta}
\]

where:

- \(y\) = actual target values
- \(X\beta\) = predicted values
- \(e\) = residual/error vector

A good regression model attempts to make these residuals small.

---

# 4. Residual Sum of Squares (RSS)

The lecture defines:

\[
\boxed{RSS=e^Te}
\]

Since:

\[
e=y-X\beta
\]

we get:

\[
\boxed{
RSS=(y-X\beta)^T(y-X\beta)
}
\]

### Meaning

RSS measures the total squared error made by the regression model.

A smaller RSS means the predictions are closer to the observed values in terms of squared error.

---

# 5. Expanding RSS

Starting with:

\[
RSS=(y-X\beta)^T(y-X\beta)
\]

Expanding:

\[
RSS=
y^Ty-y^TX\beta-\beta^TX^Ty+\beta^TX^TX\beta
\]

Because \(y^TX\beta\) is a scalar:

\[
y^TX\beta=\beta^TX^Ty
\]

Therefore:

\[
\boxed{
RSS=y^Ty-2\beta^TX^Ty+\beta^TX^TX\beta
}
\]

This expression is minimized to obtain the least-squares regression coefficients.

---

# 6. Least Squares

The objective of ordinary least squares is to find \(\beta\) that minimizes:

\[
RSS=(y-X\beta)^T(y-X\beta)
\]

In simple words:

> Choose the regression coefficients that make the total squared prediction error as small as possible.

---

# 7. Normal Equation

Minimizing RSS with respect to \(\beta\) produces:

\[
X^TX\beta=X^Ty
\]

This is called the **normal equation**.

If \(X^TX\) is invertible:

\[
\boxed{
\beta=(X^TX)^{-1}X^Ty
}
\]

### Meaning

The formula directly calculates the regression coefficients without iterative gradient descent.

---

# 8. Understanding the Matrices

Suppose:

\[
X\in\mathbb{R}^{n\times p}
\]

where:

- \(n\) = number of observations
- \(p\) = number of features

Then:

\[
y\in\mathbb{R}^{n\times1}
\]

and:

\[
\beta\in\mathbb{R}^{p\times1}
\]

Therefore:

\[
X\beta
\]

has dimensions:

\[
(n\times p)(p\times1)=n\times1
\]

which matches \(y\).

---

# 9. Why the Normal Equation Can Fail

The normal equation requires:

\[
(X^TX)^{-1}
\]

But an inverse does not exist for every matrix.

A matrix is singular when its determinant is zero:

\[
\boxed{\det(X^TX)=0}
\]

In such a case, the ordinary inverse cannot be calculated.

---

# 10. Example of a Singular Matrix

Consider:

\[
X=
\begin{bmatrix}
1&2\\
2&4\\
3&6
\end{bmatrix}
\]

The second column is exactly twice the first column.

Therefore the features are linearly dependent.

Calculate:

\[
X^TX=
\begin{bmatrix}
14&28\\
28&56
\end{bmatrix}
\]

Its determinant is:

\[
14(56)-28(28)=0
\]

Therefore:

\[
\boxed{\det(X^TX)=0}
\]

and:

\[
(X^TX)^{-1}
\]

does not exist.

---

# 11. Pseudo-Inverse

To deal with matrices where the ordinary inverse cannot be calculated, we can use the **Moore-Penrose pseudo-inverse**.

The regression solution becomes:

\[
\boxed{
\beta=X^+y
}
\]

where:

\[
X^+
\]

is the pseudo-inverse of \(X\).

### Meaning

The pseudo-inverse provides a generalized inverse and allows us to obtain a least-squares solution even when the usual inverse is unavailable.

---

# 12. Singular Value Decomposition (SVD)

SVD decomposes a matrix into three matrices:

\[
\boxed{
X=U\Sigma V^T
}
\]

where:

- \(U\) = matrix of left singular vectors
- \(\Sigma\) = diagonal matrix containing singular values
- \(V\) = matrix of right singular vectors
- \(V^T\) = transpose of \(V\)

SVD is particularly useful for understanding matrix structure and obtaining a stable pseudo-inverse solution.

---

# 13. Singular Values

The singular values of \(X\) are related to the eigenvalues of \(X^TX\):

\[
\boxed{
\sigma_i=\sqrt{\lambda_i}
}
\]

where:

- \(\sigma_i\) = singular value
- \(\lambda_i\) = corresponding eigenvalue of \(X^TX\)

Large singular values represent strong directions in the data.

Very small singular values indicate directions that may be poorly determined or nearly dependent.

---

# 14. Pseudo-Inverse Using SVD

Given:

\[
X=U\Sigma V^T
\]

the pseudo-inverse is:

\[
\boxed{
X^+=V\Sigma^+U^T
}
\]

Therefore:

\[
\beta=X^+y
\]

becomes:

\[
\boxed{
\beta=V\Sigma^+U^Ty
}
\]

This is the SVD-based regression solution.

---

# 15. Meaning of Each SVD Step

The expression:

\[
\beta=V\Sigma^+U^Ty
\]

can be understood in three stages.

### Step 1: Projection

\[
\boxed{U^Ty}
\]

projects the target vector \(y\) onto the directions represented by \(U\).

### Step 2: Scaling

\[
\boxed{\Sigma^+U^Ty}
\]

scales those projected values using the reciprocal singular values.

### Step 3: Reconstruction

\[
\boxed{
V(\Sigma^+U^Ty)
}
\]

maps the result back into the regression parameter space.

Therefore:

\[
\boxed{
\beta=V(\Sigma^+U^Ty)
}
\]

---

# 16. Constructing \(\Sigma^+\)

Suppose:

\[
\Sigma=
\begin{bmatrix}
\sigma_1&0\\
0&\sigma_2
\end{bmatrix}
\]

Then:

\[
\Sigma^+=
\begin{bmatrix}
1/\sigma_1&0\\
0&1/\sigma_2
\end{bmatrix}
\]

for non-zero singular values.

If a singular value is zero, its reciprocal is not used; the corresponding pseudo-inverse value is set to zero.

Thus:

\[
\boxed{
\sigma_i^+=
\begin{cases}
1/\sigma_i,&\sigma_i\neq0\\
0,&\sigma_i=0
\end{cases}
}
\]

---

# 17. Lecture SVD Example

The lecture gives:

\[
X=
\begin{bmatrix}
1&2\\
1&4\\
1&6\\
1&8
\end{bmatrix}
\]

and:

\[
y=
\begin{bmatrix}
3\\
7\\
5\\
10
\end{bmatrix}
\]

First:

\[
X^TX=
\begin{bmatrix}
4&20\\
20&120
\end{bmatrix}
\]

The eigenvalues are approximately:

\[
\lambda_1\approx123.36
\]

\[
\lambda_2\approx0.64
\]

Therefore the singular values are approximately:

\[
\sigma_1\approx11.11
\]

\[
\sigma_2\approx0.80
\]

The lecture gives approximately:

\[
V=
\begin{bmatrix}
0.160&-0.987\\
0.987&0.160
\end{bmatrix}
\]

and corresponding \(U\) vectors.

The pseudo-inverse diagonal matrix is approximately:

\[
\Sigma^+=
\begin{bmatrix}
0.090&0\\
0&1.25
\end{bmatrix}
\]

Then:

\[
U^Ty\approx
\begin{bmatrix}
13.16\\
1.21
\end{bmatrix}
\]

and:

\[
\Sigma^+U^Ty\approx
\begin{bmatrix}
1.184\\
1.512
\end{bmatrix}
\]

Finally:

\[
\beta=V\Sigma^+U^Ty
\]

giving the lecture's final result:

\[
\boxed{
\beta\approx
\begin{bmatrix}
1.5\\
0.95
\end{bmatrix}
}
\]

---

# 18. Normal Equation vs Pseudo-Inverse vs SVD

## Normal Equation

\[
\boxed{
\beta=(X^TX)^{-1}X^Ty
}
\]

Use when \(X^TX\) is invertible.

## Pseudo-Inverse

\[
\boxed{
\beta=X^+y
}
\]

Use when an ordinary inverse is unavailable or a generalized least-squares solution is desired.

## SVD

\[
\boxed{
X=U\Sigma V^T
}
\]

and:

\[
\boxed{
\beta=V\Sigma^+U^Ty
}
\]

SVD provides a direct way to construct the pseudo-inverse.

---

# 19. Regression Error Metrics

After finding regression coefficients, we need to measure how well the model predicts.

Let:

\[
e_i=y_i-\hat y_i
\]

be the prediction error for observation \(i\).

Different metrics measure error differently.

---

# 20. Absolute Error

For one observation:

\[
\boxed{
AE_i=|y_i-\hat y_i|
}
\]

### Meaning

Absolute error tells us how far the prediction is from the actual value without considering the direction of the error.

For example:

\[
y=100,\quad\hat y=90
\]

gives:

\[
AE=|100-90|=10
\]

---

# 21. Sum of Absolute Errors (SAE)

\[
\boxed{
SAE=\sum_{i=1}^{n}|y_i-\hat y_i|
}
\]

### Meaning

SAE adds all absolute prediction errors.

It measures the total magnitude of errors without allowing positive and negative errors to cancel each other.

---

# 22. Mean Absolute Error (MAE)

\[
\boxed{
MAE=\frac{1}{n}\sum_{i=1}^{n}|y_i-\hat y_i|
}
\]

### Meaning

MAE is the average absolute prediction error.

If:

\[
MAE=5
\]

then the model's predictions are, on average, about 5 target units away from the actual values in absolute terms.

### Important property

MAE treats errors linearly.

An error of 10 is twice as large as an error of 5.

It does not square the error, so it is less strongly affected by extreme errors than MSE.

---

# 23. Squared Error

For one observation:

\[
\boxed{
SE_i=(y_i-\hat y_i)^2
}
\]

### Meaning

The error is squared so that:

1. Positive and negative errors do not cancel.
2. Large errors receive much greater weight.

For example:

\[
e=5
\]

gives:

\[
e^2=25
\]

while:

\[
e=10
\]

gives:

\[
e^2=100
\]

The second error is twice as large, but its squared error is four times as large.

---

# 24. Sum of Squared Errors (SSE)

\[
\boxed{
SSE=\sum_{i=1}^{n}(y_i-\hat y_i)^2
}
\]

### Meaning

SSE is the total squared prediction error.

It strongly emphasizes large prediction errors.

---

# 25. Mean Squared Error (MSE)

\[
\boxed{
MSE=\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat y_i)^2
}
\]

### Meaning

MSE is the average squared prediction error.

Because errors are squared, large errors have a disproportionately large effect.

---

# 26. Root Mean Squared Error (RMSE)

\[
\boxed{
RMSE=\sqrt{MSE}
}
\]

### Meaning

RMSE is the square root of the average squared error.

Unlike MSE, RMSE is expressed in the same units as the target variable.

---

# 27. MAE vs MSE

The main difference is how they treat large errors.

### MAE

\[
MAE=\frac{1}{n}\sum|e_i|
\]

Error contribution grows linearly.

### MSE

\[
MSE=\frac{1}{n}\sum e_i^2
\]

Error contribution grows quadratically.

Therefore:

- MAE is more robust to large outliers.
- MSE heavily penalizes large errors.
- RMSE also strongly reflects large errors because it is based on squared errors.

---

# 28. Lecture Example: Why Large Errors Matter

Consider the errors:

\[
-10,-10,-10,-10,100
\]

Absolute errors are:

\[
10,10,10,10,100
\]

Therefore:

\[
SAE=10+10+10+10+100
\]

\[
\boxed{SAE=140}
\]

Now square the errors:

\[
100,100,100,100,10000
\]

Therefore:

\[
\boxed{SSE=10400}
\]

The single error of \(100\) contributes:

\[
10000
\]

to SSE.

This demonstrates how strongly squared-error metrics emphasize very large errors.

---

# 29. MAE vs SSE/MSE: Intuition

Suppose two predictions have errors:

\[
e_1=2
\]

and:

\[
e_2=20
\]

For MAE:

\[
2,\quad20
\]

The second error is 10 times larger.

For squared error:

\[
2^2=4
\]

\[
20^2=400
\]

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

\[
\boxed{\text{MAE is less sensitive to extreme errors than squared-error metrics.}}
\]

---

# 31. Relationship Between Metrics

For errors:

\[
e_i=y_i-\hat y_i
\]

we have:

\[
SAE=\sum|e_i|
\]

\[
MAE=\frac{SAE}{n}
\]

\[
SSE=\sum e_i^2
\]

\[
MSE=\frac{SSE}{n}
\]

\[
RMSE=\sqrt{\frac{SSE}{n}}
\]

So:

\[
\boxed{MAE=\frac{SAE}{n}}
\]

and:

\[
\boxed{RMSE=\sqrt{\frac{SSE}{n}}}
\]

---

# 32. Complete Matrix Regression Workflow

A complete matrix-based regression workflow is:

### Step 1: Prepare \(X\) and \(y\)

\[
X=\text{feature matrix}
\]

\[
y=\text{target vector}
\]

### Step 2: Construct the model

\[
\hat y=X\beta
\]

### Step 3: Calculate residuals

\[
e=y-X\beta
\]

### Step 4: Define RSS

\[
RSS=e^Te
\]

### Step 5: Solve for coefficients

Using the normal equation:

\[
\beta=(X^TX)^{-1}X^Ty
\]

or using the pseudo-inverse:

\[
\beta=X^+y
\]

or using SVD:

\[
\beta=V\Sigma^+U^Ty
\]

### Step 6: Predict

\[
\hat y=X\beta
\]

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

\[
X^TX\beta=X^Ty
\]

not:

\[
XX\beta=Xy
\]

---

## Mistake 2: Assuming \(X^TX\) is always invertible

It may be singular.

Always remember:

\[
\det(X^TX)=0
\]

means the ordinary inverse does not exist.

---

## Mistake 3: Confusing eigenvalues and singular values

For \(X^TX\):

\[
\boxed{\sigma_i=\sqrt{\lambda_i}}
\]

---

## Mistake 4: Forgetting the transpose in SVD

Correct:

\[
\boxed{X=U\Sigma V^T}
\]

---

## Mistake 5: Using the wrong pseudo-inverse expression

Correct:

\[
\boxed{X^+=V\Sigma^+U^T}
\]

---

## Mistake 6: Confusing MAE and MSE

MAE:

\[
\frac{1}{n}\sum|e_i|
\]

MSE:

\[
\frac{1}{n}\sum e_i^2
\]

MAE is linear in error magnitude.

MSE is quadratic.

---

# 34. Exam-Oriented Formula Sheet

### Prediction

\[
\boxed{\hat y=X\beta}
\]

### Residual

\[
\boxed{e=y-X\beta}
\]

### RSS

\[
\boxed{RSS=e^Te}
\]

\[
\boxed{
RSS=(y-X\beta)^T(y-X\beta)
}
\]

### Normal Equation

\[
\boxed{
X^TX\beta=X^Ty
}
\]

### Closed-form solution

\[
\boxed{
\beta=(X^TX)^{-1}X^Ty
}
\]

### Pseudo-inverse

\[
\boxed{
\beta=X^+y
}
\]

### SVD

\[
\boxed{
X=U\Sigma V^T
}
\]

### SVD pseudo-inverse

\[
\boxed{
X^+=V\Sigma^+U^T
}
\]

### SVD regression

\[
\boxed{
\beta=V\Sigma^+U^Ty
}
\]

### Singular values

\[
\boxed{
\sigma_i=\sqrt{\lambda_i}
}
\]

### Absolute Error

\[
\boxed{
AE_i=|y_i-\hat y_i|
}
\]

### SAE

\[
\boxed{
SAE=\sum|y_i-\hat y_i|
}
\]

### MAE

\[
\boxed{
MAE=\frac{1}{n}\sum|y_i-\hat y_i|
}
\]

### SSE

\[
\boxed{
SSE=\sum(y_i-\hat y_i)^2
}
\]

### MSE

\[
\boxed{
MSE=\frac{1}{n}\sum(y_i-\hat y_i)^2
}
\]

### RMSE

\[
\boxed{
RMSE=\sqrt{MSE}
}
\]

---

# 35. Final Concept Summary

Remember the central chain:

\[
X,\ y
\]

↓

\[
\hat y=X\beta
\]

↓

\[
e=y-X\beta
\]

↓

\[
RSS=e^Te
\]

↓

Minimize RSS

↓

\[
X^TX\beta=X^Ty
\]

↓

If invertible:

\[
\beta=(X^TX)^{-1}X^Ty
\]

If not:

\[
\beta=X^+y
\]

Using SVD:

\[
X=U\Sigma V^T
\]

and:

\[
X^+=V\Sigma^+U^T
\]

therefore:

\[
\boxed{
\beta=V\Sigma^+U^Ty
}
\]

Finally evaluate predictions using:

\[
MAE,\ SSE,\ MSE,\ RMSE
\]

The most important distinction to remember is:

\[
\boxed{\text{MAE penalizes errors linearly}}
\]

while:

\[
\boxed{\text{MSE/SSE penalize errors quadratically}}
\]

Therefore, large errors have a much stronger effect on SSE and MSE.
