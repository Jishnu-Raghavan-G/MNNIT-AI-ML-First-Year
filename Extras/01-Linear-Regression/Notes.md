# Linear Regression

## 1. Introduction

Linear Regression is a supervised machine learning algorithm used to predict a continuous numerical target.

Examples:

- Predict student score from study hours.
- Predict house price from area.
- Predict sales from advertising expenditure.
- Predict salary from years of experience.

The basic idea is to find a mathematical relationship between input features and a continuous output.

---

## 2. Simple Linear Regression

Simple Linear Regression uses one input feature.

The model is:

y = β₀ + β₁x

Where:

- y = predicted output
- x = input feature
- β₀ = intercept
- β₁ = slope/coefficient

### Interpretation

β₀ represents the predicted value of y when x = 0.

β₁ represents how much y changes when x increases by one unit.

For example:

y = 40 + 5x

If x = 4:

y = 40 + 5(4)
y = 60

The model predicts 60.

---

## 3. Multiple Linear Regression

When there are multiple features:

y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ

For example:

score = β₀ + β₁(hours) + β₂(attendance) + β₃(assignments)

Each coefficient represents the contribution of its corresponding feature while the other features are kept fixed.

---

## 4. Features and Target

In a regression problem:

X = input features

y = target/output

Example:

| Hours | Attendance | Assignments | Score |
|---:|---:|---:|---:|
| 2 | 70 | 5 | 55 |
| 4 | 80 | 7 | 68 |
| 6 | 90 | 9 | 82 |

Features:

X = Hours, Attendance, Assignments

Target:

y = Score

---

## 5. How Linear Regression Learns

The algorithm tries to find coefficients that make predictions as close as possible to the actual values.

For each training example:

prediction = model(X)

error = actual - prediction

The coefficients are adjusted so that the overall error becomes small.

A commonly used objective is the Sum of Squared Errors:

SSE = Σ(yᵢ - ŷᵢ)²

where:

- yᵢ = actual value
- ŷᵢ = predicted value

Squaring the errors makes large errors more important.

---

## 6. Least Squares

The Least Squares method chooses coefficients that minimize:

SSE = Σ(yᵢ - ŷᵢ)²

For simple linear regression:

SSE = Σ(yᵢ - (β₀ + β₁xᵢ))²

The best-fitting line is the line that produces the minimum squared error.

---

## 7. Matrix Representation

Multiple Linear Regression can be written using matrices.

The model becomes:

Y = Xβ

Where:

- Y = target vector
- X = feature matrix
- β = coefficient vector

The intercept is represented by adding a column of ones to X.

Example:

X =

[1  x₁]
[1  x₂]
[1  x₃]

β =

[β₀]
[β₁]

Y =

[y₁]
[y₂]
[y₃]

Then:

Xβ = Y

---

## 8. Normal Equation

The Least Squares solution can be written as:

β = (XᵀX)⁻¹XᵀY

This is called the Normal Equation.

It directly calculates the coefficients without iterative gradient descent.

However, calculating the inverse can become computationally expensive or numerically unstable for some datasets.

---

## 9. Pseudoinverse

Instead of explicitly calculating:

(XᵀX)⁻¹Xᵀ

we can use the Moore-Penrose pseudoinverse.

The solution becomes:

β = X⁺Y

where X⁺ is the pseudoinverse of X.

In NumPy:

np.linalg.pinv(X)

Then:

beta = np.linalg.pinv(X) @ y

The pseudoinverse is useful when XᵀX is singular or close to singular.

---

## 10. SVD

Singular Value Decomposition decomposes a matrix as:

X = UΣVᵀ

where:

- U = left singular vectors
- Σ = singular values
- Vᵀ = transpose of right singular vectors

SVD can be used to calculate the pseudoinverse.

For:

X = UΣVᵀ

the pseudoinverse is:

X⁺ = VΣ⁺Uᵀ

Therefore:

β = X⁺Y

SVD is an important numerical tool for solving linear regression problems.

---

## 11. Gradient Descent

Another way to learn regression coefficients is Gradient Descent.

The algorithm:

1. Initialize coefficients.
2. Calculate predictions.
3. Calculate the loss.
4. Calculate gradients.
5. Update coefficients.
6. Repeat.

General update:

parameter = parameter - learning_rate × gradient

A smaller learning rate means smaller updates.

A larger learning rate means larger updates.

---

## 12. Prediction

After learning β:

ŷ = Xβ

For simple regression:

ŷ = β₀ + β₁x

For multiple regression:

ŷ = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ

---

## 13. Regression Evaluation

### Mean Absolute Error

MAE:

MAE = (1/n) Σ|yᵢ - ŷᵢ|

MAE measures the average absolute prediction error.

It is relatively less sensitive to large individual errors than squared-error metrics.

---

### Mean Squared Error

MSE:

MSE = (1/n) Σ(yᵢ - ŷᵢ)²

Because errors are squared, large errors receive much greater weight.

---

### Root Mean Squared Error

RMSE:

RMSE = √MSE

RMSE has the same units as the target variable.

---

### Sum of Absolute Errors

SAE:

SAE = Σ|yᵢ - ŷᵢ|

---

### Sum of Squared Errors

SSE:

SSE = Σ(yᵢ - ŷᵢ)²

---

## 14. MAE vs MSE

Suppose the prediction errors are:

-10, -10, -10, -10, 100

Absolute errors:

10, 10, 10, 10, 100

SAE:

140

Squared errors:

100, 100, 100, 100, 10000

SSE:

10400

The large error of 100 has a much stronger effect on squared-error metrics.

---

## 15. Polynomial Regression

Linear Regression can also be extended to polynomial features.

Example:

y = β₀ + β₁x + β₂x²

For higher degree:

y = β₀ + β₁x + β₂x² + ... + βₙxⁿ

Polynomial regression can model curved relationships.

However, increasing polynomial degree too much can cause overfitting.

---

## 16. Train, Validation and Test Sets

A regression workflow commonly uses:

### Training Set

Used to learn model parameters.

### Validation Set

Used to compare models or tune hyperparameters.

### Test Set

Used for final evaluation on unseen data.

The test set should not influence model training.

---

## 17. Feature Scaling

Linear Regression itself does not always require feature scaling, but scaling can be useful when:

- gradient descent is used
- features have very different magnitudes
- regularization is applied
- numerical stability is important

Common approaches:

### Standardization

x' = (x - μ) / σ

### Min-Max Scaling

x' = (x - xmin) / (xmax - xmin)

Scaling parameters should be learned from the training data only.

---

## 18. Assumptions and Practical Considerations

Linear Regression works best when the relationship between features and target can be reasonably represented by a linear combination.

Important practical considerations include:

- Outliers
- Feature redundancy
- Multicollinearity
- Data leakage
- Nonlinear relationships
- Scale differences
- Overfitting

---

## 19. Advantages

- Simple to understand.
- Easy to implement.
- Fast to train.
- Coefficients are interpretable.
- Useful as a baseline model.
- Works well when the relationship is approximately linear.

---

## 20. Limitations

- Poor at strongly nonlinear relationships without feature transformation.
- Sensitive to outliers, especially with squared-error objectives.
- Multicollinearity can make coefficients unstable.
- High-degree polynomial models can overfit.
- Performance depends on the quality of features.

---

## 21. Basic Workflow

1. Load the dataset.
2. Inspect the data.
3. Select features and target.
4. Handle missing values if necessary.
5. Split into training and test sets.
6. Scale features if appropriate.
7. Train the regression model.
8. Predict on unseen data.
9. Calculate regression metrics.
10. Analyze errors.
11. Improve the model if necessary.

---

## 22. Important Formulas

Simple Linear Regression:

y = β₀ + β₁x

Multiple Linear Regression:

y = β₀ + β₁x₁ + ... + βₙxₙ

Matrix form:

Y = Xβ

Normal Equation:

β = (XᵀX)⁻¹XᵀY

Pseudoinverse solution:

β = X⁺Y

SVD:

X = UΣVᵀ

MAE:

MAE = (1/n)Σ|yᵢ - ŷᵢ|

MSE:

MSE = (1/n)Σ(yᵢ - ŷᵢ)²

RMSE:

RMSE = √MSE

---

## 23. Key Takeaway

Linear Regression learns a relationship between input features and a continuous target.

The central idea is:

features → linear model → prediction → error → minimize error

The matrix formulation provides a compact mathematical representation, while the pseudoinverse and SVD provide useful numerical approaches for obtaining the regression coefficients.
