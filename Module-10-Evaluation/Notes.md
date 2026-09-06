# Module 10 — Model Evaluation

## 1. Introduction

Model evaluation is the process of measuring how well a machine learning model performs.

A model should not be judged only by how well it performs on the training data.

The important question is:

> How well does the model perform on unseen data?

The evaluation metric depends on the type of machine learning problem.

For classification, common metrics include:

- Accuracy
- Precision
- Recall
- F1-score

For regression, common metrics include:

- MAE
- MSE
- RMSE

The lecture also covers:

- Training, validation and testing
- Overfitting
- Underfitting
- Bias
- Variance
- Regularization

---

## 2. Training, Validation and Testing

Machine learning data can be divided into:

`Training set`

`Validation set`

`Test set`

Each serves a different purpose.

### Training Set

The model learns patterns and parameters from the training data.

During training, model parameters are adjusted to reduce the difference between predictions and actual target values.

### Validation Set

The validation set is used to evaluate the model during development.

It helps identify:

- Overfitting
- Underfitting
- Appropriate model choices

### Test Set

The test set is used after model development for final evaluation.

It represents unseen data and provides an estimate of how well the model generalizes.

---

## 3. Why Not Evaluate Only on Training Data?

Suppose a model memorizes the training examples.

It may achieve:

`Training accuracy = 100%`

but perform poorly on new examples.

This is overfitting.

Therefore:

> Good training performance does not necessarily mean good generalization.

Evaluation should include data that was not used to directly fit the model.

---

## 4. Classification Evaluation

Classification predicts discrete classes.

Examples:

- Spam / Ham
- Pass / Fail
- Disease / No Disease
- Cat / Dog

The lecture identifies three common classification types:

### Binary Classification

Two classes.

Example:

`Pass / Fail`

### Multi-Class Classification

More than two mutually exclusive classes.

Example:

`Cat / Dog / Horse`

### Multi-Label Classification

An observation can belong to multiple labels.

Example:

An image may contain:

`Car + Person + Road`

---

## 5. Confusion Matrix

A confusion matrix summarizes classification predictions.

For binary classification:

| | Predicted Positive | Predicted Negative |
|---|---:|---:|
| Actual Positive | TP | FN |
| Actual Negative | FP | TN |

Where:

### TP — True Positive

Actual positive and predicted positive.

### TN — True Negative

Actual negative and predicted negative.

### FP — False Positive

Actual negative but predicted positive.

### FN — False Negative

Actual positive but predicted negative.

These four quantities are the foundation of common classification metrics.

---

## 6. Accuracy

Accuracy measures the proportion of correctly predicted samples.

Formula:

`Accuracy = (TP + TN) / (TP + FP + FN + TN)`

It answers:

> Out of all predictions, how many were correct?

Example:

Suppose 100 predictions are made and 90 are correct.

`Accuracy = 90 / 100 = 0.90`

Therefore:

`Accuracy = 90%`

---

## 7. Problem With Accuracy

Accuracy can be misleading when classes are highly imbalanced.

Suppose:

- 95 samples are negative
- 5 samples are positive

A model that predicts every sample as negative gets:

`Accuracy = 95%`

but it completely fails to identify the positive class.

Therefore, accuracy should not always be used alone.

---

## 8. Precision

Precision measures how many predicted positives were actually positive.

Formula:

`Precision = TP / (TP + FP)`

It answers:

> Of all samples predicted as positive, how many were actually positive?

High precision means fewer false positives.

---

## 9. Recall

Recall is also called sensitivity.

Formula:

`Recall = TP / (TP + FN)`

It answers:

> Of all actual positive samples, how many did the model correctly identify?

High recall means fewer false negatives.

---

## 10. Precision vs Recall

Precision focuses on:

> Correctness of positive predictions.

Recall focuses on:

> Finding actual positive cases.

Example:

If a spam classifier labels many legitimate emails as spam, precision may be poor.

If it allows many spam messages through, recall may be poor.

The appropriate metric depends on the application.

---

## 11. F1-Score

F1-score is the harmonic mean of precision and recall.

Formula:

`F1 = 2 × Precision × Recall / (Precision + Recall)`

It provides a balanced measure of precision and recall.

F1-score is especially useful when both false positives and false negatives matter.

---

## 12. Example of Classification Metrics

Suppose:

`TP = 40`

`TN = 50`

`FP = 5`

`FN = 5`

Then:

`Accuracy = (40 + 50) / 100`

`Accuracy = 0.90`

Precision:

`Precision = 40 / (40 + 5)`

`Precision ≈ 0.889`

Recall:

`Recall = 40 / (40 + 5)`

`Recall ≈ 0.889`

F1:

`F1 ≈ 0.889`

---

## 13. Which Classification Metric Should Be Used?

### Accuracy

Useful when classes are reasonably balanced and overall correctness matters.

### Precision

Important when false positives are costly.

### Recall

Important when false negatives are costly.

### F1-score

Useful when a balance between precision and recall is required.

There is no single universally best metric.

The metric should match the problem.

---

## 14. Regression Evaluation

Regression predicts continuous numerical values.

Examples:

- House price
- Salary
- Temperature
- Student score

The lecture introduces:

- MSE
- MAE
- RMSE

These metrics compare actual values with predicted values.

---

## 15. Prediction Error

For a data point:

`Error = y - y_pred`

where:

- `y` = actual value
- `y_pred` = predicted value

The sign indicates whether the prediction is above or below the actual value.

For many evaluation metrics, the magnitude of the error is more important than its direction.

---

## 16. Mean Absolute Error — MAE

MAE stands for:

> Mean Absolute Error

Formula:

`MAE = (1/n) Σ |yi - y_pred_i|`

It calculates the average absolute prediction error.

Example:

Actual:

`[10, 20, 30]`

Predicted:

`[12, 18, 27]`

Errors:

`[-2, 2, 3]`

Absolute errors:

`[2, 2, 3]`

Therefore:

`MAE = (2 + 2 + 3) / 3`

`MAE = 2.33`

---

## 17. Interpretation of MAE

MAE is easy to interpret because it uses the same units as the target.

If:

`MAE = 5`

then the model's predictions are off by about 5 target units on average in absolute terms.

MAE does not square the errors.

Therefore, it is less dominated by very large errors than squared-error metrics.

---

## 18. Mean Squared Error — MSE

MSE stands for:

> Mean Squared Error

Formula:

`MSE = (1/n) Σ (yi - y_pred_i)^2`

Each error is squared before taking the average.

Example:

Errors:

`[-2, 2, 3]`

Squared errors:

`[4, 4, 9]`

Therefore:

`MSE = (4 + 4 + 9) / 3`

`MSE = 5.67`

---

## 19. Why Does MSE Penalize Large Errors?

Suppose the errors are:

`10`

and:

`100`

Absolute errors are:

`10` and `100`

Squared errors are:

`100` and `10,000`

The large error becomes much more influential after squaring.

Therefore:

> MSE places greater emphasis on large errors.

---

## 20. Root Mean Squared Error — RMSE

RMSE is the square root of MSE.

Formula:

`RMSE = sqrt((1/n) Σ (yi - y_pred_i)^2)`

Because the square root is taken, RMSE has the same units as the target.

For the previous example:

`MSE = 5.67`

Therefore:

`RMSE = sqrt(5.67)`

`RMSE ≈ 2.38`

---

## 21. MAE vs MSE

| Property | MAE | MSE |
|---|---|---|
| Error operation | Absolute value | Square |
| Large errors | Less emphasis | Strong emphasis |
| Outlier sensitivity | Lower | Higher |
| Units | Same as target | Squared target units |
| Interpretation | Easy | Less direct |

The lecture explains that MAE is more robust to outliers, while SSE/MSE gives greater emphasis to large deviations.

---

## 22. SAE and SSE

The lecture also compares:

### Sum of Absolute Errors

`SAE = Σ |yi - y_pred_i|`

### Sum of Squared Errors

`SSE = Σ (yi - y_pred_i)^2`

MAE is essentially the average version of absolute error.

MSE is the average version of squared error.

---

## 23. Lecture Example — SAE vs SSE

Suppose the prediction errors are:

`-10, -10, -10, -10, 100`

SAE:

`10 + 10 + 10 + 10 + 100`

`SAE = 140`

SSE:

`100 + 100 + 100 + 100 + 10000`

`SSE = 10400`

The large error of 100 has a much greater effect on SSE.

This illustrates why squared-error measures are more sensitive to large deviations.

---

## 24. When to Prefer MAE

MAE can be useful when:

- Outliers should not dominate the metric
- Robustness to large deviations is important
- Easy interpretation is desired

The lecture gives examples such as house-price prediction and situations where data can contain outliers.

---

## 25. When to Prefer MSE

MSE can be useful when:

- Large errors should be penalized strongly
- Large deviations are particularly undesirable
- Squared-error optimization is appropriate

The lecture gives examples such as weather prediction and energy consumption forecasting.

---

## 26. Evaluation Is More Than One Number

A single metric may not fully describe model performance.

For classification, examine:

- Confusion matrix
- Accuracy
- Precision
- Recall
- F1-score

For regression, examine:

- MAE
- MSE
- RMSE

Also compare:

- Training performance
- Validation performance
- Test performance

---

## 27. Overfitting

Overfitting occurs when a model learns the training data too well, including noise and randomness.

The model performs well on training data but fails to generalize to unseen data.

A typical pattern is:

`Training error ↓`

while:

`Validation/Test error ↑`

This indicates that the model may be memorizing the training data.

---

## 28. Underfitting

Underfitting occurs when a model is too simple to capture the underlying structure of the data.

The model performs poorly on both:

- Training data
- Unseen data

A typical pattern is:

`Training error = high`

`Validation/Test error = high`

The model has not learned enough from the data.

---

## 29. Good Fit

A good model should learn useful patterns without simply memorizing the training data.

Ideally:

`Training performance = good`

`Validation performance = good`

`Test performance = good`

The goal is good generalization.

---

## 30. Bias

The lecture describes bias as the inability of a model to properly capture the relationship in the training data.

High bias is commonly associated with underfitting.

A high-bias model is often too simple.

---

## 31. Variance

Variance refers to a model's sensitivity to small changes in the training data.

A high-variance model can capture noise along with the actual pattern.

High variance is commonly associated with overfitting.

---

## 32. Bias-Variance Tradeoff

There is a tradeoff between model complexity, bias and variance.

Very simple model:

`High bias + Low variance`

Very complex model:

`Low bias + High variance`

The goal is to find a useful balance.

Conceptually:

`Model complexity increases`

`↓`

`Bias tends to decrease`

`↓`

`Variance tends to increase`

A good model balances the two.

---

## 33. Solving Underfitting

The lecture suggests methods such as:

### Increase Model Complexity

For example:

- More complex model
- More hidden layers
- More neurons

### Feature Engineering

Add useful features or transformations.

### Reduce Regularization

If regularization is too strong, the model may be overly constrained.

### Train Longer

More training can allow the model to learn additional patterns.

### Change Algorithm

A more powerful algorithm may capture relationships that the current model cannot.

---

## 34. Solving Overfitting

The lecture suggests:

### Simplify the Model

Reduce model complexity.

### Regularization

Use techniques such as:

- L1 regularization
- L2 regularization
- Weight decay

### Increase Training Data

More representative training data can improve generalization.

### Data Augmentation

Generate additional training examples where appropriate.

### Early Stopping

Stop training when validation performance starts degrading.

### Cross-Validation

Use techniques such as k-fold cross-validation to assess generalization.

### Pruning

For tree-based models, limit depth or remove unnecessary branches.

### Reduce Feature Redundancy

Remove irrelevant or highly correlated features using feature selection or PCA.

---

## 35. L1 Regularization

L1 regularization adds a penalty related to the absolute values of model parameters.

Conceptually:

`Objective = Original loss + λ Σ|wi|`

where:

- `λ` controls regularization strength
- `wi` are model parameters

L1 regularization can encourage some coefficients to become exactly zero.

This can produce sparse models.

---

## 36. L2 Regularization

L2 regularization adds a penalty based on squared parameter values.

Conceptually:

`Objective = Original loss + λ Σwi²`

It discourages excessively large parameter values.

L2 regularization is commonly associated with Ridge regression and weight decay in neural networks.

---

## 37. Regularization Strength

The parameter `λ` controls how strongly regularization affects the model.

If regularization is too weak:

`Overfitting may remain`

If regularization is too strong:

`Model may become too simple`

Therefore, regularization strength must be selected appropriately.

---

## 38. Model Evaluation Workflow

A practical evaluation workflow is:

`Load dataset`

`↓`

`Clean and prepare data`

`↓`

`Split into training / validation / test`

`↓`

`Train model`

`↓`

`Evaluate validation performance`

`↓`

`Tune model`

`↓`

`Evaluate final model on test set`

`↓`

`Analyze metrics`

`↓`

`Check generalization`

---

## 39. Classification Evaluation Workflow

For classification:

`Actual labels`

`+`

`Predicted labels`

`↓`

`Confusion Matrix`

`↓`

`TP / TN / FP / FN`

`↓`

`Accuracy`

`Precision`

`Recall`

`F1-score`

---

## 40. Regression Evaluation Workflow

For regression:

`Actual values`

`+`

`Predicted values`

`↓`

`Calculate errors`

`↓`

`MAE`

`MSE`

`RMSE`

Then compare the results across models or datasets.

---

## 41. Threshold and Classification

Many binary classifiers produce a probability rather than a direct class.

For example:

`Prediction probability = 0.72`

A threshold converts this probability into a class.

With threshold 0.5:

`0.72 >= 0.5 → Positive`

If the threshold is changed:

`0.72 >= 0.8 → Negative`

Therefore, changing the threshold changes the predicted classes.

---

## 42. Threshold and Precision-Recall Tradeoff

Changing the classification threshold changes the balance between precision and recall.

Generally:

### Lower Threshold

More observations are classified as positive.

This can increase recall but may also increase false positives.

### Higher Threshold

Fewer observations are classified as positive.

This can increase precision but may also increase false negatives.

Therefore:

> The classification threshold should be selected according to the application's objective.

---

## 43. Example Threshold Experiment

Suppose predicted probabilities are:

`0.10, 0.35, 0.55, 0.70, 0.90`

At threshold `0.5`:

`0, 0, 1, 1, 1`

At threshold `0.7`:

`0, 0, 0, 1, 1`

Changing the threshold changes the predicted classes.

Therefore, precision, recall and F1-score can also change.

---

## 44. Comparing Models

Suppose two classification models produce:

| Metric | Model A | Model B |
|---|---:|---:|
| Accuracy | 0.90 | 0.88 |
| Precision | 0.82 | 0.91 |
| Recall | 0.95 | 0.78 |
| F1 | 0.88 | 0.84 |

Model A has higher recall.

Model B has higher precision.

The better model depends on the application's requirements.

Therefore:

> Do not select a model using one metric without considering the problem.

---

## 45. Evaluation on Unseen Data

The test set should represent data that the model has not used for training.

A strong evaluation asks:

- Does the model generalize?
- Are errors acceptable?
- Which classes are misclassified?
- Are false positives important?
- Are false negatives important?
- Does the model overfit?
- Does the metric match the application?

---

## 46. Common Evaluation Mistakes

### Mistake 1 — Using Training Accuracy as Final Performance

Training performance can be overly optimistic.

### Mistake 2 — Using Accuracy for Every Classification Problem

Accuracy can be misleading with imbalanced classes.

### Mistake 3 — Ignoring Precision and Recall

Different applications have different costs for FP and FN.

### Mistake 4 — Comparing MAE and MSE Without Understanding Their Difference

MSE gives much greater emphasis to large errors.

### Mistake 5 — Using the Test Set Repeatedly for Model Tuning

This can cause the test set to influence model development.

### Mistake 6 — Ignoring Overfitting

A model with excellent training performance may still generalize poorly.

---

## 47. Important Formulas

### Accuracy

`Accuracy = (TP + TN) / (TP + FP + FN + TN)`

### Precision

`Precision = TP / (TP + FP)`

### Recall

`Recall = TP / (TP + FN)`

### F1-score

`F1 = 2 × Precision × Recall / (Precision + Recall)`

### MAE

`MAE = (1/n) Σ|yi - y_pred_i|`

### MSE

`MSE = (1/n) Σ(yi - y_pred_i)^2`

### RMSE

`RMSE = sqrt((1/n) Σ(yi - y_pred_i)^2)`

### SAE

`SAE = Σ|yi - y_pred_i|`

### SSE

`SSE = Σ(yi - y_pred_i)^2`

---

## 48. Important Terms for Exams

Remember:

- Training set
- Validation set
- Test set
- Confusion matrix
- True Positive
- True Negative
- False Positive
- False Negative
- Accuracy
- Precision
- Recall
- F1-score
- MAE
- MSE
- RMSE
- SAE
- SSE
- Overfitting
- Underfitting
- Bias
- Variance
- L1 regularization
- L2 regularization
- Generalization
- Classification threshold

---

## 49. Final Mental Model

For classification:

`Predictions`

`↓`

`Confusion Matrix`

`↓`

`TP / TN / FP / FN`

`↓`

`Accuracy / Precision / Recall / F1`

For regression:

`Predictions`

`↓`

`Prediction Errors`

`↓`

`MAE / MSE / RMSE`

For model quality:

`Training performance`

`+`

`Validation performance`

`+`

`Test performance`

`↓`

`Generalization`

And always remember:

> A good machine learning model is not simply the model with the best training score. It is the model that performs well on unseen data using metrics appropriate for the problem.
