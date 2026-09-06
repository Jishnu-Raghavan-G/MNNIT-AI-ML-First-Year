# Module 03 — Naive Bayes

## 1. Introduction

Naive Bayes is a statistical classification algorithm based on Bayes' theorem.

It performs probabilistic prediction by calculating the probability that a given observation belongs to each possible class.

The main idea is:

Given some observed features, calculate the probability of each possible class and choose the class with the highest probability.

Naive Bayes makes a simplified assumption that the features are conditionally independent given the class.

This assumption is called "naive" because real-world features are often not completely independent.

The lecture introduces Naive Bayes as:

- A statistical classifier
- Based on Bayes' theorem
- Using conditional independence between features
- Producing probabilistic class predictions

## 2. Bayes' Theorem

Bayes' theorem describes how the probability of a hypothesis changes when evidence is observed.

The basic formula is:

`P(A | B) = [P(B | A) × P(A)] / P(B)`

Where:

- `P(A | B)` = posterior probability of A given B
- `P(B | A)` = likelihood of B given A
- `P(A)` = prior probability of A
- `P(B)` = probability of the evidence

For Naive Bayes, let:

- `Cᵢ` = a class
- `F` = the complete feature set
- `F = {f₁, f₂, ..., fₙ}`

Then:

`P(Cᵢ | F) = [P(F | Cᵢ) × P(Cᵢ)] / P(F)`

The objective is to determine which class has the highest posterior probability.

## 3. Posterior Probability

The posterior probability represents the probability of a class after considering the observed features.

For example:

`P(Yes | F)`

means the probability that an observation belongs to the `Yes` class given the observed feature set `F`.

Similarly:

`P(No | F)`

means the probability that the observation belongs to the `No` class.

The predicted class is the class with the larger posterior probability.

## 4. Simplifying Bayes' Theorem for Classification

For classification, `P(F)` is the same for every candidate class.

Therefore, when comparing classes, it can be ignored.

The classification expression becomes:

`P(Cᵢ | F) ∝ P(F | Cᵢ) × P(Cᵢ)`

Therefore, we calculate:

`P(F | Cᵢ) × P(Cᵢ)`

for every class and select the class with the maximum value.

## 5. Conditional Independence Assumption

Suppose the feature set is:

`F = {f₁, f₂, ..., fₙ}`

Naive Bayes assumes that the features are conditionally independent given the class.

Therefore:

`P(F | Cᵢ) = P(f₁ | Cᵢ) × P(f₂ | Cᵢ) × ... × P(fₙ | Cᵢ)`

The complete classification expression becomes:

`P(Cᵢ | F) ∝ P(Cᵢ) × ∏ P(fⱼ | Cᵢ)`

This assumption makes Naive Bayes computationally simple and fast.

In practice, features may have dependencies, so the independence assumption is not always strictly true.

## 6. Components of Naive Bayes

Naive Bayes classification mainly requires:

1. Prior probability
2. Likelihood of each feature given a class
3. Posterior probability

### Prior Probability

The prior probability of class `C` is:

`P(C) = Number of samples in class C / Total number of samples`

For example, if 9 out of 14 observations belong to the `Yes` class:

`P(Yes) = 9 / 14`

If 5 out of 14 belong to the `No` class:

`P(No) = 5 / 14`

### Likelihood

The likelihood describes how probable a feature value is given a particular class.

For example:

`P(Income = Medium | Yes)`

### Posterior

The posterior combines the prior and likelihood:

`P(Class | Features) ∝ P(Features | Class) × P(Class)`

## 7. Categorical Features

When a feature is categorical, its conditional probability can be estimated from the relative frequency of that feature value within a class.

For example, suppose:

`P(Student = Yes | Buys = Yes)`

is required.

It can be calculated as:

`Number of Yes-class samples where Student = Yes / Total number of Yes-class samples`

The same calculation is performed for every feature and every candidate class.

## 8. Categorical Naive Bayes Example

The lecture uses a small "buys computer" dataset containing:

- Age
- Income
- Student
- Credit Rating
- Buys Computer

The target class is:

- Yes
- No

The observation to classify is approximately:

- Age = <= 30
- Income = Medium
- Student = Yes
- Credit Rating = Fair

We calculate the probability for each class.

### For the Yes Class

The lecture calculates:

`P(Yes | F) ∝ P(Age <= 30 | Yes) × P(Income = Medium | Yes) × P(Student = Yes | Yes) × P(Credit = Fair | Yes) × P(Yes)`

Using the frequencies in the dataset:

`P(Yes | F) ∝ (2/9) × (4/9) × (6/9) × (6/9) × (9/14)`

This gives approximately:

`P(Yes | F) = 0.028`

### For the No Class

Similarly:

`P(No | F) ∝ P(Age <= 30 | No) × P(Income = Medium | No) × P(Student = Yes | No) × P(Credit = Fair | No) × P(No)`

Using the dataset frequencies:

`P(No | F) ∝ (3/5) × (2/5) × (1/5) × (2/5) × (5/14)`

This gives approximately:

`P(No | F) = 0.007`

Since:

`0.028 > 0.007`

the observation is classified as:

`Yes`

## 9. A Second Categorical Example

The lecture also uses the classic "Play Golf" dataset.

Features include:

- Outlook
- Temperature
- Humidity
- Windy

Target:

- Play = Yes
- Play = No

The observation to classify is:

- Outlook = Rainy
- Temperature = Cool
- Humidity = High
- Windy = True

For the `Yes` class, the lecture obtains approximately:

`P(Yes | A) = 0.00529`

For the `No` class:

`P(No | A) = 0.02057`

Since:

`0.02057 > 0.00529`

the observation is classified as:

`No`

The important procedure is:

1. Calculate the prior probability of each class.
2. Calculate the conditional probability of every feature for each class.
3. Multiply the probabilities.
4. Compare the resulting class scores.
5. Select the class with the highest score.

## 10. Zero-Frequency Problem

A major problem can occur when one of the conditional probabilities is zero.

Suppose:

`P(feature value | class) = 0`

Then the complete Naive Bayes product becomes zero:

`P(Class | Features) ∝ P(Class) × 0 × ... = 0`

This means that a single unseen feature-category combination can cause the entire class probability to become zero.

A common solution is Laplace smoothing.

## 11. Laplace Smoothing

Laplace smoothing adjusts frequency-based probabilities so that unseen feature values do not produce zero probability.

A common form is:

`P(feature value | class) = (count + 1) / (class count + number of possible feature values)`

The `+1` prevents the numerator from becoming zero.

For example, if:

- Count of a feature value = 0
- Number of possible feature values = 3
- Number of samples in the class = 10

then:

`P = (0 + 1) / (10 + 3)`

`P = 1 / 13`

The exact denominator depends on the number of possible values for the feature.

Laplace smoothing is particularly useful for categorical Naive Bayes when some combinations do not appear in the training data.

## 12. Continuous Features

Naive Bayes can also work with continuous numerical features.

For a continuous feature, the lecture uses a Gaussian distribution.

The probability density is:

`g(f, μ, σ) = [1 / (σ√(2π))] × exp[-(1/2)((f - μ)/σ)²]`

Where:

- `f` = observed feature value
- `μ` = mean of the feature for the class
- `σ` = standard deviation of the feature for the class

For each class, the mean and standard deviation of each continuous feature are estimated from the training data.

The Gaussian probability density is then used as the likelihood.

## 13. Gaussian Naive Bayes

For continuous features, the process is:

1. Separate training observations according to class.
2. Calculate the prior probability for every class.
3. Calculate the mean of every numerical feature for each class.
4. Calculate the standard deviation of every numerical feature for each class.
5. Calculate Gaussian likelihoods for the new observation.
6. Multiply the likelihoods with the class prior.
7. Compare the resulting class scores.
8. Predict the class with the highest score.

For two continuous features `f₁` and `f₂`:

`P(C | f₁, f₂) ∝ P(f₁ | C) × P(f₂ | C) × P(C)`

## 14. Gaussian Naive Bayes Example

The lecture provides an example involving diabetes classification.

Features include:

- Glucose
- BMI

Classes:

- No Diabetes
- Yes Diabetes

The example first calculates the class priors.

For the given six observations:

`P(No) = 3/6 = 0.5`

`P(Yes) = 3/6 = 0.5`

For the No Diabetes class, the lecture calculates statistics such as:

`μGlucose,No = 85`

and:

`σ²Glucose,No = 16.67`

For BMI:

`μBMI,No ≈ 25.17`

and:

`σ²BMI,No ≈ 1.08`

The same process is performed for the Yes Diabetes class.

For a new patient, Gaussian likelihoods are calculated for each feature under each class.

The lecture then compares:

`P(No | Features)`

with:

`P(Yes | Features)`

and selects the class with the larger probability.

In the lecture's example, the resulting comparison is:

`P(No | 120, 30) = 0`

`P(Yes | 120, 30) = 0.0000000000015`

Therefore, the prediction is:

`Yes Diabetes`

## 15. Why Naive Bayes Works

The main advantage of Naive Bayes is that the conditional independence assumption greatly simplifies probability calculations.

Instead of estimating the probability of a complete combination of all features directly, we estimate individual conditional probabilities and multiply them.

This makes the algorithm:

- Simple
- Fast
- Computationally efficient
- Suitable for classification problems
- Capable of producing probabilities

## 16. Advantages

According to the lecture, Naive Bayes has the following advantages:

- Fast to train.
- Fast to classify.
- Can handle real and discrete data.
- Uses a relatively simple probabilistic framework.
- Works well when the independence assumption is reasonably suitable.

## 17. Disadvantages

The major disadvantage is the conditional independence assumption.

The lecture notes that:

- Features are assumed to be independent given the class.
- In practical datasets, dependencies often exist between variables.

Therefore, the assumption may not always accurately represent real-world data.

Another practical issue is the zero-frequency problem, which can be addressed using smoothing techniques such as Laplace smoothing.

## 18. Naive Bayes Classification Workflow

A complete Naive Bayes workflow is:

Training Dataset
↓
Identify Classes
↓
Calculate Class Priors
↓
Calculate Feature Likelihoods
↓
Apply Conditional Independence
↓
Calculate Class Scores
↓
Compare Class Scores
↓
Select Highest Probability
↓
Final Prediction

For categorical data:

`Likelihood = Relative Frequency`

For continuous data:

`Likelihood = Gaussian Probability Density`

## 19. Important Formula Summary

### Bayes' Theorem

`P(C | F) = [P(F | C) × P(C)] / P(F)`

### Classification Form

`P(C | F) ∝ P(F | C) × P(C)`

### Naive Independence

`P(F | C) = ∏ P(fᵢ | C)`

### Combined Naive Bayes Formula

`P(C | F) ∝ P(C) × ∏ P(fᵢ | C)`

### Gaussian Density

`g(f, μ, σ) = [1 / (σ√(2π))] × exp[-(1/2)((f - μ)/σ)²]`

### Laplace Smoothing

`P(feature value | class) = (count + 1) / (class count + number of possible values)`

## 20. Key Points

- Naive Bayes is a probabilistic classification algorithm.
- It is based on Bayes' theorem.
- It assumes conditional independence between features.
- The prior represents the probability of a class before observing the features.
- The likelihood represents the probability of observing a feature given a class.
- The posterior represents the updated probability of a class after observing the features.
- The class with the highest posterior score is selected.
- Categorical features can be handled using relative frequencies.
- Continuous features can be modeled using a Gaussian distribution.
- Laplace smoothing helps prevent zero probabilities.
- Naive Bayes is fast to train and fast to classify.
- Its main limitation is the conditional independence assumption.
