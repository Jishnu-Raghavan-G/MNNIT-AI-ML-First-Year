# Module 09 — Neural Network Basics

## 1. Introduction

A neural network is a machine learning model inspired by the basic idea of interconnected neurons.

Neural networks learn patterns from data by adjusting numerical parameters called weights and biases.

The lecture deck lists Dense Neural Network, CNN, and LSTM among machine learning algorithms, but it does not provide a detailed derivation of neural networks.

Therefore, this module focuses on the fundamental ideas required to understand a basic dense neural network and its implementation from scratch.

The main idea is:

> A neural network learns by repeatedly making predictions, measuring the error, and adjusting its weights and biases to reduce that error.

---

## 2. Neural Network and Machine Learning

A neural network can be used for supervised learning tasks such as:

- Classification
- Regression

For classification:

`Features → Neural Network → Class prediction`

For regression:

`Features → Neural Network → Continuous value`

The input consists of features, and during supervised training the network uses the known target values to learn the relationship between inputs and outputs.

---

## 3. Basic Structure of a Neural Network

A basic neural network contains:

1. Input layer
2. One or more hidden layers
3. Output layer

Conceptually:

`Input Layer → Hidden Layer(s) → Output Layer`

Example:

`x1, x2, x3`

`↓`

`Hidden neurons`

`↓`

`Output`

Each connection has a weight.

Each neuron generally has a bias.

---

## 4. Input Layer

The input layer receives the features.

Suppose a student dataset contains:

- Hours studied
- Attendance
- Assignment score

Then the input may be represented as:

`x = [hours, attendance, assignment_score]`

If there are three input features, the network receives three input values.

The input layer does not perform the main learning itself.

It passes the feature values to the next layer.

---

## 5. Neuron

A neuron takes inputs, multiplies them by weights, adds a bias, and applies an activation function.

The basic calculation is:

`z = w1x1 + w2x2 + ... + wnxn + b`

or:

`z = w^T x + b`

where:

- `x` = input values
- `w` = weights
- `b` = bias
- `z` = weighted sum

The activation function then produces:

`a = activation(z)`

---

## 6. Weights

Weights determine how strongly individual inputs influence a neuron.

For example:

`z = w1x1 + w2x2 + b`

If `w1` is large, the first input has a stronger influence on the weighted sum.

During training, the neural network changes the weights to improve its predictions.

Therefore:

> Learning in a neural network largely means learning appropriate weights and biases.

---

## 7. Bias

The bias is an additional learnable parameter.

For a neuron:

`z = w^T x + b`

The bias allows the neuron to shift the activation function.

Without a bias, the model can be unnecessarily restricted.

Both weights and biases are learned during training.

---

## 8. Hidden Layers

Hidden layers are the layers between the input and output layers.

A hidden layer contains multiple neurons.

Example:

`Input`

`↓`

`Hidden Layer 1`

`↓`

`Hidden Layer 2`

`↓`

`Output`

Adding hidden layers allows the network to learn increasingly complex representations.

A network with multiple hidden layers is commonly referred to as a deep neural network.

---

## 9. Dense Layer

In a dense layer, each neuron in one layer is connected to every neuron in the previous layer.

For example:

`Input 1 ─┐`

`Input 2 ─┼→ Hidden neuron`

`Input 3 ─┘`

Every hidden neuron can receive all input features.

Dense neural networks are therefore also called fully connected neural networks.

---

## 10. Forward Propagation

Forward propagation means passing input data through the network to produce an output.

For a single neuron:

`z = w^T x + b`

Then:

`a = f(z)`

where `f` is the activation function.

For multiple layers:

`Input → Linear transformation → Activation → Linear transformation → Activation → Output`

The final output is the model's prediction.

---

## 11. Activation Functions

Activation functions introduce non-linearity into neural networks.

Without nonlinear activation functions, stacking many linear layers would still result in an overall linear transformation.

Important activation functions include:

- Step function
- Sigmoid
- Tanh
- ReLU
- Softmax

---

## 12. Step Function

A simple step function can be represented as:

`f(z) = 1 if z >= 0`

`f(z) = 0 otherwise`

It produces a binary output.

Although useful for understanding the basic idea of a neuron, the step function is not convenient for gradient-based neural network training because it does not provide a useful derivative for optimization.

---

## 13. Sigmoid Function

The sigmoid function is:

`sigmoid(z) = 1 / (1 + e^(-z))`

Its output lies between:

`0 and 1`

Therefore, sigmoid is particularly useful for binary classification output.

For example:

`output = 0.90`

can be interpreted as a high predicted probability for the positive class.

---

## 14. Properties of Sigmoid

Important properties:

- Output range is `(0,1)`
- Smooth function
- Useful for probabilities
- Commonly used for binary classification

A problem with sigmoid is that its gradient can become very small for very large positive or negative values.

This can contribute to the vanishing-gradient problem in deep networks.

---

## 15. Tanh

The hyperbolic tangent function is:

`tanh(z)`

Its output lies between:

`-1 and +1`

Unlike sigmoid, tanh is centered around zero.

It can be useful in hidden layers, although ReLU-family activations are commonly preferred in many modern dense networks.

---

## 16. ReLU

ReLU stands for:

> Rectified Linear Unit

The formula is:

`ReLU(z) = max(0,z)`

Therefore:

- If `z < 0`, output is `0`
- If `z > 0`, output is `z`

ReLU is widely used in hidden layers because it is simple and computationally efficient.

---

## 17. Softmax

Softmax converts multiple output scores into values that sum to 1.

For class score `zi`:

`softmax(zi) = e^(zi) / sum(e^(zj))`

Softmax is commonly used in the output layer for multiclass classification.

Example:

`Class A = 0.10`

`Class B = 0.70`

`Class C = 0.20`

The probabilities sum to:

`1.00`

The class with the highest probability can be selected as the predicted class.

---

## 18. Choosing the Output Layer

The output layer depends on the task.

### Binary Classification

A common design is:

`1 output neuron + sigmoid`

Output:

`0 to 1`

### Multiclass Classification

A common design is:

`one output neuron per class + softmax`

### Regression

A common design is:

`1 output neuron + linear activation`

The exact architecture depends on the problem.

---

## 19. Loss Function

A neural network needs a way to measure how wrong its prediction is.

This is the purpose of a loss function.

The training process tries to minimize the loss.

Conceptually:

`Prediction`

`↓`

`Compare with target`

`↓`

`Calculate loss`

`↓`

`Adjust parameters`

---

## 20. Mean Squared Error

For regression, Mean Squared Error can be used:

`MSE = (1/n) sum(yi - y_pred_i)^2`

where:

- `yi` = actual value
- `y_pred_i` = predicted value
- `n` = number of observations

Lower MSE indicates smaller squared prediction errors.

---

## 21. Binary Cross-Entropy

For binary classification, binary cross-entropy can be used:

`L = -[y log(p) + (1-y) log(1-p)]`

where:

- `y` = actual class
- `p` = predicted probability

The loss strongly penalizes confident incorrect predictions.

---

## 22. Multiclass Cross-Entropy

For multiclass classification, categorical cross-entropy is commonly used.

The network produces a probability for each class.

The loss measures how well the predicted probability distribution matches the actual class.

Softmax is commonly paired with multiclass cross-entropy.

---

## 23. Training a Neural Network

Training generally follows this cycle:

`Initialize weights`

`↓`

`Forward propagation`

`↓`

`Calculate loss`

`↓`

`Backpropagation`

`↓`

`Calculate gradients`

`↓`

`Update weights and biases`

`↓`

`Repeat`

The objective is to reduce the loss over training iterations.

---

## 24. Gradient Descent

Gradient descent is an optimization method used to minimize the loss.

The general parameter update is:

`parameter_new = parameter_old - learning_rate × gradient`

For a weight:

`w_new = w_old - η(dL/dw)`

For a bias:

`b_new = b_old - η(dL/db)`

where:

- `η` = learning rate
- `L` = loss

---

## 25. Learning Rate

The learning rate controls the size of parameter updates.

If the learning rate is too large:

- Training may become unstable
- The model may overshoot useful values
- Loss may fail to decrease

If the learning rate is too small:

- Training can become very slow
- Many iterations may be required

Therefore, choosing an appropriate learning rate is important.

---

## 26. Backpropagation

Backpropagation is the process used to calculate how the loss changes with respect to network parameters.

The basic idea is:

1. Perform forward propagation.
2. Calculate the loss.
3. Propagate error information backward.
4. Calculate gradients for weights and biases.
5. Use the gradients to update the parameters.

Backpropagation relies on the chain rule of calculus.

---

## 27. Why Backpropagation Is Important

Consider:

`Input → Hidden Layer → Output`

The output depends on the hidden layer.

The hidden layer depends on the input and its weights.

Therefore, changing a hidden-layer weight can affect the final loss indirectly.

The chain rule allows the network to calculate these dependencies efficiently.

Conceptually:

`Loss`

`↓`

`Output parameters`

`↓`

`Hidden-layer parameters`

`↓`

`Earlier parameters`

---

## 28. A Simple Neuron Example

Suppose:

`x1 = 2`

`x2 = 3`

`w1 = 0.5`

`w2 = -0.2`

`b = 0.1`

Then:

`z = (0.5)(2) + (-0.2)(3) + 0.1`

`z = 1 - 0.6 + 0.1`

`z = 0.5`

Using sigmoid:

`a = 1 / (1 + e^(-0.5))`

The neuron therefore produces a value between 0 and 1.

---

## 29. One Hidden-Layer Neural Network

A simple neural network can have:

`Input Layer`

`↓`

`Hidden Layer`

`↓`

`Output Layer`

Suppose:

- 3 input features
- 4 hidden neurons
- 1 output neuron

Then the architecture is:

`3 → 4 → 1`

The hidden layer learns intermediate representations.

The output neuron produces the final prediction.

---

## 30. Matrix Representation

For a layer, instead of calculating every neuron separately, we can use matrix operations.

For one layer:

`Z = XW + b`

Then:

`A = f(Z)`

where:

- `X` = input matrix
- `W` = weight matrix
- `b` = bias
- `Z` = weighted sums
- `A` = activated outputs

This makes neural network computation efficient.

---

## 31. Multiple Layers

For a network with two hidden layers:

`A1 = f(XW1 + b1)`

`A2 = f(A1W2 + b2)`

`Output = g(A2W3 + b3)`

where:

- `f` = hidden-layer activation
- `g` = output activation

This is the mathematical structure of forward propagation.

---

## 32. Epoch

An epoch means one complete pass through the training dataset.

For example:

`1000 training samples`

One epoch means the model has processed all 1000 samples once.

Training for:

`50 epochs`

means the dataset is processed 50 times.

---

## 33. Batch

A batch is a subset of the training data used for one parameter update.

Suppose there are:

`1000 training samples`

and:

`batch size = 100`

Then one epoch contains approximately:

`1000 / 100 = 10 batches`

The model performs parameter updates after processing each batch.

---

## 34. Batch Gradient Descent

In full-batch gradient descent, the entire training dataset is used to calculate each update.

Advantages:

- Stable gradient estimate
- Simple concept

Disadvantages:

- Can be computationally expensive for large datasets

---

## 35. Stochastic Gradient Descent

In stochastic gradient descent, individual samples can be used for updates.

Advantages:

- Frequent updates
- Can be useful for large datasets

Disadvantages:

- Updates can be noisy

Mini-batch gradient descent provides a compromise by using small batches.

---

## 36. Neural Network Initialization

Weights need initial values before training.

They are commonly initialized with small values.

If all weights are initialized identically, neurons may learn the same thing.

Therefore, suitable weight initialization is important.

Biases are often initialized separately, depending on the architecture and activation functions.

---

## 37. Neural Network Prediction

After training, the network can be used to predict unseen data.

The process is:

`New input`

`↓`

`Forward propagation`

`↓`

`Output`

For binary classification:

`Probability → Threshold → Class`

For example:

`p = 0.82`

Using threshold:

`0.5`

the predicted class is:

`1`

---

## 38. Classification Threshold

For binary classification with sigmoid output:

`p >= threshold → class 1`

`p < threshold → class 0`

A common threshold is:

`0.5`

but it does not have to be fixed at 0.5.

Changing the threshold affects:

- Precision
- Recall
- F1 score
- Number of positive predictions

---

## 39. Neural Networks and Overfitting

Neural networks can overfit when they become too specialized to the training data.

The lecture discusses overfitting and methods such as:

- Simplifying the model
- Regularization
- Increasing training data
- Data augmentation
- Early stopping
- Cross-validation

For neural networks specifically, the lecture mentions:

- Dropout
- Weight decay
- Fewer layers
- Fewer neurons

These techniques can help improve generalization.

---

## 40. Dropout

Dropout is a regularization technique for neural networks.

During training, some neurons are temporarily ignored according to the dropout rate.

This discourages the network from relying too heavily on particular neurons.

The lecture identifies dropout as a method for reducing overfitting.

---

## 41. Weight Decay

Weight decay is a form of regularization that discourages excessively large weights.

It is related to L2 regularization.

Conceptually, the objective becomes:

`Total loss = Data loss + Regularization penalty`

This encourages simpler parameter values and can improve generalization.

---

## 42. Early Stopping

Early stopping monitors validation performance during training.

If validation performance starts getting worse while training performance continues improving, the model may be overfitting.

Training can then be stopped.

The lecture specifically lists early stopping as a method for solving overfitting.

---

## 43. Underfitting

Underfitting occurs when the model is too simple to capture important patterns in the data.

For neural networks, possible solutions include:

- Increasing model complexity
- Adding hidden layers
- Adding neurons
- Adding useful features
- Reducing excessive regularization
- Training for longer

These approaches are consistent with the lecture's discussion of solving underfitting.

---

## 44. Overfitting vs Underfitting

### Underfitting

The model is too simple.

Typical signs:

- Poor training performance
- Poor validation performance

### Overfitting

The model learns the training data too specifically.

Typical signs:

- Very good training performance
- Worse validation/test performance

### Good Fit

The model captures useful patterns and generalizes well to unseen data.

---

## 45. Training, Validation and Testing

A neural network should be evaluated on data that was not used to directly fit its parameters.

A common workflow is:

`Training data → Learn parameters`

`Validation data → Tune model choices`

`Test data → Final evaluation`

The goal is to measure generalization rather than memorization.

---

## 46. Feature Scaling

Neural networks often benefit from appropriately scaled numerical features.

For standardization:

`z = (x - mean) / standard deviation`

Scaling can make optimization more stable and help different features operate on comparable numerical ranges.

Scaling parameters should be learned from the training data and then applied to validation/test data.

---

## 47. Dense Neural Network Workflow

A practical workflow is:

`Load dataset`

`↓`

`Inspect and clean data`

`↓`

`Separate X and y`

`↓`

`Train/test split`

`↓`

`Scale features`

`↓`

`Initialize network`

`↓`

`Forward propagation`

`↓`

`Calculate loss`

`↓`

`Backpropagation`

`↓`

`Update weights`

`↓`

`Repeat for epochs`

`↓`

`Evaluate on unseen data`

`↓`

`Make predictions`

---

## 48. Neural Network From Scratch

A simple implementation can be created using NumPy without using a neural network library.

The implementation needs:

- Weight matrices
- Bias vectors
- Activation functions
- Forward propagation
- Loss calculation
- Backpropagation
- Gradient descent
- Prediction

This makes the internal learning process easier to understand.

---

## 49. Basic From-Scratch Architecture

For a binary classification problem, a simple network can use:

`Input → ReLU hidden layer → Sigmoid output`

For example:

`3 → 8 → 1`

where:

- `3` = number of input features
- `8` = hidden neurons
- `1` = binary output

The exact number of neurons is a design choice.

---

## 50. Important Neural Network Equations

### Neuron

`z = w^T x + b`

### Activation

`a = f(z)`

### Layer

`Z = XW + b`

`A = f(Z)`

### Sigmoid

`σ(z) = 1 / (1 + e^(-z))`

### ReLU

`ReLU(z) = max(0,z)`

### Gradient Descent

`θ_new = θ_old - η∇L`

### Mean Squared Error

`MSE = (1/n) sum(y-y_pred)^2`

### Binary Cross-Entropy

`L = -[y log(p) + (1-y)log(1-p)]`

---

## 51. Dense Neural Network vs Traditional ML

A neural network differs from simpler machine learning models because it can learn multiple layers of representations.

For example:

`Raw features`

`↓`

`Hidden representation`

`↓`

`Another representation`

`↓`

`Prediction`

Traditional models such as KNN, Decision Tree, Logistic Regression and SVM also learn patterns, but their mathematical structures and optimization procedures are different.

---

## 52. Neural Network Advantages

- Can model nonlinear relationships
- Can learn complex patterns
- Can handle many types of data
- Can be extended to deeper architectures
- Useful for classification and regression
- Forms the basis of many deep learning systems

---

## 53. Neural Network Limitations

- Can require significant computation
- Often needs careful preprocessing
- Hyperparameter selection can matter
- Can overfit
- Training can be difficult to interpret
- Requires appropriate architecture and optimization

---

## 54. Dense Neural Network, CNN and LSTM

The lecture's algorithm overview includes:

- Dense Neural Network
- CNN
- LSTM

They are neural-network-based approaches, but they are designed for different kinds of problems.

### Dense Neural Network

Uses fully connected layers.

### CNN

Convolutional Neural Networks are commonly associated with structured spatial data such as images.

### LSTM

Long Short-Term Memory networks are designed to handle sequential dependencies.

The supplied lecture deck lists these algorithms but does not provide detailed derivations of CNN or LSTM.

This module therefore concentrates on basic dense neural networks.

---

## 55. Common Mistakes

### Mistake 1 — Forgetting the Bias

A neuron is generally:

`z = w^T x + b`

not just:

`z = w^T x`

### Mistake 2 — Using the Wrong Output Activation

Binary classification commonly uses sigmoid.

Multiclass classification commonly uses softmax.

Regression commonly uses a linear output.

### Mistake 3 — Not Scaling Features

Very different feature scales can make training more difficult.

### Mistake 4 — Confusing Epoch and Batch

An epoch is one complete pass through the training data.

A batch is a subset processed for one update.

### Mistake 5 — Evaluating Only on Training Data

A model can perform well on training data and still generalize poorly.

### Mistake 6 — Learning From the Test Set

The test set should be reserved for final evaluation.

---

## 56. Exam-Oriented Summary

Remember the following chain:

`Input`

`↓`

`Weighted Sum`

`z = w^T x + b`

`↓`

`Activation Function`

`a = f(z)`

`↓`

`Prediction`

`↓`

`Loss`

`↓`

`Backpropagation`

`↓`

`Gradients`

`↓`

`Gradient Descent`

`↓`

`Updated Weights`

`↓`

`Repeat`

This is the fundamental learning loop of a neural network.

---

## 57. Final Mental Model

A neural network is a collection of connected computational units.

Each neuron:

`takes inputs`

`→ multiplies them by weights`

`→ adds a bias`

`→ applies an activation`

The network:

`makes a prediction`

`→ calculates error`

`→ propagates the error backward`

`→ calculates gradients`

`→ updates weights`

`→ repeats`

The network learns when its parameters gradually change so that the loss becomes smaller and its predictions improve.

The most important concepts to remember are:

- Neuron
- Weight
- Bias
- Layer
- Dense layer
- Activation function
- Forward propagation
- Loss function
- Backpropagation
- Gradient
- Learning rate
- Gradient descent
- Epoch
- Batch
- Overfitting
- Regularization
- Generalization
