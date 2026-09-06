# MNNIT AI & Machine Learning (First Year)

> Personal notes, implementations, experiments, and projects developed while learning Artificial Intelligence and Machine Learning during my first year at Motilal Nehru National Institute of Technology Allahabad (MNNIT Allahabad).

## About

This repository contains my **first-year Artificial Intelligence and Machine Learning coursework, notes, mathematical explanations, Python implementations, experiments, and projects**.

The course concepts are based primarily on the lecture material provided during the AI/ML course. The material has been **rewritten, reorganized, and implemented in my own words and code** for learning, revision, and practical understanding.

This repository is a **personal learning resource** and is **not an official MNNIT repository or official course material**.

## Topics Covered

### Machine Learning
- Introduction to Machine Learning
- Why Machine Learning?
- Supervised Learning
- Unsupervised Learning
- Reinforcement Learning
- Classification
- Regression
- Features and Targets
- Training, Validation and Testing
- ML Workflow

### Classification
- Naive Bayes
- K-Nearest Neighbours (KNN)
- Decision Trees
- Logistic Regression
- Support Vector Machines (SVM)

### Unsupervised Learning
- K-Means Clustering
- Euclidean Distance
- Manhattan Distance
- Centroids
- WCSS
- Elbow Method
- Silhouette Score
- Feature Scaling

### Neural Networks
- Neurons
- Weights and Biases
- Dense Layers
- Forward Propagation
- Activation Functions
- Loss Functions
- Gradient Descent
- Backpropagation
- Neural Network Basics
- Overfitting Prevention

### Model Evaluation
- Confusion Matrix
- Accuracy
- Precision
- Recall
- F1 Score
- MAE
- MSE
- RMSE
- SAE
- SSE
- Classification Thresholds
- Overfitting
- Underfitting
- Bias
- Variance
- Regularization

### Artificial Intelligence
- Artificial Agents
- Rational Agents
- State Spaces
- Trees
- Graphs
- BFS
- DFS
- Uniform-Cost Search
- Best-First Search
- Greedy Best-First Search
- A*
- Heuristics
- Admissibility
- Consistency
- 8-Puzzle
- Tree Traversals
- Search Complexity

## Repository Structure

```text
MNNIT-AI-ML-First-Year/

├── datasets/
│   ├── README.md
│   ├── student_performance.csv
│   ├── customers.csv
│   └── spam_messages.csv
│
├── assets/
│   └── README.md
│
├── Module-01-Introduction/
│   ├── Notes.md
│   └── ml_workflow.py
│
├── Module-02-Data-Preparation/
│   ├── Notes.md
│   └── data_preparation.py
│
├── Module-03-Naive-Bayes/
│   ├── Notes.md
│   └── code/
│       ├── naive_bayes_from_scratch.py
│       └── gaussian_naive_bayes.py
│
├── Module-04-KNN/
│   ├── Notes.md
│   └── code/
│       └── knn_from_scratch.py
│
├── Module-05-Decision-Tree/
│   ├── Notes.md
│   └── code/
│       └── decision_tree_from_scratch.py
│
├── Module-06-Logistic-Regression/
│   ├── Notes.md
│   └── code/
│       └── logistic_regression_from_scratch.py
│
├── Module-07-SVM/
│   ├── Notes.md
│   └── code/
│       └── svm_margin_demo.py
│
├── Module-08-KMeans/
│   ├── Notes.md
│   └── code/
│       ├── kmeans_from_scratch.py
│       └── kmeans_evaluation.py
│
├── Module-09-Neural-Network-Basics/
│   ├── Notes.md
│   └── code/
│       └── neural_network_from_scratch.py
│
├── Module-10-Evaluation/
│   ├── Notes.md
│   └── code/
│       ├── evaluation_metrics.py
│       └── threshold_experiment.py
│
└── Extras/
    ├── 01-Linear-Regression/
    │   ├── Notes.md
    │   └── linear_regression.py
    │
    ├── 02-AI-Search/
    │   ├── Notes.md
    │   └── search_algorithms.py
    │
    ├── 03-Project-Student-Performance/
    │   ├── README.md
    │   └── student_performance_project.py
    │
    ├── 04-Project-Customer-Segmentation/
    │   ├── README.md
    │   └── customer_segmentation_project.py
    │
    ├── 05-Project-Spam-Classifier/
    │   ├── README.md
    │   └── spam_classifier_project.py
    │
    ├── 06-AI-Agent-Assignment/
    │   ├── README.md
    │   └── artificial_agents.py
    │
    ├── 07-Bias-Variance/
    │   ├── Notes.md
    │   └── bias_variance_demo.py
    │
    ├── 08-Regularization/
    │   ├── Notes.md
    │   └── regularization_demo.py
    │
    ├── 09-Regression-Matrix-SVD/
    │   ├── Notes.md
    │   └── matrix_svd_regression.py
    │
    ├── 10-Tree-Traversal/
    │   ├── Notes.md
    │   └── tree_traversal.py
    │
    └── 11-Search-Complexity/
        ├── Notes.md
        └── search_complexity.py
```

## Modules

### Module 01 — Introduction

Fundamentals of Machine Learning and the complete ML workflow.

- Machine Learning
- Supervised, Unsupervised and Reinforcement Learning
- Classification
- Regression
- Features and Targets
- Training, Validation and Testing
- ML Workflow

Includes a practical ML workflow implementation using the student-performance dataset.

### Module 02 — Data Preparation

Preparing datasets before model training.

- Data inspection
- Missing values
- Duplicate records
- Numerical and categorical data
- Feature/target separation
- Train/test splitting
- Validation
- Standardization
- Min-Max scaling
- Data leakage
- Reproducibility

### Module 03 — Naive Bayes

Probability-based classification.

- Bayes' Theorem
- Prior
- Likelihood
- Posterior
- Conditional Independence
- Categorical Naive Bayes
- Gaussian Naive Bayes
- Laplace Smoothing
- Zero-Frequency Problem
- Worked Examples

Includes categorical Naive Bayes from scratch and Gaussian Naive Bayes.

### Module 04 — KNN

Distance-based classification.

- KNN
- Choice of K
- Euclidean Distance
- Manhattan Distance
- Neighbour Selection
- Majority Voting
- Weighted KNN
- Feature Scaling
- Curse of Dimensionality

Includes a KNN classifier implemented from scratch.

### Module 05 — Decision Tree

Tree-based classification.

- Tree Structure
- Gini Impurity
- Entropy
- Information Gain
- Splitting
- Recursive Construction
- Stopping Conditions
- Pruning
- Feature Importance

Includes a numerical Decision Tree classifier implemented from scratch.

### Module 06 — Logistic Regression

Classification using probabilities and the sigmoid function.

- Binary Classification
- Sigmoid Function
- Probability
- Log-Odds
- Decision Boundary
- Binary Cross-Entropy
- Gradient Descent
- Learning Rate
- Regularization
- Classification Threshold

Includes Logistic Regression implemented from scratch.

### Module 07 — SVM

Fundamentals of Support Vector Machines.

- Hyperplanes
- Margins
- Support Vectors
- Hard Margin
- Soft Margin
- C Parameter
- Hinge Loss
- Kernel Methods
- Linear Kernel
- Polynomial Kernel
- RBF Kernel
- Gamma
- Feature Scaling

Includes geometric demonstrations and practical SVM experiments.

### Module 08 — K-Means

Clustering and cluster evaluation.

- Clustering
- Centroids
- Euclidean Distance
- Manhattan Distance
- Assignment Step
- Centroid Update
- WCSS
- Elbow Method
- Silhouette Score
- Feature Scaling
- K-Means++
- Convergence
- Customer Segmentation

Includes a from-scratch implementation and a scikit-learn evaluation implementation.

### Module 09 — Neural Network Basics

Fundamentals of dense neural networks.

- Neurons
- Weights
- Biases
- Dense Layers
- Forward Propagation
- Activation Functions
- ReLU
- Sigmoid
- Tanh
- Softmax
- Loss Functions
- Gradient Descent
- Backpropagation
- Epochs
- Batches
- Weight Initialization
- Overfitting Prevention

Includes a small neural network implemented from scratch using NumPy.

### Module 10 — Evaluation

Evaluation of classification and regression models.

#### Classification

- Confusion Matrix
- TP
- TN
- FP
- FN
- Accuracy
- Precision
- Recall
- F1 Score

#### Regression

- Absolute Error
- SAE
- MAE
- Squared Error
- SSE
- MSE
- RMSE

Also covers:

- Training/Validation/Test evaluation
- Classification thresholds
- Overfitting
- Underfitting
- Bias
- Variance
- Regularization

## Extras

### Extra 01 — Linear Regression

- Simple Linear Regression
- Multiple Linear Regression
- Least Squares
- Matrix Representation
- Normal Equation
- Pseudoinverse
- SVD
- Gradient Descent
- Regression Metrics
- Polynomial Regression

### Extra 02 — AI Search

- State Spaces
- Search Trees
- Graphs
- Nodes
- Queues
- Stacks
- Priority Queues
- BFS
- DFS
- Uniform-Cost Search
- Greedy Best-First Search
- A*
- Heuristics
- Admissibility
- Consistency
- 8-Puzzle

### Extra 03 — Student Performance Project

End-to-end regression project using student-performance data.

- Data Loading
- Data Inspection
- Train/Test Split
- Linear Regression
- Prediction
- MAE
- MSE
- RMSE
- New Student Prediction

### Extra 04 — Customer Segmentation Project

K-Means customer segmentation project.

- Feature Scaling
- K-Means
- Distance Calculations
- WCSS
- Elbow Analysis
- Silhouette Score
- Cluster Assignments
- Cluster Profiling

### Extra 05 — Spam Classifier Project

Text-classification project for spam detection.

- Text Processing
- Train/Test Split
- TF-IDF
- Multinomial Naive Bayes
- Classification Metrics
- Confusion Matrix
- New Message Prediction

### Extra 06 — AI Agent Assignment

Artificial agents and agent architectures.

- Agents
- Environments
- Rational Agents
- Simple Reflex Agents
- Model-Based Agents
- Goal-Based Agents
- Utility-Based Agents
- Environment Properties
- Agent Architecture
- Vacuum-World Example
- Navigation Example

### Extra 07 — Bias-Variance

- Bias
- Variance
- Underfitting
- Overfitting
- Good Fit
- Bias-Variance Tradeoff
- Model Complexity
- Training Error
- Test Error
- Validation
- Cross-Validation
- Regularization

### Extra 08 — Regularization

- Regularization
- L1 / Lasso
- L2 / Ridge
- Regularization Strength
- Feature Scaling
- Polynomial Regression
- Overfitting Control
- Model Selection

### Extra 09 — Regression Matrix & SVD

Mathematical exploration of matrix-based Linear Regression.

- Matrix Form
- Prediction Vector
- Residual Vector
- RSS
- Normal Equation
- Singular Matrices
- Pseudoinverse
- SVD
- Singular Values
- Eigenvalues
- SVD-Based Regression
- SAE
- MAE
- SSE
- MSE
- RMSE
- Worked Numerical Examples

### Extra 10 — Tree Traversal

- Tree Terminology
- Preorder
- Inorder
- Postorder
- BFS
- DFS
- Queue
- Stack
- Tree Search
- Graph Search
- Visited Sets
- Binary Search Trees
- Traversal Complexity

### Extra 11 — Search Complexity

- Big-O
- Linear Search
- Binary Search
- BFS Complexity
- DFS Complexity
- Uniform-Cost Search
- Greedy Best-First Search
- A*
- Tree Traversal Complexity
- Branching Factor
- Solution Depth
- Maximum Search Depth
- Time Complexity
- Space Complexity
- Admissible Heuristics
- Consistent Heuristics
- Manhattan Distance

## Datasets

### Student Performance

`datasets/student_performance.csv`

Used for regression and classification experiments.

Features:

- Hours
- Attendance
- Assignments
- Score

### Customers

`datasets/customers.csv`

Used for K-Means clustering and customer segmentation.

Features:

- Age
- Income
- Spending Score

### Spam Messages

`datasets/spam_messages.csv`

Used for text classification and spam detection.

Features:

- Message
- Label

## Learning Approach

The repository follows:

```text
Concept
   ↓
Understand Mathematics
   ↓
Work Through Examples
   ↓
Implement From Scratch
   ↓
Experiment With Data
   ↓
Evaluate
   ↓
Revise
```

Where practical, algorithms are implemented from scratch to understand their internal working before using high-level library implementations.

## Technologies

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Jupyter Notebook

## Source & Academic Attribution

The primary academic source used for the course-derived concepts in this repository is the lecture material:

**Dr. Abhinav Kumar**  
Assistant Professor  
Department of Computer Science and Engineering  
Motilal Nehru National Institute of Technology Allahabad

**Lecture Material:** *Introduction to Machine Learning / Introduction to AI-ML*

### Source Mapping

The course lecture deck was used as the primary reference for the following areas:

- ML fundamentals and learning paradigms
- Classification and regression
- Naive Bayes
- Classification metrics
- Linear Regression
- Matrix-based regression
- SVD and pseudoinverse
- MAE, MSE and related regression metrics
- Overfitting and underfitting
- Bias and variance
- Regularization
- K-Means
- Artificial Intelligence
- Artificial Agents
- Trees and Graphs
- BFS and DFS
- Heuristic Search
- Best-First Search
- A*
- 8-Puzzle
- AI-related assignments

### Important Attribution Note

The original lecture material belongs to its respective author and institution.

This repository contains:

- Rewritten notes
- Reorganized explanations
- Independently written Python implementations
- Mathematical explanations for learning
- Experiments
- Projects
- Supplementary material

Some implementation details, experiments, mathematical extensions, and supplementary topics go beyond the exact content of the lecture slides and have been added independently for learning purposes.

The original lecture slides are **not presented as my own work**.

This repository is **not an official MNNIT repository** and should not be interpreted as official MNNIT teaching material.

## Citation

The AI/ML concepts and course structure in this repository are based on academic lecture material from the:

**Department of Computer Science and Engineering**  
**Motilal Nehru National Institute of Technology Allahabad (MNNIT Allahabad)**

**Reference:** *Introduction to Machine Learning / Introduction to AI-ML*. Course lecture slides, Department of Computer Science and Engineering, Motilal Nehru National Institute of Technology Allahabad.

The lecture material is acknowledged as the academic source for the course-derived concepts covered in this repository.

If referencing the notes, code, experiments, or projects in this repository, please cite the repository separately as a personal learning and implementation resource.

## Academic Integrity

Attribution is provided wherever the repository is based on course material from the Department of Computer Science and Engineering, MNNIT Allahabad.

The notes have been intentionally **rewritten, reorganized, and expanded in my own words** rather than reproducing the lecture slides verbatim. The implementations, experiments, projects, and code are independently written as learning exercises.

The purpose of this repository is to document my own understanding and practical work while acknowledging the academic source material that provided the foundation for the course-derived concepts.

## Objectives

- Build strong fundamentals in AI and Machine Learning
- Understand the mathematics behind ML algorithms
- Implement algorithms in Python
- Practice implementing algorithms from scratch
- Work with real and synthetic datasets
- Learn model evaluation
- Understand overfitting and underfitting
- Understand bias, variance and regularization
- Develop intuition through experiments
- Understand fundamental AI search algorithms
- Build a structured first-year AI/ML knowledge base
- Prepare a foundation for advanced AI, ML and Data Science

## Repository Status

**First-year AI & Machine Learning coursework and supplementary learning material organized and completed.**

The repository may continue to evolve with improved implementations, experiments, projects, and additional learning material.

## Acknowledgements

Special acknowledgement to the:

**Department of Computer Science and Engineering**  
**Motilal Nehru National Institute of Technology Allahabad (MNNIT Allahabad)**

for providing the academic course material that served as the primary reference for the course-derived concepts documented in this repository.

**Institution:** Motilal Nehru National Institute of Technology Allahabad (MNNIT Allahabad)

## License

This project is licensed under the **MIT License**.

The MIT License applies to my original repository content, including my notes, source code, experiments, projects, and organization.

Third-party materials and course materials referenced or acknowledged in this repository remain subject to their respective ownership and licensing terms.
This project is licensed under the **MIT License**.

The MIT License applies to my original repository content, including my notes, source code, experiments, projects, and organization.

Third-party materials and course materials referenced or acknowledged in this repository remain subject to their respective ownership and licensing terms.
