# Decision Tree

```{tableofcontents}
```

## What is a decision tree ?

Decision trees are a machine learning technique that utilizes a tree data structure to partition data based on their dimensions. Decision trees are considered explainable models, meaning we can comprehend the model's output simply by examining its rules.

```{figure} https://raw.githubusercontent.com/TheRayquaza/therayquaza.github.io/main/images/machine_learning/decision_tree/Explainability-vs-Accuracy.png
Explainability vs Accuracy (taken from [medium](https://medium.com/@joachimvanneste1/explainable-ai-adapting-lime-for-video-model-interpretation-74c85502b0d0))
```

The downside of high explainability is its impact on performance. Typically, performance is compromised when the model is overly explainable.

## How works a decision tree ?

The training of a decision tree involves splitting features using the CART (Classification and Regression Trees) cost function. This function can be adjusted according to different contexts and impurity measures.

Let's denote $I$ as the impurity metric we aim to minimize. For classification tasks, this could be Gini impurity or entropy, while for regression tasks, it is [MSE](https://therayquaza.github.io/machine_learning/metrics_and_losses/metrics_and_losses.html#mse). Let $k$ represent the feature to select, $t_k$ the threshold value for data splitting, and $m$ the number of samples.

The cost function for the binary decision tree we seek to minimize is as follows:

$$ J(k, t_k) = \frac{m_{left}}{m} I_{left} + \frac{m_{right}}{m} I_{right} $$

The $left$ and $right$ indices denote the threshold splitting: $left$ represents values lower than the threshold $t_k$ on its $k$ feature.

Decision trees can also use more general data structures such as a general tree to split data into more than two parts. For a general tree with $n$ children, the formula becomes:

$$ J(k, t_k) = \sum_{i=1}^{n} \frac{m_i}{m} I_i $$

## Assumption

- Assumes that the data can be split into homogeneous regions based on the values of the independent variables
- Assumes that the relationships between the independent variables and the dependent variable are non-linear

## Why is it an explainable model ?

By design, inner rules from decision trees can be extracted.
In scikit learn we can directly extract rules from a decision tree with graphviz.

```{figure} https://raw.githubusercontent.com/TheRayquaza/therayquaza.github.io/main/images/machine_learning/decision_tree/DT_rules.png
example of rules for the [wine dataset](https://archive.ics.uci.edu/dataset/109/wine)
```
In this tree, leaf nodes are called "pure" because they hold only one class and so their gini impurity become 0.

Aditionnaly, decision trees are considered explainable since the prediction process involves selecting the path by looking at the specific feature values of the input data and traversing the decision tree structure based on these values. This allows a clear understanding of how each decision contributes to the final prediction outcome.

## Overfitting with decision tree

One problem with decision tree is they tend to overfit the training set when no pruning is given.
Pruning is the action of reducing some part of the tree by removing part that gives little to no information on the classification.
Pruning reduces the complexity of the final classifier and hence improves predictive accuracy by the reduction of overfitting.

## Model complexity

Decision trees perform really well when it comes to predict the class. Since the model is pre-built during training, prediction will retrieve from the inner tree structure, thus: $O(\log_2(k)) = O(\log(k))$ for binary tree implementation where k is the number of selected nodes (decision is considered to be constant).

Decision trees are however slow to train. Training the decision tree requires to select the best split among the training observation and best features $n$. Finally, it builds its structure recursively, thus: $O(n \cdot k \cdot log(k))$

## References
1. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow by Aurélien Géron