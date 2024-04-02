# Decision Tree

![Classification](https://raw.githubusercontent.com/TheRayquaza/therayquaza.github.io/main/images/badges/Classification.svg)
![Regression](https://raw.githubusercontent.com/TheRayquaza/therayquaza.github.io/main/images/badges/Regression.svg)

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

## Decision tree for classification

### Implementation

Here is a naive implementation following instructions:
- Find the best split by selecting the best feature $k$ and the best split value $v$
- Split the data on feature k: values greater than v on feature k go to the right otherwise it goe to the left
- Build the tree recursively until max_depth is zero or the node is pure

```python
import numpy as np

class TreeNode:
    def __init__(self, X: np.array, y: np.array, impurity: float, feature: int, value: float, left=None, right=None):
        self.X = X
        self.y = y
        self.impurity = impurity
        self.feature = feature
        self.value = value
        self.left = left
        self.right = right

    def select_value(self) -> float:
        return np.argmax(np.bincount(self.y))

    @property
    def is_terminal(self) -> bool:
        return self.left == self.right == None

class DecisionTreeClassifier:
    def __init__(self, max_depth=None, method="gini"):
        self.root = None
        self.max_depth = max_depth
        self.method = method

    def _calculate_impurity(self, y: np.array) -> float:
        if self.method == "gini":
             _, counts = np.unique(y, return_counts=True)
            probs = counts / len(y)
            return 1 - np.sum(probs**2)
        else:
            _, counts = np.unique(y, return_counts=True)
            probs = counts / len(y)
            return -np.sum(probs * np.log2(probs))

    def _build_tree(self, X: np.array, y: np.array, depth=None):
        samples, features = X.shape
        if depth == 0 or len(np.unique(y)) <= 1:
            return TreeNode(X, y, 0, 0, 0)

        feature, value, impurity_reduction = self._find_best_split(X, y)
        root = TreeNode(X, y, impurity_reduction, feature, value)
        X_left, y_left = X[X[:, feature] <= value], y[X[:, feature] <= value]
        X_right, y_right = X[X[:, feature] > value], y[X[:, feature] > value]
        root.left = self._build_tree(X_left, y_left, depth - 1 if depth else depth)
        root.right = self._build_tree(X_right, y_right, depth - 1 if depth else depth)
        return root

    def _find_best_split(self, X: np.array, y: np.array):
        best_feature = None
        best_split_value = None
        best_impurity_reduction = float('-inf')

        impurity = self._calculate_impurity(y)

        for feature in range(X.shape[1]):
            for v in np.unique(X[:, feature]):
                left_indices = X[:, feature] <= v
                y_left, y_right = y[left_indices], y[~left_indices]

                impurity_left = self._calculate_impurity(y_left)
                impurity_right = self._calculate_impurity(y_right)

                weighted_impurity = (len(y_left) / len(y)) * impurity_left + (len(y_right) / len(y)) * impurity_right
                impurity_reduction = impurity - weighted_impurity

                if impurity_reduction > best_impurity_reduction:
                    best_impurity_reduction = impurity_reduction
                    best_feature = feature
                    best_split_value = v
        return best_feature, best_split_value, best_impurity_reduction

    def fit(self, X: np.array, y: np.array):
        self.root = self._build_tree(X, y, self.max_depth)
        return self

    def _make_prediction(self, X: np.array) -> float:
        current = self.root
        while not current.is_terminal:
            current = current.right if X[current.feature] > current.value else current.left
        return current.select_value()

    def predict(self, X: np.array) -> np.array:
        return np.array([self._make_prediction(X[i]) for i in range(X.shape[0])])
```
https://github.com/TheRayquaza/ml_lib/blob/main/src/tree/decision_tree_classifier.py

### Moons

If solving linear relationship using any linear model is straightforward, for the moon dataset we need a more accurate model.
Decision tree can help solving non linear relation such as the classic moon dataset.

I trained and plot the decision boundary of the model:

```{figure} https://raw.githubusercontent.com/TheRayquaza/therayquaza.github.io/main/images/machine_learning/decision_tree/Decision_Boundary_Moons.png
Decision Boundary of a Non-Pruned Decision Tree (0.9 Accuracy)
```

The specifity of the decision tree's decision boundary lies in its shape. As the space data is splitted among its dimensions with the chosen value maximizing the information gain, we get a split a split in 1 dimension (vertical and horizontal separation for 2D observation). The more precise the model is, the more detailled are those boundaries.

For simpler decision boundaries (maximum 1 or 2 decisions), we would split the dataset with less detailled boundaries (few vertical / horizontal split);

```{figure} https://raw.githubusercontent.com/TheRayquaza/therayquaza.github.io/main/images/machine_learning/decision_tree/Decision_Boundary_Moons_worse.png
Decision Boundary of a Pruned Decision Tree (0.8 < Accuracy)
```

### Iris Dataset

I trained the model above with the classic iris dataset:

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
data = load_iris()

X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2)

model = DecisionTreeClassifier().fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f"Accuracy {np.mean(y_test == y_pred)}")

>>> Accuracy 0.9
```

## Decision tree for regression

Decision tree can be adapted for regression task.
Instead of minimzing the gini or entropy impurity adapted for classifcation, we use any regression metric such as [MSE](https://therayquaza.github.io/machine_learning/metrics_and_losses/metrics_and_losses.html#mse).

### Implementation

Decision Tree for regression is almost the same as the model for classification. The only difference lies in the calculation of impurities and the value selection because we are not working with classes anymore.

```python
import numpy as np

class TreeNode:
    ...

    def select_value(self) -> float:
        return np.mean(self.y)

    ...

class DecisionTreeRegressor:

    ...

    def _find_best_split(self, X, y):
        best_feature = None
        best_split_value = None
        best_impurity_reduction = None

        impurity = np.mean(np.square(y - np.mean(y)))

        for feature in range(X.shape[1]):
            for v in np.unique(X[:, feature]):
                ...

                impurity_left = np.mean(np.square(y_left - np.mean(y_left)))
                impurity_right = np.mean(np.square(y_right - np.mean(y_right)))

                ...

    ...
```
https://github.com/TheRayquaza/ml_lib/blob/main/src/tree/decision_tree_regressor.py


### Experimenting Decision Tree Regressor on Linear dataset

Detecting a linear realtionship with decision tree is straightforward:

```python
from sklearn.model_selection import train_test_split

X = np.linspace(1, 100)
y = 2.5 * X + 10

X_train, X_test, y_train, y_test = train_test_split(X.reshape(-1, 1), y, test_size=0.2)

model = DecisionTreeRegressor().fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f"MSE = {np.mean(np.square(y_test - y_pred))}")

>> MSE = 33.16664931278639
```

## References
1. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow by Aurélien Géron