# Decision tree for regression

Decision tree can be adapted for regression task.
Instead of minimzing the gini or entropy impurity adapted for classifcation, we use any regression metric such as [MSE](https://therayquaza.github.io/machine_learning/metrics_and_losses/metrics_and_losses.html#mse).

## Naive Implementation

Here a naive implementation rom my github repo:

```python
import numpy as np

class TreeNode:
    def __init__(
        self, X, y, feature, value, left=None, right=None
    ):
        self.X = X
        self.y = y
        self.feature = feature
        self.value = value
        self.left = left
        self.right = right

    @property
    def is_terminal(self):
        return self.left is None and self.right is None

class DecisionTreeRegressor:
    def __init__(self, max_depth=None):
        self.root = None
        self.max_depth = max_depth

    def _build_tree(self, X, y, depth=None):
        if depth == 0 or len(np.unique(y)) <= 1:
            return TreeNode(X, y, 0, 0)

        feature, value, impurity_reduction = self._find_best_split(X, y)
        root = TreeNode(X, y, feature, value)
        X_left, y_left = X[X[:, feature] <= value], y[X[:, feature] <= value]
        X_right, y_right = X[X[:, feature] > value], y[X[:, feature] > value]

        root.left = self._build_tree(X_left, y_left, depth - 1 if depth else depth)
        root.right = self._build_tree(X_right, y_right, depth - 1 if depth else depth)
        return root

    def _find_best_split(self, X, y):
        best_feature = None
        best_split_value = None
        best_impurity_reduction = None

        impurity = np.mean(np.square(y - np.mean(y)))

        for feature in range(X.shape[1]):
            for v in np.unique(X[:, feature]):
                left_indices = X[:, feature] <= v
                right_indices = ~left_indices
                y_left, y_right = y[left_indices], y[right_indices]

                impurity_left = np.mean(np.square(y_left - np.mean(y_left)))
                impurity_right = np.mean(np.square(y_right - np.mean(y_right)))

                weighted_impurity = (len(y_left) / len(y)) * impurity_left + (
                    len(y_right) / len(y)
                ) * impurity_right
                impurity_reduction = impurity - weighted_impurity

                if (
                    best_impurity_reduction is None
                    or impurity_reduction > best_impurity_reduction
                ):
                    best_impurity_reduction = impurity_reduction
                    best_feature = feature
                    best_split_value = v
        return best_feature, best_split_value, best_impurity_reduction

    def _make_prediction(self, X):
        current = self.root
        while not current.is_terminal:
            current = current.right if X[current.feature] > current.value else current.left
        return np.mean(current.y)

    def fit(self, X, y):
        self.root = self._build_tree(X, y, self.max_depth)
        return self

    def predict(self, X):
        return np.array([self._make_prediction(X[i]) for i in range(X.shape[0])])
```
https://github.com/TheRayquaza/ml_lib/blob/main/src/tree/decision_tree_regressor.py


## Experimenting Decision Tree Regressor on Linear dataset

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