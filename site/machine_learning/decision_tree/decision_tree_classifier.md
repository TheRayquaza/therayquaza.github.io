# Decision tree for classification

## Naive Implementation

Here is a naive implementation:
- Find the best split by selecting the best feature $k$ and the best split value $v$
- Split the data on feature k: values greater than v on feature k go to the right otherwise it goe to the left
- Build the tree recursively until max_depth is zero or the node is pure

```python
import numpy as np

class TreeNode:
    def __init__(
        self, X: np.array, y: np.array, impurity: float, feature: int, value: float, left=None, right=None
    ):
        self.X = X
        self.y = y
        self.impurity = impurity
        self.feature = feature
        self.value = value
        self.left = left
        self.right = right

    def select_value(self) -> float:
        if self.mode == "classification":
            return np.argmax(np.bincount(self.y))
        else:
            return np.mean(self.y)

    @property
    def is_terminal(self) -> bool:
        return self.left == self.right == None

class DecisionTreeClassifier():
    def __init__(
        self,
        max_depth=None,
        method="gini",
    ):
        self.root = None
        self.max_depth = max_depth
        self.method = method

        if not method in ["gini", "entropy"]:
            raise ValueError("Invalid method " + method)

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
        best_impurity_reduction = None

        impurity = self._calculate_impurity(y)

        for feature in range(X.shape[1]):
            for v in np.unique(X[:, feature]):
                left_indices = X[:, feature] <= v
                right_indices = ~left_indices
                y_left, y_right = y[left_indices], y[right_indices]

                impurity_left = self._calculate_impurity(y_left)
                impurity_right = self._calculate_impurity(y_right)

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

## Classic Classification: Moon

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

## Classic Classification: Iris Dataset

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