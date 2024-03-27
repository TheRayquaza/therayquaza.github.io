# KNN for classification

![Classification](https://raw.githubusercontent.com/TheRayquaza/therayquaza.github.io/main/images/badges/Classification.svg)

## Naive implementation

This is a really simple implementation of the KNN classifier using the naive approach from my github repo.
```python
class KNeighborsClassifier():
    def __init__(self, k=5, distance_method=np.linalg.norm, n_jobs=-1):
        if k <= 0:
            raise ValueError("KNeighborsClassifier: k should be greater than 0")
        self.k = k
        self.distance_method = distance_method
        self.n_jobs = cpu_count() if n_jobs == -1 else n_jobs
    
    def fit(X, y):
        pass

    def _make_prediction(self, X: np.array):
        distances = [self.distance_method(x, X) for x in self.X]
        sorted_indices = np.argsort(distances)[:self.k]
        best_classes = self.y[sorted_indices]
        return np.argmax(np.bincount(best_classes))

    def predict(self, X: np.array) -> np.array:
        samples = X.shape[0]
        result = np.zeros(samples, dtype=int)
        if not self.n_jobs:
            for i in range(samples):
                result[i] = self._make_prediction(X[i])
        else:
            with ThreadPoolExecutor(max_workers=self.n_jobs) as pool:
                future_to_pred = {pool.submit(self._make_prediction, X[i]): i for i in range(samples)}
                for future in as_completed(future_to_pred):
                    result[future_to_pred[future]] = future.result()
        return result
```
https://github.com/TheRayquaza/ml_lib/blob/main/src/knn/knn_classifier.py

```{note}
Due to performance issue (because of the naive approach), I added the possibility to use multi threading. Specifying the number of jobs will enable threads.
```

## Experimenting KNN on MNIST

I am using the classic MNIST dataset to illustrate KNN performance and k tuning.
For this task, I am using scikit-learn (https://scikit-learn.org/stable/) which provides metrics, MNIST dataset and of course an optimized KNN model for classification.

### Imports

First, we need to import the necessary libraries:
```python
import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy
from sklearn.neighbors import KNeighborsClassifier

from concurrent.futures import ThreadPoolExecutor, as_completed
```

### Preprocessing

Then, we need to apply some preprocessing to our datasets by using 127 as a threshold ($pixels \in [0, 255]$):
```python
X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False, parser="pandas")
X = X > 127
y = y.astype(np.uint8)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

Here is an example of a sample with this preprocessing:
```python
plt.imshow(X[0].reshape((28, 28)), cmap="binary")
```
```{figure} https://raw.githubusercontent.com/TheRayquaza/therayquaza.github.io/main/images/machine_learning/knn/5_mnist.png
5 preprocessed in MNIST
```

### Training

We can now simply train the model with $k = 3$ for example and make prediction:
```python
knnclassifier = KNeighborsClassifier(n_neighbors=3, n_jobs=-1)
knnclassifier.fit(X_train, y_train)
y_pred = knnclassifier.predict(X_test)
```

### Metrics

We can get the confusion matrix as well as the metrics for each class:
```python
confusion_matrix(y_test, y_pred)
```
```
[[1329    1    5    0    0    2    3    2    1    0]
 [   0 1592    1    1    0    1    0    4    0    1]
 [  13   31 1304    2    2    1    4   19    2    2]
 [   2    9   15 1366    0   14    2   10    7    8]
 [   1   15    2    0 1240    1    3    3    1   29]
 [   6   12    0   22    4 1212   13    1    1    2]
 [   7    2    1    0    5    4 1377    0    0    0]
 [   1   26    4    0    1    0    0 1459    0   12]
 [   4   28    7   29    7   37    5   15 1212   13]
 [   8   10    1   10   21    3    0   21    0 1346]]
```

With the confusion matrix, we can calculate metrics for each class:
```python
classification_report(y_test, y_pred)
```
```
    precision    recall  f1-score   support

0       0.97      0.99      0.98      1343
1       0.92      0.99      0.96      1600
2       0.97      0.94      0.96      1380
3       0.96      0.95      0.95      1433
4       0.97      0.96      0.96      1295
5       0.95      0.95      0.95      1273
6       0.98      0.99      0.98      1396
7       0.95      0.97      0.96      1503
8       0.99      0.89      0.94      1357
9       0.95      0.95      0.95      1420
```
