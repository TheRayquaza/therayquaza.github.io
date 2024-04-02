# K-nearest neighbors

![Classification](https://raw.githubusercontent.com/TheRayquaza/therayquaza.github.io/main/images/badges/Classification.svg)
![Regression](https://raw.githubusercontent.com/TheRayquaza/therayquaza.github.io/main/images/badges/Regression.svg)

## What is KNN?

K-nearest neighbors (KNN) is a non-parametric supervised learning technique created in 1951. It's based on the nearest neighbors algorithm, which is a special case of KNN ($k = 1$). KNN is often considered the simplest machine learning technique.

The goal of KNN is to memorize the training dataset and classify a new sample based on proximity. This is achieved by evaluating the relationship between memorized samples and the target sample with different types of distances, usually the Euclidean distance.

KNN is primarily a classification method but can also be adapted for regression. For regression tasks, the average target features of the k neighbors are used to predict the target feature of the sample.

```{figure} https://raw.githubusercontent.com/TheRayquaza/therayquaza.github.io/main/images/machine_learning/knn/KNN.png
illustration from [wikipedia](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)
```

## Assumption

KNN assumes that similar data points tend to belong to the same class.

KNN should be used in some specific circumstances:
- Small dataset
- Well distributed dataset
- Low dimensional data
- No noise in data

## How does KNN work for classification?

Let's denote n as the number of classified samples and k as the number of neighbors selected ($k < n$). Let x be the target sample to classify.

The KNN classifier follows a straightforward algorithm:
- Select the first k closest samples to x.
- Determine the most common class among these k samples.

There are multiple implementation of the KNN classifier. One of them consists of just calculating all distances and then extracting the k closest samples, this is: Brute-Force. They are other implementation such as **K-d tree** or **Ball tree** that increase the performance of the model.

## Selecting k

The choice of k depends on the dataset:

A larger value of k tends to pay more attention to noisy data, resulting in a less clear decision boundary.
A smaller value of k increases the risk of overfitting.
In general, a larger k value leads to underfitting (high bias, low variance), while a smaller k value leads to overfitting (low bias, high variance).

## Limitation of KNN

### Distribution issue

While running a little experiment, I ran into the most important issue of KNN: we need a great distribution in our training set.
Look at the following code and its resulting prediction:

```python
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt
import numpy as np

# x = 10y + 1.2
X = np.linspace(0, 10, num=50)
y = np.linspace(0, 100, num=50) + 1.2

X_train, X_test = X[:40], X[40:]
y_train, y_test = y[:40], y[40:]

model = KNeighborsRegressor(n_neighbors=2, n_jobs=-1)
model.fit(X_train.reshape(-1, 1), y_train)
y_pred = model.predict(X_test.reshape(-1, 1))

plt.xlabel("x")
plt.ylabel("y")
plt.plot(X_train, y_train)
plt.scatter(X_test, y_test)
plt.scatter(X_test, y_pred)
```
```{figure} https://raw.githubusercontent.com/TheRayquaza/therayquaza.github.io/main/images/machine_learning/knn/KNN_distribution_issue.png
KNN limitation
```

We see that predictions have gone really bad. It seems that all predictions are stuck near the last point.
KNN is really bad at predicting values it has never seen. This makes it bad if the dataset is not enough generic and balanced.
KNN is also very sensitive to the scale of the dataset.

### Memory issue

All the dataset needs to be loaded into memory which makes it unusable for larger dataset.

### Speed issue

KNN is not a fast machine learning. Some implementation using pre-computed data structure to organize data in a more usable way can be used to reduce computation but it remains slow during prediction.

### Curse of dimensionality
The curse of dimensionality in KNN refers to the inefficiency of the Euclidean distance metric in high-dimensional data spaces. As the number of dimensions increases, the distance between points becomes less meaningful.

To solve this issue, various techniques can be used, including:
- Feature extraction
- Dimension reduction

## K-nearest neighbors for classification

### Implementation

This is a really simple implementation of the KNN classifier using the naive approach from my github repo.
```python
import numpy as np

class KNeighborsClassifier():
    def __init__(self, k=5, distance_method=np.linalg.norm):
        self.k = k
        self.distance_method = distance_method

    def _make_prediction(self, X: np.array):
        distances = [self.distance_method(x, X) for x in self.X]
        sorted_indices = np.argsort(distances)[:self.k]
        best_classes = self.y[sorted_indices]
        return np.argmax(np.bincount(best_classes))

    def predict(self, X: np.array) -> np.array:
        result = np.zeros(X.shape[0], dtype=int)
        for i in range(X.shape[0]):
            result[i] = self._make_prediction(X[i])
        return result
```
https://github.com/TheRayquaza/ml_lib/blob/main/src/knn/knn_classifier.py

### Experimenting KNN on MNIST

I am using the classic MNIST dataset to illustrate KNN performance and k tuning.
For this task, I am using scikit-learn (https://scikit-learn.org/stable/) which provides metrics, MNIST dataset and of course an optimized KNN model for classification.

First, we need to import the necessary libraries:
```python
import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy
from sklearn.neighbors import KNeighborsClassifier
```

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

We can now simply train the model with $k = 3$ for example and make prediction:

```python
knnclassifier = KNeighborsClassifier(n_neighbors=3, n_jobs=-1)
knnclassifier.fit(X_train, y_train)
y_pred = knnclassifier.predict(X_test)
```

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

## K-nearest neighbors for regression

KNN classifier can be adapted for regression task. It performs just as its classifier version but instead of using the voting principle, we calculate the mean on each features. 

### Implementation

This is a really simple implementation of the KNN regressor using the naive approach from my github repo.

```python
import numpy as np

class KNeighborsRegressor:
    def __init__(self, k=5, distance_method=np.linalg.norm):
        self.k = k
        self.distance_method = distance_method

    def _make_prediction(self, X: np.array):
        distances = [self.distance_method(x, X) for x in self.X]
        sorted_indices = np.argsort(distances)[:self.k]
        best_features = self.y[sorted_indices]
        return np.mean(best_features)

    def predict(self, X: np.array):        
        samples = X.shape[0]
        return np.array([self._make_prediction(X[i] for i in range(samples))])
```
https://github.com/TheRayquaza/ml_lib/blob/main/src/knn/knn_regressor.py

### Experimenting KNN regressor on linear data

This little piece of code tries to learn the linear relationship $y = 10x + 1.2$.

```python
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt
import numpy as np

X_train = np.linspace(0, 10, num=20)
y_train = np.linspace(0, 100, num=20) + 1.2

X_test = np.random.uniform(low=0, high=10, size=(10,))
y_test = X_test * 10 + 1.2

model = KNeighborsRegressor(n_neighbors=2, n_jobs=-1)
model.fit(X_train.reshape(-1, 1), y_train)
y_pred = model.predict(X_test.reshape(-1, 1))

plt.xlabel("x")
plt.ylabel("y")
plt.plot(X_train, y_train)
plt.scatter(X_test, y_test)
plt.scatter(X_test, y_pred)
```
```{figure} https://raw.githubusercontent.com/TheRayquaza/therayquaza.github.io/main/images/machine_learning/knn/KNN_regressor_linear.png
$y = 10x + 1.2$
```

## References
1. Complexity: [Towards Data Science](https://towardsdatascience.com/k-nearest-neighbors-computational-complexity-502d2c440d5)
2. KDTree: [Univeristy of Utah](https://users.cs.utah.edu/~lifeifei/cis5930/kdtree.pdf)
3. Usage case: [Towards Data Science](https://towardsdatascience.com/knn-algorithm-what-when-why-how-41405c16c36f)