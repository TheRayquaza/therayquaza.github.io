# KNN for regression

KNN classifier can be adapted for regression task. It performs just as its classifier version but instead of using the voting principle, we use the mean feature. 

## Naive implementation

This is a really simple implementation of the KNN regressor using the naive approach from my github repo.
```python
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count

class KNeighborsRegressor(Model):
    def __init__(self, k=5, distance_method=np.linalg.norm, n_jobs=None):
        if k < 0:
            raise ValueError("KNeighborsRegressor: k should be greater than 0")
        self.k = k
        self.distance_method = distance_method
        self.n_jobs = cpu_count() if n_jobs == -1 else n_jobs

    def fit(self, X: np.array, y: np.array):
        pass

    def _make_prediction(self, X: np.array):
        distances = [self.distance_method(x, X) for x in self.X]
        sorted_indices = np.argsort(distances)[:self.k]
        best_features = self.y[sorted_indices]
        return np.mean(best_features)

    def predict(self, X: np.array):        
        samples = X.shape[0]
        result = np.zeros((samples))
        
        if not self.n_jobs:
            for i in range(samples):
                result[i] = self._make_prediction(X[i])
        else:
            pool = ThreadPoolExecutor(max_workers=self.n_jobs)
            future_to_pred = {
                pool.submit(self._make_prediction, X[i]): i for i in range(samples)
            }
            for future in as_completed(future_to_pred):
                result[future_to_pred[future]] = future.result()
        return result
```
https://github.com/TheRayquaza/ml_lib/blob/main/src/knn/knn_regressor.py

## Experimenting KNN regressor on linear data

This little piece of code tries to learn the linear relationship $y = 10x + 1.2$.

```python
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt
import numpy as np

X_train = np.linspace(0, 10, num=20)
y_train = np.linspace(0, 100, num=20) + 1.2

# linear function defined as: y = 10x + 1.2

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
