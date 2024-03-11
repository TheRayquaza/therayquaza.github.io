# K-nearest neighbors
## What is KNN?

K-nearest neighbors (KNN) is a non-parametric supervised learning technique created in 1951. It's based on the nearest neighbors algorithm, which is a special case of KNN ($k = 1$). KNN is often considered the simplest machine learning technique.

The goal of KNN is to memorize the training dataset and classify a new sample based on proximity. This is achieved by evaluating the relationship between memorized samples and the target sample with different types of distances, usually the Euclidean distance.

KNN is primarily a classification method but can also be adapted for regression. For regression tasks, the average target features of the k neighbors are used to predict the target feature of the sample.

```{figure} https://raw.githubusercontent.com/TheRayquaza/therayquaza.github.io/main/images/machine_learning/knn/KNN.png
illustration from [wikipedia](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)
```

## How does KNN work for classification?

Let's denote n as the number of classified samples and k as the number of neighbors selected ($k < n$). Let x be the target sample to classify.

The KNN classifier follows a straightforward algorithm:
- Select the first k closest samples to x.
- Determine the most common class among these k samples.

There are multiple implementation of the KNN classifier. One of them consists of just computing all distances and then extracting the k closest sample, this is: Brute-Force.

## KNN Implementation

I have listed some implementation of the KNN model with their respective complexity:

Let's denote: d as the number of dimension, n the number of sample and k the neighbor hyperparameter.
- Brute-Force: naive implementation of KNN
    * Training time / space complexity: $O(1)$ (No training for this method)
    * Prediction time complexity: $O(k * \log(n))$
    * Prediction space complexity: $O(1)$
- K-d tree: construction of a k-d tree during training to simplify searching. It is similar to BST but it also supports multi-dimensional data.
    * Training time complexity: $O(d * n * \log(n))$
    * Training space complexity: $O(d * n)$
    * Prediction time complexity: $O(k * \log(n))$
    * Prediction space complexity: $O(1)$
- Ball tree

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
![KNN limitation](https://raw.githubusercontent.com/TheRayquaza/therayquaza.github.io/main/images/machine_learning/knn/KNN_distribution_issue.png)

We see that predictions have gone really bad. It seems that all predictions are stuck near the last point.
KNN is really bad at predicting values it has never seen. This makes it bad if the dataset is not enough generic and balanced.
KNN is also very sensitive to the scale of the dataset.

### Memory issue

All the dataset needs to be loaded into memory which makes it unusable for larger dataset.

### Speed issue

KNN is not a fast machine learning. Some implementation using pre-computed data structure to organize data in a more usable way can be used to reduce computation but it reamins slow during prediction.

### Curse of dimensionality
The curse of dimensionality in KNN refers to the inefficiency of the Euclidean distance metric in high-dimensional data spaces. As the number of dimensions increases, the distance between points becomes less meaningful.

To solve this issue, various techniques can be used, including:
- Feature extraction
- Dimension reduction

### Interpretability issue

KNN uses the training set, which means it cannot help to understand hidden relationships within the training set.
Thus, it is not interpretable, which may be an issue in some real-world problems requiring decision explanation.

## When to use KNN ?

KNN performs bad in general case for real world problem.
However, KNN can be used in some specific circumstances:
- Small dataset
- Well distributed dataset
- Low dimensional data
- No noise in data

## References
1. https://towardsdatascience.com/k-nearest-neighbors-computational-complexity-502d2c440d5
2. https://users.cs.utah.edu/~lifeifei/cis5930/kdtree.pdf
3. https://towardsdatascience.com/knn-algorithm-what-when-why-how-41405c16c36f
