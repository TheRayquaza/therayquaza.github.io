# DBSCAN
![Clustering](https://raw.githubusercontent.com/TheRayquaza/therayquaza.github.io/main/images/badges/Clustering.svg)

DBSCAN (Density-based spatial clustering of applications with noise).
DBSCAN algorithm is a density based algorithm. It is based on identifying high density regions and detecting outliers in low density regions. 
DBSCAN is adapted to identify cluster of different size, it can handle varying densities.

## How works DBSCAN ?

DBSCAN has three hyperameters:
- $\epsilon$: the minimum distance to consider that two obsverations are next to each other
- $n$: the number of samples next to an obsveration required to consider it as a **core-point**
- $d$: the distance function (usually the euclidean distance $||x - y||_2$)

DBSCAN algorithm has two different type of points: **core points** & **non-core points**.
Core-points are all points having more than $n$ neighbors with distance lower than $\epsilon$.

The algorithm of DBSCAN is the following (from [wikipedia](https://en.wikipedia.org/wiki/DBSCAN#Abstract_algorithm)):
1. Find all core points (formally $ \{ |\{d(x, y) < \epsilon, y \in X\}| \geq n, x \in X\} $)
2. Select a random core point and assign it a new cluster $i$
    * Spread *cluster assignment* for each core point neighbors and assign them cluster $i$
    * Assign cluster $i$ to non-core point neighbors without spreading the cluster
    * If all core points have been assigned, stop otherwise repeat step 2

```{note}
Non-core points without any cluster are called noise points (hence Density-based spatial clustering of applications **with noise**)
```

## Assumption

- Assumes that clusters can be identified using density
- Assumes that any point not associated with a high-density region is considered noise

## Implementation

Here is my iterative implementation of DBSCAN using a queue. The *espilon* is the minimal distance $\epsilon$ and *n* is the number of required neighbors.

```python
class DBSCAN:

    def __init__(self, epsilon=0.5, n=5, d=np.linalg.norm):
        self.epsilon = epsilon
        self.n = n
        self.d = d

    def fit(self, X):
        self.dist_matrix = self.d(X[:, None, :] - X[None, :, :], axis = -1)
        self.core_points_indices = np.argwhere(np.sum(self.dist_matrix < self.epsilon, axis=1) >= self.n + 1).flatten()
        self.clusters = np.full((X.shape[0],), -1)
        self.n_clusters = 0
        
        points_assigned = np.full((X.shape[0],), False)
        queue = []
        while not np.all(points_assigned[self.core_points_indices]):
            first_indice = np.argwhere(~points_assigned).flatten()[0]
            queue.append(first_indice)
            self.clusters[first_indice] = self.n_clusters
            points_assigned[first_indice] = True
            while queue:
                sample = queue.pop(0)
                neighbors_indices = np.argwhere(np.logical_and(self.dist_matrix[sample] != 0, self.dist_matrix[sample] < self.epsilon))[:, 1]
                self.clusters[neighbors_indices] = self.n_clusters
                for indice in np.intersect1d(neighbors_indices, self.core_points_indices):
                    if not points_assigned[indice]:
                        queue.append(indice)
                points_assigned[neighbors_indices] = True
            self.n_clusters += 1
        return self

    def fit_predict(self, X):
        self.fit(X)
        return self.clusters
```

Some line are really confusing as I really tried to use numpy for the sake of optimization. This code is modular, it allows the usage of a custom distance function (it needs to have an optionnal parameter axis to apply the distance on the last axis). X needs to be a matrix, any number of observations and features can be used in this code.

## Detecting outliers (noise) with DBSCAN

A lot of model are adapted for outlier detection. DBSCAN is one of them.

DBSCAN can be used as for anomaly detection, outliers will be observations detected as **noise**.

I used a really simple dataset from [kaggle](https://www.kaggle.com/datasets/krishnaraj30/weight-and-height-data-outlier-detection) with 2 features (easier for 2D plot) to illustrate how works outlier detection with DBSCAN. I am using scikit-learn as it is largely more optimized than my naive implementation (I could not even use it for 10000 observations ...)

```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

data = pd.read_csv("weight-height.csv").to_numpy()[:, 1:]

core_samples, labels = DBSCAN(eps=4, min_samples=2).fit_predict(data)

plt.figure(figsize=(8, 6))

plt.scatter(data[labels == -1, 0], data[labels == -1, 1], c='gray', label='Non-core points', s=50)

plt.scatter(data[core_samples, 0], data[core_samples, 1], c=labels[core_samples], cmap='viridis', s=50, label='Core points')

plt.title('DBSCAN Clustering')
plt.xlabel('Weight')
plt.ylabel('Height')
plt.legend()
plt.colorbar(label='Cluster Label')
plt.savefig('dbscan_weight_height.png')
plt.show()
```

```{figure} https://raw.githubusercontent.com/TheRayquaza/therayquaza.github.io/main/images/machine_learning/clustering/dbscan_weight_height.png
DBSCAN - detection of outliers on weight & height dataset ($n$ = 10 and $\epsilon = 3$)
```

With this method can easily detect all obsverations not belonging to any dense regions. If we change a little bit the hyperparameters, we get really different result. Here I decided to use a scaler because clustering (and in general model based on distance calculation) performs better with normalized data:

```{figure} https://raw.githubusercontent.com/TheRayquaza/therayquaza.github.io/main/images/machine_learning/clustering/dbscan_weight_height_normalized.png
DBSCAN - detection of outliers on weight & height dataset normalized ($n$ = 10 and $\epsilon = 0.1$)
```