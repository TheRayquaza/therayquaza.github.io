# K-Means
![Clustering](https://raw.githubusercontent.com/TheRayquaza/therayquaza.github.io/main/images/badges/Clustering.svg)

K-Means is centroid-based algorithm that creates $k$ clusters with $k$ given by the user. K-Means algorithm is guarenteed to converge to an optimal solution (it can converge to a local optimum depending on the initalization step).

## How works K-Means ?

K-Means is purely based on the [Lloyd’s algorithm](https://en.wikipedia.org/wiki/Lloyd%27s_algorithm).

K-Means algorithm is straightforward:
1. Initialize random centroids or select pre-defined great centroids.
2. Repeat step until centroids are not updated anymore:
    - Assign each observation to the nearest cluster
    - Calculate the new centroids of each cluster

Other implementation such as **K-Means++**, **Batch K-means** or **Spectral Clustering** are "adaptations" of K-Means.

## Assumption

- Assumes the right metric is used
- Assumes clusters have spherical-shaped
- Assumes features are continuous

## Implementation

```python
import numpy as np

class KMeans():
    def __init__(self, n_clusters, max_iter=100)
        self.n_clusters = n_clusters
        self.max_iter = max_iter

    def _assign_cluster(self, i):
        for c in range(self.n_clusters):
            self.distances[i, c] = np.linalg.norm(self.X[i] - self.centroids[c])
        return np.argmin(self.distances[i, :]), i

    def _compute_centroid(self, c):
        X_c = self.X[np.where(self.clusters == c)]
        return np.mean(X_c, axis=0) if len(indices) > 0 else self.centroids[c], c

    def fit(self, X: np.array):
        self.centroids = X[
            np.random.choice(np.arange(self.samples), size=self.n_clusters, replace=False)
        ]
        self.last_centroids = None
        self.distances = np.zeros((self.samples, self.n_clusters))
        self.clusters = np.zeros(self.samples)
        i = 0
        while (self.last_centroids is None or not np.all(self.last_centroids == self.centroids)) and (i < self.max_iter):
            self.last_centroids = np.copy(self.centroids)
            self.clusters = [self._assign_cluster(j)[1] for j in range(self.datapoints)]
            self.centroids = [self._compute_centroid(j)[0] for j in range(self.n_clusters)]
            i += 1
        return self

    def _find_cluster(self, X: np.array):
        best_cluster, min_dist = None, float("inf")
        for c in range(self.n_clusters):
            dist = np.linalg.norm(X - self.centroids[c])
            if best_cluster is None or min_dist > dist:
                best_cluster, min_dist = c, dist
        return best_cluster

    def predict(self, X: np.array):
        return np.array([self._find_cluster(X[i]) for i in range(X.shape[0])])
```
from my github repo https://github.com/TheRayquaza/ml_lib/blob/main/src/cluster/kmeans.py

## Inspect customer habits with K-Means

We can use K-Means to regroup customers into different categories. The centroids of each clusters describe a typical customer within the cluster.

I used a dataset on customer from [kaggle](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python). It includes multiple features: Age, Gender, Anual Income and Spending score (score between 1 & 100).

### Import dataset

```python
df = pd.read_csv("~/Mall_Customers.csv")
df.set_index("CustomerID", inplace=True)
print(df.describe())
```

### Fit model

Let's select $k = 3$ and fit the model:

```python
k = 3
X = df.to_numpy()[:, 1:]
model = KMeans(k)
model.fit(X)
```

```{note}
I removed the gender feature as it is a categorical feature and I have already 3 features.
```

### Plot result

I am using plotly to plot my result in 3D:

```python
import plotly as py
import plotly.graph_objs as go

pred = model.predict(X)
data.append(go.Scatter3d(
        x = model.centroids[:, 0],
        y = model.centroids[:, 1],
        z = model.centroids[:, 2],
        mode = 'markers',
        name = 'Centroids',
        marker = dict(
            size = 10,
            color = 'black',
            symbol = 'diamond',
        )
    ))
for i in range(k):
    data.append(go.Scatter3d(
        x = X[pred == i, 0],
        y = X[pred == i, 1],
        z = X[pred == i, 2],
        mode = 'markers',
        name = name,
        marker = dict(
            size = 5
        )
    ))

layout = go.Layout(
    title = 'Clusters by K-Means',
    scene = dict(
            xaxis = dict(title = 'Age'),
            yaxis = dict(title = 'Spending Score'),
            zaxis = dict(title = 'Annual Income')
        )
)

fig = go.Figure(data=data, layout=layout)
py.offline.iplot(fig)
```

```{figure} https://raw.githubusercontent.com/TheRayquaza/therayquaza.github.io/main/images/machine_learning/clustering/kmeans_plot.png
Clusters & Centroids of K-Means with $k = 3$
```