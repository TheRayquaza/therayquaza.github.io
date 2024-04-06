# Affinity propagation

![Clustering](https://raw.githubusercontent.com/TheRayquaza/therayquaza.github.io/main/images/badges/Clustering.svg)

## How does affinity propagation works ?

Affinity propagation (AP) considers each observations as a node in a graph. Each node receives information from neighbor nodes to estimate which nodes should be considered as examplar. After many iterations, each nodes will have its own examplar node assigned and will therefore have its own cluster centered on this examplar node.

As input, AP receives a similarity matrix $S$ that is usually a (euclidean) distance matrix of the dataset.

The matrix $R$ (known as the **responsibility** matrix) assigns values $r_{ik}$ indicating the suitability of $x_k$ as an exemplar for $x_i$ compared to other observations $x_i$.

The matrix $A$ (known as the **availability** matrix) gives us the value $a_{ik}$ which shows how suitable it is for an observation $x_i$ to choose another point $x_k$ as its example. It considers how much other points prefer $x_k$ as an example.

## Algorithm

Let's denote, $X$ the observations, $N$ the number of observations and $k$ the number of features.

- The first step in the algorithm is to compute the similarity matrix and fill our $R$ and $A$ matrices with zeros:

$$ \forall i < N, \forall j < N, S_{ij} = || X_i - X_j ||_2 $$

Now we iterate until convergence of $R$ and $A$:
- $\forall i < N, \forall k < N, R_{ik} = S_{ik} - max_{j \neq k}(A_{ij} + S_{ij})$
- $\forall i < N, \forall k < N, i \neq k, A_{ik} = min(0, R_{k,k} + \sum_{j \notin {i,k}} max(0, R_{jk}))$
- $\forall i < N, A_{ii} = \sum_{j \neq i} max(0, R_{ji})$

Once convergence, we can calculate the examplar observation $E$ for all observations and thus find the clusters:
$ \forall i < N, E_{i} = argmax_k(A_{ik} + R_{ik}) $

## Implementation in python

```python
import numpy as np

class AffinityPropagation:
    def __init__(self, max_iter=200, tolerance=1e-5):
        self.max_iter = max_iter
        self.tolerance = tolerance
        
    def update_R(self):
        new_R = np.copy(self.R)
        for i in range(self.N):
            for k in range(self.N):
                new_R[i, k] = self.S[i, k] - np.max([self.A[i, j] + self.S[i, j] for j in range(self.N) if j != k])
        return new_R

    def update_A(self):
        new_A = np.copy(self.A)
        for i in range(self.N):
            for k in range(self.N):
                if i != k:
                    new_A[i, k] = min(0, self.R[k, k] + np.sum([max(0, self.R[j, k]) for j in range(self.N) if j != i and j != k]))
        for i in range(self.N):
            new_A[i, i] = np.sum([max(0, self.R[j, i]) for j in range(self.N) if j != i])
        return new_A

    def fit(self, X):
        self.N, self.k = X.shape
        self.S = np.sqrt(np.sum(np.square(X - X[:, None]), axis=-1))
        self.R = np.zeros((self.N, self.N))
        self.A = np.zeros((self.N, self.N))
        for _ in range(self.max_iter):
            old_R, old_A = np.copy(self.R), np.copy(self.A)
            self.R = self.update_R()
            self.A = self.update_A()
            if np.max(np.abs(old_R - self.R)) < self.tolerance and np.max(np.abs(old_A - self.A)) < self.tolerance:
                break

        self.E = np.zeros(self.N, dtype=int)
        for i in range(N):
            self.E[i] = np.argmax(self.A[i] + self.R[i])
        
        self.cluster_centers_indices_ = np.unique(self.E)
        self.n_clusters = len(self.cluster_centers_indices_)
        self.centroids = np.zeros((self.n_clusters, self.k))
        
        label_dict = {center_index: label for label, center_index in enumerate(self.cluster_centers_indices_)}
        self.labels_ = np.array([label_dict[center_index] for center_index in self.E])

        for i in range(self.n_clusters):
            self.centroids[i] = np.mean(X[self.E == self.cluster_centers_indices_[i]], axis=0)
        return self

    def predict(self, X: np.array) -> np.array:
        distances = np.linalg.norm(X[:, np.newaxis, :] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)
```