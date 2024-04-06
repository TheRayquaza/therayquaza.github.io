# Clustering

```{tableofcontents}
```

## What is clustering ?

The task of grouping unlabeled observations together is called **clustering**. Clustering is an unsupervised machine learning technique: it involves unlabeled data.

Clustering is used in for diverse applications including:
- Data Analysis
- Dimensionality reduction techniques
- Image segmentation
- Outlier detection

## What are the different types of clustering techniques?

### Centroid-based

Centroid techniques regroup all techniques that involve grouping observations using defined centroid, centers of each clusters. It creates a non-hierarchical clusters meaning that it directly assigns data points to clusters without forming a hierarchy. It is based on a distance metric, which can lead to ineffective results as observations have more dimensions.

One famous centroid technique is [K-Means](https://therayquaza.github.io/machine_learning/clustering/kmeans.html).

```{figure} https://raw.githubusercontent.com/TheRayquaza/therayquaza.github.io/main/images/machine_learning/clustering/kmeans_clusters_with_decision_boundary.png
K-Means of scikit applied for 3 centers
```

### Density-based

Density based techniques involves all techniques trying to connects area of with dense obsvervations into clusters. It is also based on distance and have same problematic as Centroid-based techniques.

The most popular and simple technique using density is [DBSCAN](https://therayquaza.github.io/machine_learning/clustering/dbscan.html)

```{figure} https://raw.githubusercontent.com/TheRayquaza/therayquaza.github.io/main/images/machine_learning/clustering/dbscan_clusters.png
DBSCAN of scikit identifying clusters ($\epsilon$ = 0.5 & $min_samples=5$)
```

### Hierarchical

Hierarchical techniques are all techniques involving separating clusters in a hirarchical tree structure.

Hierarchical techniques are divided in 2 separates type of techniques:
- Agglomerative (*bottom-up* approach)
- Divisive (*top-down* approach)

Agglomerative techniques are more common.

Hierarchical techniques can be plotted as a dendrogram (using plot code from [Scikit](https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_dendrogram.html#sphx-glr-auto-examples-cluster-plot-agglomerative-dendrogram-py)):

```{figure} https://raw.githubusercontent.com/TheRayquaza/therayquaza.github.io/main/images/machine_learning/clustering/hierarchical_dendogram.png
Agglomerative Hierarchical clustering on customer's data for behavior analysis (taken from [Kaggle](https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis)]
```

The dendogram above is an effective way of representing clusters of customer behaviors. Note that the dataset included 28 features (both numerical and categorical) which could never be plotted as we did for the other techniques. Each hierarchy can then have its own analysis to understand the type of customers and how we could have impact on each category.

### Distribution-based

Distribution-based techniques are less commonly used because they require knowledge of the underlying distribution in our data.
These techniques try to find optimal distrubtions that represent the best each obsverations.
Distribution-based techniques can perform really well on clusters having different sizes (which is not the case for Centroid-based for example).

Here is an example of using distrubtion-based clustering with a [gaussian mixture model](https://therayquaza.github.io/machine_learning/clustering/gaussian_mixture.html) (hence guassian assumption):

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

mean1, cov1 = np.array([-2, 1]), np.array([[2, 1], [1, 3]])
n1 = np.random.multivariate_normal(mean1, cov1, 100)

mean2, cov2 = np.array([4, -3]), np.array([[4, 2], [2, 3]])
n2 = np.random.multivariate_normal(mean2, cov2, 100)

dataset = np.vstack([n1, n2])
clusters = GaussianMixture(n_components=2).fit_predict(dataset)
```

```{figure} https://raw.githubusercontent.com/TheRayquaza/therayquaza.github.io/main/images/machine_learning/clustering/density_based_GMM.png
GMM sklearn - 2 bivariate distributions $\mathcal{N}(\begin{bmatrix} -2 \\ 1 \end{bmatrix},\begin{bmatrix} 2 & 1 \\ 1 & 3 \end{bmatrix})\,$ and $\mathcal{N}(\begin{bmatrix} 4 \\ -3 \end{bmatrix},\begin{bmatrix} 4 & 2 \\ 2 & 3 \end{bmatrix})\,$
```

I used matrix notation to define $\mu$ the mean of the joint distrubtion and $\sigma$ the covariance matrix (needs to be symetric because $\sigma_{XY} = \sigma_{YX}$).

```{note}
In this example, GMM performs really well since the observations have been created directly using gaussian distrubtions. It is never the case ! GMM performs really well when we know that observations' distrubtions are gaussians.
```

### Exemplar-based

Exemplar-based clustering is a type of clustering where we try to find the most representative observations. Exemplar-based clustering does not require the number of clusters to be given by the user.

The most popular models are [K-Medoids](https://medium.com/@ali.soleymani.co/beyond-scikit-learn-is-it-time-to-retire-k-means-and-use-this-method-instead-b8eb9ca9079a) and [Affinity propagation](https://therayquaza.github.io/machine_learning/clustering/affinity_propagation.html).

Here is an example of AP algorithm applied on a 2D simple dataset:

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AffinityPropagation
from sklearn.datasets import make_blobs

centers = np.array([[1, 2], [-1, 1], [3, 1]])
X, y_true = make_blobs(n_samples=100, centers=centers)

af = AffinityPropagation(preference=-100, random_state=42).fit(X)
centers_indices = af.cluster_centers_indices_

cmap = plt.get_cmap('tab10')
colors = [cmap(i) for i in range(len(centers_indices))]

for k in range(len(centers_indices)):
    X_k = X[af.labels_ == k]
    center = X[centers_indices[k]]
    for x in X_k:
        plt.plot(np.hstack([x[0], center[0]]), np.hstack([x[1], center[1]]), color=colors[k], alpha=0.5)
        plt.scatter(x[0], x[1], color=colors[k], alpha=0.7)
for center in centers:
    plt.scatter(center[0], center[1], color='black', marker='*')

plt.title("AP on 2D dataset with clusters at (1,2) (-1,1) and (3,1)")
plt.show()
```

```{figure}  https://raw.githubusercontent.com/TheRayquaza/therayquaza.github.io/main/images/machine_learning/clustering/ap_on_2d.png
Affinity propagation with preference -100 on 2D dataset
```

## References
1. Unsupervised Learning: Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow by Aurélien Géron
2. Clustering: [Google Clustering course](https://developers.google.com/machine-learning/clustering/clustering-algorithms)
3. K-Means: [Wikipedia](https://en.wikipedia.org/wiki/K-means_clustering)
4. DBSCAN: [Medium](https://medium.com/@okanyenigun/dbscan-demystified-understanding-how-this-parameter-free-algorithm-works-89e03d7d7ab)
5. DBSCAN: [Wikipedia](https://en.wikipedia.org/wiki/DBSCAN)
6. Clustering: [Western Michigan University](https://cs.wmich.edu/alfuqaha/summer14/cs6530/lectures/ClusteringAnalysis.pdf)
7. Great video on GM: [Youtube](https://www.youtube.com/watch?v=EWd1xRkyEog)
8. Taxonomy of clustering techniques: [Medium](https://medium.com/@sayahfares19/k-means-clustering-algorithm-for-unsupervised-learning-tasks-f761ed7f37c0)
9. EM algorithm for GMM: [Medium](https://jonathan-hui.medium.com/machine-learning-expectation-maximization-algorithm-em-2e954cb76959)
10. AP: [Scikit-Lean](https://scikit-learn.org/stable/modules/clustering.html#affinity-propagation)