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

...

### Distribution-based

Distribution-based techniques are less commonly used because they require knowledge of the underlying distribution in our data.
...

## References
1. Unsupervised Learning: Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow by Aurélien Géron
2. Clustering: [Google Clustering course](https://developers.google.com/machine-learning/clustering/clustering-algorithms)
3. K-Means: [Wikipedia](https://en.wikipedia.org/wiki/K-means_clustering)
4. DBSCAN: [Medium](https://medium.com/@okanyenigun/dbscan-demystified-understanding-how-this-parameter-free-algorithm-works-89e03d7d7ab)
5. DBSCAN: [Wikipedia](https://en.wikipedia.org/wiki/DBSCAN)