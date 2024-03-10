# K-nearest neighbors
## What is KNN?

K-nearest neighbors (KNN) is a non-parametric supervised learning technique created in 1951. It's based on the nearest neighbors algorithm, which is a special case of KNN ($k = 1$). KNN is often considered the simplest machine learning model.

The goal of KNN is to classify a sample based on the proximity to the training dataset. This is achieved by evaluating the relationship between two samples using different types of distances, usually the Euclidean distance.

KNN is primarily a classification method but can also be adapted for regression. For regression tasks, the average target features of the k neighbors are used to predict the target feature of the sample.

## How does KNN work for classification?

Let's denote n as the number of classified samples and k as the number of neighbors selected ($k < n$). Let x be the target sample to classify.

The KNN classifier follows a straightforward algorithm:
- Select the first k closest samples to x.
- Determine the most common class among these k samples and assign it to the target sample (x).
- Selecting the value of k

There are multiple implementation of the KNN classifier. One of them consists of just computing all distances and then extracting the k closest sample, this is: Brute-Force.

## KNN Implementation

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
In general, a larger k value leads to underfitting, while a smaller k value leads to overfitting.

## Curse of dimensionality
The curse of dimensionality in KNN refers to the inefficiency of the Euclidean distance metric in high-dimensional data spaces. As the number of dimensions increases, the distance between points becomes less meaningful.

To address the curse of dimensionality in KNN, various techniques can be employed, including:
- Feature extraction
- Dimension reduction

## References
1. KNN complexity: https://towardsdatascience.com/k-nearest-neighbors-computational-complexity-502d2c440d5
2. K-d trees: https://users.cs.utah.edu/~lifeifei/cis5930/kdtree.pdf