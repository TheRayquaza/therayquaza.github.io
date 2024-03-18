# Metrics & Losses

Metrics and Losses are mathematical functions widely used in Machine Learning.
While metrics used to evaluate the performance of a model, losses are used for training model.

Metrics and losses can be confused but they are not the same. Fundamentally, losses ajdust models and are diffenratiable thus all metrics differentiable can also be used as losses.

In this section, I will draw a list of metrics and losses commonly used in machine learning task.


## Supervised Learning: Regression

### MSE
![Loss](https://raw.githubusercontent.com/TheRayquaza/therayquaza.github.io/main/images/fundamentals/metrics/Type-Loss-3A8EDF.svg)

The MSE (Mean squared error) is the most used **metrics** for regression and is the default choice for regression task.
MSE is also used in some **loss** function because of its differentiable nature.

$$ MSE(y, f(x)) = \frac{\sum_{i=1}^n (y_i - f(x_i))^2}{n} $$

| Pros      | Cons |
| ----------- | ----------- |
| Differentiable | Metric output is not in the same unit |
| | Penalize outliers |

```python
def mse(y_true: np.array, y_pred: np.array) -> float:
    return np.mean(np.square(y_true - y_pred))
```

### MAE
![Loss](https://raw.githubusercontent.com/TheRayquaza/therayquaza.github.io/main/images/fundamentals/metrics/Type-Loss-3A8EDF.svg)

$$ MAE(y, f(x)) = \frac{\sum_{i=1}^n |y_i - f(x_i)|}{n} $$

| Pros      | Cons |
| ----------- | ----------- |
| Same unit as the output variable | Not differentiable, we need to apply optimizers |
| Robust to outliers | |

```python
def mae(y_true: np.array, y_pred: np.array) -> float:
    return np.mean(np.abs(y_true - y_pred))
```

### RMSE
![Metric](https://raw.githubusercontent.com/TheRayquaza/therayquaza.github.io/main/images/fundamentals/metrics/Type-Metric-orange.svg)

RMSE (Root mean squared error) is less used, its formula is given by:

$$ RMSE(y, f(x)) = \sqrt{\frac{\sum_{i=1}^n (y_i - f(x_i))^2}{n}} $$

| Pros      | Cons |
| ----------- | ----------- |
| Same unit as the ouput variable      | Not robust to outliers compared to MAE       |

```python
def rmse(y_true: np.array, y_pred: np.array) -> float:
    return np.sqrt(np.mean(np.square(y_true - y_pred)))
```

### Huber Loss
![Loss](https://raw.githubusercontent.com/TheRayquaza/therayquaza.github.io/main/images/fundamentals/metrics/Type-Loss-3A8EDF.svg)

If MSE is great for penalizing outliers and MAE is great for ignoring outliers, Huber loss lies somewhere between.
For a given $\delta$, its formula is given by:

$$ 
\begin{equation}
  L_{\delta}(y, f(x)) =
    \begin{cases}
      \frac{1}{2} \cdot (y - f(x))^2, for |y - f(x)| \le \delta\\ 
      \delta \cdot (|y - f(x)| - \frac{1}{2}\delta), otherwise
    \end{cases}
\end{equation}
$$

```python
def huber_loss(y_true: np.array, y_pred: np.array, delta:float) -> float:
    return np.mean(np.where(np.abs(y_true - y_pred) <= delta, 
            0.5 * np.square(y_true - y_pred),
            delta * (np.abs(y_true - y_pred) - 0.5 * delta)))
```

### Quantile Loss
![Loss](https://raw.githubusercontent.com/TheRayquaza/therayquaza.github.io/main/images/fundamentals/metrics/Type-Loss-3A8EDF.svg)

Quantile loss is used when the goal of the model is to predict a particular variable quantile.
Considering $\alpha$ the quantile we want to predict, the formula is:

$$
\begin{equation}
  L_{\alpha}(y, f(x)) =
    \begin{cases}
      \alpha \cdot (y - f(x)), f(x) \le y \\ 
      (1 - \alpha) \cdot (y - f(x)), otherwise
    \end{cases}       
\end{equation}
$$

```python
def quantile_loss(y_true: np.array, y_pred: np.array, q:float) -> float:
    return np.mean(np.where(y_pred <= y_true, 
            q * (y_true - y_pred),
            (1 - q) * (y_true - y_pred)))
```

## Supervised Learning: Classification

### Accuracy
![Metric](https://raw.githubusercontent.com/TheRayquaza/therayquaza.github.io/main/images/fundamentals/metrics/Type-Metric-orange.svg)

Accuracy measures the proportion of well classified sample.
It can be used for any types of classification: binary, multi-class or multi-label.

One definition (from [scikit-learn.org](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics)) would be:

$$ accuracy(y, f(x)) = \frac{\sum_{i=1}^N 1(y_i = f(x_i))}{N}$$

where $1(x)$ is the indicator function.

```python
def accuracy(y_true: np.array, y_pred:np.array):
    return np.mean(y_true == y_pred)
```

### Confusion Matrix

The confusion matrix is a $n$x$n$ matrix where $n$ is the number of classes in our sample.
The column represents the predicted class and the column the real class.

The cell in row $i$ and column $j$ represents the number of instances where the true class was the $i$th class and the predicted class was the $j_{th}$ class.

```python
def confusion_matrix(y_true: np.array, y_pred: np.array) -> np.array:
    classes = np.unique(np.concatenate((y_true, y_pred)))
    n_classes = len(classes)
    true_class_indices = np.searchsorted(classes, y_true)
    pred_class_indices = np.searchsorted(classes, y_pred)
    return np.bincount(n_classes * true_class_indices + pred_class_indices, minlength=n_classes * n_classes).reshape(n_classes, n_classes)

>>> confusion_matrix(np.array([0, 1, 0, 2, 1, 0]), np.array([1, 1, 0, 1, 1, 2]))
array([[1, 1, 1],
       [0, 2, 0],
       [0, 1, 0]])
```

Ideal matrix is a **diagonal matrix**.

This matrix is a valuable tool for evaluating the performance of a classification model, as it provides insight into the model's ability to correctly classify instances across different classes. From the confusion matrix, various metrics can be derived, such as precision, recall, and F1 score.

### Precision & Recall
![Metric](https://raw.githubusercontent.com/TheRayquaza/therayquaza.github.io/main/images/fundamentals/metrics/Type-Metric-orange.svg)

Precision on a class $k_i$ measure the proportion of true classification (the sample is truly of class $k_i$) with the proportion of false classification (the sample is not of class $k_i$).
The formula is given by:

$$ precision = \frac{TP}{TP + FP} $$ 
where 
- TP is "True positive": the classifier predicted class $k_i$ and the sample is truly of class $k_i$
- FP is "False positive": the classifier predicted class $k_i$, but the sample is not truly of class $k_i$

Recall is a similar metric. Recall on a class $k_i$ measure the proportion of true classification (the sample is truly of class $k_i$) with the proportion of false classification (the sample is not of class k_i).
The formula is given by:

$$ recall = \frac{TP}{TP + FN} $$
where FN is "False negative": the classifier did not predict $k_i$ but the sample is of class $k_i$ 

```{warning}
Precision and Recall are similar but not the same ! I was really confused when I discovered those metrics. Check out the [crash course from google](https://developers.google.com/machine-learning/crash-course/classification/precision-and-recall) to understand this concept.
```

Recall and Precision have different meaning and importance depending on the situation.

#### Spam or Ham ?

Let's take a classifier that identifies whether the email is a spam or a valid email.
Let's consider TP as "The email is valid". The following table illustrates two situation completly different for the classifier task.


| | Maximizing Recall | Maximizing Precision |
| ----------- | ----------- | ----------- |
| Prefered situation | Avoid classifing spam as valid email | Avoid classifing valid email as spam |
| Minimize | FN (False Negative) | FP (False Positive) |

Depending on the situation and the meaning of TP, precision and recall have different aspects. For each classification, be sure to understand both and maximizing the one you are interested in.

### F1
![Metric](https://raw.githubusercontent.com/TheRayquaza/therayquaza.github.io/main/images/fundamentals/metrics/Type-Metric-orange.svg)

F1 is a metric combining both precision and recall used to measure the model performance for both metrics.
It is defined as follow:

$$ F_1 = 2\frac{precision \cdot recall}{precision + recall} $$

Using $F_1$ gives us a balanced evaluation of the model's ability to correctly identify positive cases while minimizing FP and FN.

### Log-Loss
![Loss](https://raw.githubusercontent.com/TheRayquaza/therayquaza.github.io/main/images/fundamentals/metrics/Type-Loss-3A8EDF.svg)

Log-Loss is a well known metric used to measure the performance of a classification model where the prediction output is a probability value between 0 and 1. <br>
Log-Loss is also a loss function commonly called **Binary Cross-Entropy**.

$$ LogLoss(y, f(x)) = - \frac{1}{N} \sum_{i = 1}^N y_i \cdot log(f(x_i)) + (1 - y_i) \cdot log(1 - f(x_i)) $$

Log-Loss is widely used in Neural Network and in Logistic Regression.

```python
def log_loss(y_true: np.array, y_pred: np.array) -> float:
    return - np.mean(y_true * np.log(y_pred + 1e-10) + (1 - y_true) * np.log(1 - y_pred + 1e-10))

>>> log_loss(np.array([0, 1, 0, 0, 1, 0]), np.array([0.1, 0.9, 0.15, 0., 1., 0.5]))
0.1777311901055839
```

### Categorical Cross-Entropy
![Loss](https://raw.githubusercontent.com/TheRayquaza/therayquaza.github.io/main/images/fundamentals/metrics/Type-Loss-3A8EDF.svg)

Categorical Cross-Entropy is the generalization of the Binary Cross-Entropy.
The classifier is now predicting the probability for each class to be the valid class:

$$ CE(y, f(x)) = - \sum_{i = 1}^N y_i \cdot log(f(x)_i) $$

```python
def categorical_cross_entropy(y_true: np.array, y_pred: np.array) -> float:
    return - np.sum(y_true * np.log(y_pred + 1e-10))

>>> categorical_cross_entropy(np.array([[1,0,0],[0,1,0],[0,0,1],[1,0,0]]), np.array([[0.91,0.04,0.05],[0.11,0.8,0.09],[0.3,0.1,0.6],[0.25,0.4,0.35]]))
2.2145742148697756
```

### Hinge Loss
![Loss](https://raw.githubusercontent.com/TheRayquaza/therayquaza.github.io/main/images/fundamentals/metrics/Type-Loss-3A8EDF.svg)

Hinge Loss is a loss function used in SVM model.

$$ Hinge(y, f(x)) = max(0, 1 - y \cdot f(x))$$

```python
def hinge_loss(y_true: np.array, y_pred: np.array) -> float:
    return np.max([np.zeros(y_true.shape), 1 - y_true * y_pred])

>>> hinge_loss(np.array([0, 1, 2, 3]), np.array([1, 1, 2, 3]))
1.0
```

## Unsupervised Learning: Clustering

### Silhouette Score
![Metric](https://raw.githubusercontent.com/TheRayquaza/therayquaza.github.io/main/images/fundamentals/metrics/Type-Metric-orange.svg)

The silhouette score is a metric used in clustering task to measures how similar is an object with its own cluster compared to the other clusters.
To define the silhouette score, we need two values $a_i$ (intra-cluster distance) and $b_i$ (nearest-cluster distance):

For $i \in C_I$, where $C_I$ is the set of point in cluster $I$ and a distance $d$ (usually euclidean or manhattan) we have:

$$ a_i = \frac{1}{card(C_I) - 1} \sum_{j \in C_I, i \neq j} d(i, j) $$

and

$$ b_i = min_{J \neq I} \frac{1}{card(C_I)} \sum_{j \in C_J} d(i, j) $$

We can compute the silhouette score for the datapoint $i$:

$$
\begin{equation}
  s_i =
    \begin{cases}
      1 - \frac{a_i}{b_i}, a_i < b_i \\ 
      0, a_i = b_i \\
      \frac{b_i}{a_i} - 1, a_i > b_i  
    \end{cases}
\end{equation}
$$

## References
1. Regression & Classification: [Scikit-Learn Documentation](https://scikit-learn.org/)
2. Regression & Classification: Hands-On Machine Learning with Scikit-Learn by Aurélien Géron
3. Classifcation Metrics: [Google Course](https://developers.google.com/machine-learning/crash-course)
4. F1-Score: [Wikipedia](https://en.wikipedia.org/wiki/F-score)
5. Log-Loss: [Towards Data Science](https://towardsdatascience.com/intuition-behind-log-loss-score-4e0c9979680a)
6. Huber Loss: [Wikipedia](https://en.wikipedia.org/wiki/Huber_loss)
7. Quantile Loss: [Towards Data Science](https://towardsdatascience.com/quantile-loss-and-quantile-regression-b0689c13f54d)
8. Categorical Cross-Entropy: [Medium](https://neuralthreads.medium.com/categorical-cross-entropy-loss-the-most-important-loss-function-d3792151d05b)
9. Hinge Loss: [Wikipedia](https://en.wikipedia.org/wiki/Hinge_loss)
10. Silhouette: [Wikipedia](https://en.wikipedia.org/wiki/Silhouette_(clustering))
