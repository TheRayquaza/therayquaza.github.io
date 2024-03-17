# Metrics & Losses

Metrics and Losses are mathematical functions widely used in Machine Learning.
While metrics used to evaluate the performance of a model, losses are used for training model.

Metrics and losses can be confused but they are not the same. Fundamentally, losses ajdust models and are diffenratiable thus all metrics differentiable can also be used as losses.

In this section, I will draw a list of metrics and losses commonly used in machine learning task.

## Supervised Learning: Regression

### MSE
![Loss](https://raw.githubusercontent.com/TheRayquaza/therayquaza.github.io/main/images/fundamentals/metrics/Type-Loss-3A8EDF.svg) ![Metric](https://raw.githubusercontent.com/TheRayquaza/therayquaza.github.io/main/images/fundamentals/metrics/Type-Metric-orange.svg)

The MSE (Mean squared error) is the most used **metrics** for regression and is the default choice for regression task.
MSE is also used in some **loss** function because of its differentiable nature.

$$ MSE = \frac{\sum_{i=1}^n (y_i - x_i)^2}{n} $$

| Pros      | Cons |
| ----------- | ----------- |
| Differentiable | Metric output is not in the same unit |
| | Penalize outliers |

### MAE
![Loss](https://raw.githubusercontent.com/TheRayquaza/therayquaza.github.io/main/images/fundamentals/metrics/Type-Loss-3A8EDF.svg) ![Metric](https://raw.githubusercontent.com/TheRayquaza/therayquaza.github.io/main/images/fundamentals/metrics/Type-Metric-orange.svg)

$$ MAE = \frac{\sum_{i=1}^n |y_i - x_i|}{n} $$

| Pros      | Cons |
| ----------- | ----------- |
| Same unit as the output variable | Not differentiable, we need to apply optimizers |
| Robust to outliers | |

### RMSE
![Metric](https://raw.githubusercontent.com/TheRayquaza/therayquaza.github.io/main/images/fundamentals/metrics/Type-Metric-orange.svg)

RMSE (Root mean squared error) is less used, its formula is given by:

$$ RMSE = \sqrt{\frac{\sum_{i=1}^n (y_i - x_i)^2}{n}} $$

| Pros      | Cons |
| ----------- | ----------- |
| Same unit as the ouput variable      | Not robust to outliers compared to MAE       |

## Supervised Learning: Classification

### Accuracy
![Metric](https://raw.githubusercontent.com/TheRayquaza/therayquaza.github.io/main/images/fundamentals/metrics/Type-Metric-orange.svg)

Accuracy measures the proportion of well classified sample.
It can be used for any types of classification: binary, multi-class or multi-label.

One definition (from [scikit-learn.org](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics)) would be:

$$ accuracy(y, x) = \frac{\sum_{i=1}^n 1(y = x)}{n}$$

where $1(x)$ is the indicator function.

### Confusion Matrix

The confusion matrix is a $n$x$n$ matrix where $n$ is the number of classes in our sample.
The column represents the predicted class and the column the real class.

The cell in row $i$ and column $j$ represents the number of instances where the true class was the $i$th class and the predicted class was the $j$th class.

This matrix is a valuable tool for evaluating the performance of a classification model, as it provides insight into the model's ability to correctly classify instances across different classes.

From the confusion matrix, various metrics can be derived, such as accuracy, precision, recall, and F1 score, which help assess the overall effectiveness of the classification model.

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
Precision and Recall are similar but not the same ! I was really confused when I was learning about those metrics. Check out the [crash course from google](https://developers.google.com/machine-learning/crash-course/classification/precision-and-recall) to understand this concept.
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

### Log-Loss (Cross-Entropy Loss)
![Loss](https://raw.githubusercontent.com/TheRayquaza/therayquaza.github.io/main/images/fundamentals/metrics/Type-Loss-3A8EDF.svg) ![Metric](https://raw.githubusercontent.com/TheRayquaza/therayquaza.github.io/main/images/fundamentals/metrics/Type-Metric-orange.svg)

Log-Loss is a well known metric used to measure the performance of a classification model where the prediction output is a probability value between 0 and 1.



## References
1. https://scikit-learn.org/
2. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow by Aurélien Géron
3. https://developers.google.com/machine-learning/crash-course
4. https://en.wikipedia.org/wiki/F-score
5. https://towardsdatascience.com/intuition-behind-log-loss-score-4e0c9979680a
6. https://medium.com/@sujathamudadla1213/what-is-the-difference-between-loss-function-and-metrics-in-machine-learning-3fea45d5882b
