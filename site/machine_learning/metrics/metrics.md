# Metrics

Metrics are mathematical functions used to evaluate the performance of a model.
In this section, I will draw a list of metrics that are commonly used in machine learning task.

## Metrics for regression (Supervised Learning)

### MSE (Mean squared error)

The MSE is the most used metrics for regression and is the default choice for regression task.
MSE is also used in some loss function because of its differentiable nature.

$$ MSE = \frac{\sum_{i=1}^n (y_i - x_i)^2}{n} $$

Pros:
- Differentiable

Cons:
- Metric output is not in the same unit
- Penalize outliers

### MAE (Mean absolute error)

$$ MAE = \frac{\sum_{i=1}^n |y_i - x_i |}{n} $$

Pros:
- Same unit as the output variable
- Robust to outliers

Cons:
- Not differentiable, we need to apply optimizers

### RMSE (Root mean squared error)

$$ RMSE = \sqrt{\frac{\sum_{i=1}^n (y_i - x_i)^2}{n}} $$

Pros:
- Same unit as the ouput variable

Cons:
- Not robust to outliers compared to MAE

## Metrics for classification (Supervised Learning)

### Accuracy

Accuracy measures the proportion of well classified sample.
It can be used for any types of classification: binary, multi-class or multi-label.

One definition (from [scikit-learn.org](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics)) would be:

$$ accuracy(y, x) = \frac{\sum_{i=1}^n 1(y = x)}{n}$$

where $1(x)$ is the indicator function.

### Confusion Matrix

The confusion matrix is a $n$x$n$ matrix where $n$ is the number of classes in our sample.
The column represents the predicted class and the column the real class.

The cell in row $i$ and column $j$ represents the number of instances where the true class was the $i$th class and the predicted class was the $j$th class.

This matrix is a valuable tool for evaluating the performance of a classification model, as it provides insight into the model's ability to correctly classify instances across different classes. From the confusion matrix, various metrics can be derived, such as accuracy, precision, recall, and F1 score, which help assess the overall effectiveness of the classification model.

Additionally, visual representations of confusion matrices, such as heatmaps, can aid in interpreting and identifying patterns in classification errors, thus guiding further model improvement efforts.

### Precision

### Recall

### F1

## References
1. https://scikit-learn.org/
2. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow by Aurélien Géron