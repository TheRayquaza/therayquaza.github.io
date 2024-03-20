# Regularized Linear Model

Linear Regression is the simplest type of regression model. It does not include regularization in its cost function which help preventing the weights to become insanly huge.

In this section I will draw some of the regularization linear model for regression task.

Considering:

$$ MSE(\theta) = \frac{1}{2N} (\theta X - y)^2 $$
$$ \nabla_{\theta} MSE(\theta) = \frac{1}{N} X^T(X\theta - y) $$

We have the following models:

|                   | Ridge Regression | Lasso Regression | Elastic Net |
|-------------------|------------------|------------------|-------------|
| Parameters        | $ \theta ,  \lambda $ | $ \theta , \lambda $ | $ \theta ,  \lambda_1 ,  \lambda_2 $ |
| Regularization    | $l_2$            | $l_1$            | $l_1$ and $l_2$ |
| Cost Function ($J(\theta)$)    | $ MSE(\theta) + \lambda \sum_{j=1}^{n} \theta_j^2 $ | $ MSE(\theta) + \lambda \sum_{j=1}^{n} \| \theta_j \| $ | $ MSE(\theta) + \lambda_1 \lambda_2 \sum_{j=1}^{n} \| \theta_j \| + \lambda_1 \frac{1 - \lambda_2}{2} \sum_{j=1}^{n} \theta_j^2 $ |
| Gradient of Cost Function ($ \nabla_{\theta} J(\theta)$) | $ \nabla_{\theta} MSE(\theta)+ \lambda \theta $ | $ \nabla_{\theta} MSE(\theta)+ \lambda \cdot sign(\theta) $ | $ \nabla_{\theta} MSE(\theta)+ \lambda_1 \cdot sign(\theta) + \lambda_2 \theta $ |
| Usage case        | Suitable for multicollinear data   | Suitable for feature selection     | Combines benefits of $l_1$ and $l_2$ regularization |

## Ridge Regression

Here a naive implementation of the ridge regression:

```python
class RidgeRegression:
    def __init__(self, it=20, lr=1e-4, alpha=1):
        self.it = it
        self.lr = lr
        self.alpha = 1

    def fit(self, X, y):
        self.weights = np.random.randn(X.shape[1] + 1)
        X = np.hstack([X, np.ones((X.shape[0], 1))])
        for _ in range(self.it):
            self.weights -= self.lr * (0.5 * np.mean(X.T @ (X @ self.weights - y)) + self.alpha * self.weights)

    def predict(self, X):
        return np.hstack([X, np.ones((X.shape[0], 1))]) @ self.weights
```

## Lasso Regression

```python
class LassoRegression:
    def __init__(self, it=20, lr=1e-4, alpha=1):
        self.it = it
        self.lr = lr
        self.alpha = 1

    def fit(self, X, y):
        self.weights = np.random.randn(X.shape[1] + 1)
        X = np.hstack([X, np.ones((X.shape[0], 1))])
        for _ in range(self.it):
            self.weights -= self.lr * (0.5 * np.mean(X.T @ (X @ self.weights - y)) + self.alpha * np.sign(self.weights))

    def predict(self, X):
        return np.hstack([X, np.ones((X.shape[0], 1))]) @ self.weights
```

## Elastic Net

```python
class ElasticNet:
    def __init__(self, it=20, lr=1e-4, alpha=1, beta=1):
        self.it = it
        self.lr = lr
        self.alpha = 1
        self.beta = beta

    def fit(self, X, y):
        self.weights = np.random.randn(X.shape[1] + 1)
        X = np.hstack([X, np.ones((X.shape[0], 1))])
        for _ in range(self.it):
            self.weights -= self.lr * (0.5 * np.mean(X.T @ (X @ self.weights - y)) + self.alpha * np.sign(self.weights) + self.beta * self.weights)

    def predict(self, X):
        return np.hstack([X, np.ones((X.shape[0], 1))]) @ self.weights
```