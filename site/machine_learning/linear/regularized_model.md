# Regularized Linear Model

![Regression](https://raw.githubusercontent.com/TheRayquaza/therayquaza.github.io/main/images/badges/Regression.svg)

Linear Regression is the simplest type of regression model. It does not include regularization to help preventing overfitting. For ridge regression, lasso regression and elastic net, regularization adds constraint on the coefficient using different norm ($l_1$ or/and $l_2$).

In this section I will draw some of the regularization linear model for regression task.

Considering:

$$ MSE(\theta) = \frac{1}{2N} (\theta X - y)^2 $$

$$ \nabla_{\theta} MSE(\theta) = \frac{1}{N} X^T(X\theta - y) $$

## Ridge Regression

Ridge regression uses $l_2$ regularization: it applies a $l_2$ norm on each of the model's weights in the cost function.
Ridge regression is suitable for multicollinear data.

### Cost function & Gradient

$$ J(\theta) = MSE(\theta) + \lambda \sum_{j=1}^{n} \theta_j^2 $$

$$ \nabla_{\theta} J(\theta) = \nabla_{\theta} MSE(\theta) + \lambda \theta $$

### Implementation

Here a naive implementation of the ridge regression:

```python
class RidgeRegression:
    def __init__(self, it=30, lr=1e-3, alpha=1):
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

Here a graph to show the impact of the regularization factor $\lambda$ (here called alpha) on a polynomial dataset:

```{figure} https://raw.githubusercontent.com/TheRayquaza/therayquaza.github.io/main/images/machine_learning/linear/ridge_regression.png
Ridge regression with different $\alpha$
```

## Lasso Regression

Lasso regression is really similar to ridge regression. It uses $l_1$ regularization instead of $l_2$. Lasso regression is suitable for feature selection.

### Cost function & Gradient

$$ J(\theta) = MSE(\theta) + \lambda \sum_{j=1}^{n} \| \theta_j \| $$

$$ \nabla_{\theta} J(\theta) = \nabla_{\theta} MSE(\theta)+ \lambda \cdot sign(\theta) $$

### Implementation

Here a naive implementation of Lasso regression:

```python
class LassoRegression:
    def __init__(self, it=30, lr=1e-3, alpha=0.1):
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

Here a graph to show the impact of the regularization factor $\lambda$ (called alpha) on a polynomial dataset:

```{figure} https://raw.githubusercontent.com/TheRayquaza/therayquaza.github.io/main/images/machine_learning/linear/lasso_regression.png
Lasso regression with different $\alpha$
```

## Elastic Net

Elastic Net is using a mix of $l_1$ and $l_2$ taking advantage of both Ridge & Lasso Regression.

### Cost function & Gradient

$$ J(\theta) = MSE(\theta) + \lambda_1 \lambda_2 \sum_{j=1}^{n} \| \theta_j \| + \lambda_1 \frac{1 - \lambda_2}{2} \sum_{j=1}^{n} \theta_j^2 $$

$$ \nabla_{\theta} J(\theta) = \nabla_{\theta} MSE(\theta)+ \lambda_1 \cdot sign(\theta) + \lambda_2 \theta $$

### Implementation

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