# Linear Regression

![Regression](https://raw.githubusercontent.com/TheRayquaza/therayquaza.github.io/main/images/badges/Regression.svg)

## Assumption

Linear model assumes there is a **linear** relationship within the set of observation.

## Diving into math

Linear model for a regression task in $N$ dimensions have equation of the form:

$$ y = \theta_{0} + \sum_{i = 1}^N \theta_{i} x_i $$

$\theta_{0}$ is the constant factor and the $\theta_{i}$ are the coefficients associated with each features. Those variables are called **weights**.

In general, for a $N$ dimensions problem, a linear model will try to create a $N-1$ hyperplane to minimize its loss function.

Linear regression can be adapted for classification task (see for example [LogisticRegression](https://therayquaza.github.io/machine_learning/linear/logisitic_regression.html))

We define the model's prediction function $f_{\theta}(x)$ where $x$ are the observations and $\theta$ the parameters. For linear regression, the prediction function is simply defined as:

$$ f_{\theta}(x) = \theta_{0} + \sum_{i = 1}^n \theta_{i} x_i $$

```{note}
This can also be written as a dot product to which we add the y slope intercept (for each obsverations): $ f_{\theta}(X) = \theta_{0} + X \cdot \theta $
```

The model tries to minimize its loss function (generally an adaptation of [MSE](https://therayquaza.github.io/machine_learning/metrics_and_losses/metrics_and_losses#mse)). Given $f_{\theta}$ the function we are trying to approximate, we have:

Considering a cost function $J(\theta)$:

$$ J(\theta) = \frac{1}{2} MSE(f_{\theta}, y) = \frac{1}{2N} \sum_{i = 1}^N (f_{\theta}(x_i) - y_i)^2 $$

And using the vectorized notation:

$$ J(\theta) = (X\theta - y)^T (X\theta - y) $$

## Analytical Solution

Linear regression models can be solved easily with a simple equation:

$$ \theta = (X^TX)^{-1} X^Ty $$

The proof (from [Toward Data Science](https://towardsdatascience.com/analytical-solution-of-linear-regression-a0e870b038d5)) is the following:

Using some tricks (as [explained](https://towardsdatascience.com/analytical-solution-of-linear-regression-a0e870b038d5)), we can write the cost function $J({\theta})$:

$$ J(\theta) = \theta^TX^TX\theta - 2y^TX\theta + y^Ty $$

We can compute the derivative with respect to $\theta$:

$$ 
\begin{equation}
\begin{split}
\frac{\partial J(\theta)}{\partial \theta} &= \frac{\partial (\theta^TX^TX\theta)}{\partial \theta} - 2y^TX\frac{\partial \theta}{\partial \theta} \\
& = 2X^TX\theta - 2y^TX
\end{split}
\end{equation}
$$

In order to minimize the cost function, we need the derivative to be equal to zero:

$$
\begin{align}
\frac{\partial J(\theta)}{\partial \theta} &= 0 \\
& \Rightarrow 2X^TX\theta - 2y^TX = 0 \\
& \Rightarrow X^TX\theta = y^TX \\
& \Rightarrow \theta = (X^TX)^{-1} X^Ty
\end{align}
$$

## Gradient Descent

Another well known technique is called Gradient Descent. Gradient descent is an **optimization** technique used to slve many problems thanks to its genericity.

The goal of gradient descent is to **minimze** a model's cost function by calculating the **gradient** with respect to the model's parameters.

```{figure} https://raw.githubusercontent.com/TheRayquaza/therayquaza.github.io/main/images/machine_learning/linear/GD_Basic.png
Gradient Descent on $y = x^2$ and $lr = 0.1$
```

How are the parameters updated ?

We compute the partial derivative of the cost function $J(\theta)$ with respect to each features. There is special and great operator that computes the partial derivative with respect to each of the feature of a given function: **the gradient**. Gradient is of a cost function with parameters $\theta_{i}$ is given by:

$$ \nabla_{\theta} J(\theta) = \begin{bmatrix} \frac{\partial}{\partial \theta_0} J(\theta) \\  \frac{\partial}{\partial \theta_1} J(\theta) \\ ... \\ \frac{\partial}{\partial \theta_n} J(\theta) \end{bmatrix} $$

Parameters can be updated with the computed gradient and a learning rate ($\alpha$):

$$ \theta = \theta - \alpha \nabla_{\theta} J(\theta)$$

For linear regression, we try to minimize the MSE loss function as defined earlier. Calculating its derivative with respect to $n$ features, we get:

$$ \nabla_{\theta} MSE(\theta) = \frac{2}{N} X^T(X\theta - y)$$
$$ \theta = \theta - \alpha \frac{2}{N} X^T(X\theta - y) $$

```{note}
For gradient descent, we need our loss function to be differentiable !
```

## Linear Regression with gradient descent

I will provide a simple implementation of a linear regression model using the classic gradient descent:

```python
class LinearRegression:
    def __init__(self, it=20, lr=1e-3):
        self.it = it
        self.lr = lr

    def fit(self, X, y):
        self.weights = np.random.randn(X.shape[1] + 1)
        X = np.hstack([X, np.ones((X.shape[0], 1))])
        for _ in range(self.it):
            self.weights -= self.lr * (0.5 * np.mean(X.T @ (X @ self.weights - y)))
        self.coef_ = self.weights[:-1]
        self.intercept_ = self.weights[-1]

    def predict(self, X):
        return np.hstack([X, np.ones((X.shape[0], 1))]) @ self.weights
```

This a naive implementation of the linear regression using the classic gradient descent. Other technique involving partial gradient on a part of the dataset can also be used (see Mini-Batch GD & Stochastic GD).

```{note}
I added a weight corresponding to the y-intercept. To add the bias to the final prediction, we need to create a synthetic feature by stacking one values.
```

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

np.random.seed(42)
X = 15 * np.random.rand(100, 1) - 5
y = X + 2 + np.random.randn(100, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

lin_reg = LinearRegression()
lin_reg.fit(X, y)
plt.scatter(X_train, y_train, color='blue', alpha=0.5)
plt.scatter(X_test, y_test, color='green', alpha=0.5)
plt.plot(X_test, lin_reg.predict(X_test), color='red', alpha=0.5)
plt.title(f'y = {float(lin_reg.coef_[0]):.2f}x + {float(lin_reg.intercept_):.2f}')
plt.xlabel('Feature')
plt.ylabel('Target')

plt.tight_layout()
plt.show()
```

```{figure}  https://raw.githubusercontent.com/TheRayquaza/therayquaza.github.io/main/images/machine_learning/linear/Linear_With_GD.png
Linear Regression with GD
```

## Using Linear Regression for polynomial problem

If linear model cannot recognize non linear pattern, we can use a technique called **feature engineering**. The goal of feature engineering is to create or modify features to improve model's performance. This helps simple model to identify complex pattern and at the same time keeping the simplicity.

Here is simple example where we know the expected polynomial form of target. We can directly generate the right feature but in real-world problem, **this is never the case**.

Let's create a synthetic dataset to represent $ y = x^2 + x + 2$. We add some noise into the dataset and we split our train and test datasets:

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

np.random.seed(42)
X = 15 * np.random.rand(100, 1) - 5
y = X**2 + X + 2 + np.random.randn(100, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)
```

Using a simple linear regression, we can get a linear approximation of the dataset (something like $ y = \alpha x + \beta $):

```python
lin_reg = LinearRegression()
lin_reg.fit(X, y)
plt.scatter(X_train, y_train, color='blue', alpha=0.5)
plt.scatter(X_test, y_test, color='green', alpha=0.5)
plt.plot(X_test, lin_reg.predict(X_test), color='red', alpha=0.5)
plt.title(f'y = {float(lin_reg.coef_[0]):.2f}x + {float(lin_reg.intercept_):.2f}')
plt.xlabel('Feature')
plt.ylabel('Target')

plt.tight_layout()
plt.show()
```

```{figure} https://raw.githubusercontent.com/TheRayquaza/therayquaza.github.io/main/images/machine_learning/linear/Linear_Polynomial_Failure.png
Linear model failed learning non linear relations with a linear form assumption
```

We are not satisfied with this prediction ! The model could not extract the polynomial relationship. Hopefully, we can create a new feature $x^2$ to modelize a polynomial relationship. With this synthetic feature, the model will now try to minimize its loss function for a function of the form $y = \alpha x^2 + \beta x + \lambda$:

```python
lin_reg = LinearRegression()
X_trans = np.hstack([X_train ** 2, X_train])
lin_reg.fit(X_train, y_train)
plt.scatter(X_train, y_train, color='blue', alpha=0.5)
X_test = np.sort(X_test.flatten()).reshape(-1, 1)
plt.plot(X_test, lin_reg.coef_[1] * X_test ** 2 + lin_reg.coef_[0] * X_test + lin_reg.intercept_, color='red')
plt.title(f'y = {float(lin_reg.coef_[1]):.2f}x² + {float(lin_reg.coef_[0]):.2f}x + {float(lin_reg.intercept_):.2f}')
plt.xlabel('Feature (Transformed)')
plt.ylabel('Target')

plt.tight_layout()
plt.show()
```

```{figure} https://raw.githubusercontent.com/TheRayquaza/therayquaza.github.io/main/images/machine_learning/linear/Linear_Polynomial_Success.png
Linear model learning non linear realations ($y = x^2 + x + 2$)
```

This is an example where we know the type of relation between x and y.

In this example, I just created a new feature $x^2$ but feature engineering includes other techniques: Feature Selection, **Feature Transformation** ($x^2$), Feature Creation and Feature Extraction.