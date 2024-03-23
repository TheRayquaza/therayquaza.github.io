# Logistic Regression

(Heavily inspired by HOML)

Logistic regression is an adaptation of the [linear regression](https://therayquaza.github.io/machine_learning/linear/linear_regression.html) for binary classification tasks.

## How works a logistic regression ?

Logistic regression is different from linear regression and its regularization peers. The cost $J(\theta)$ function and the prediction $f_{\theta}$ function have been changed to fit classification tasks.

The prediction function is based on the probability of an obsvervation to belong to the positive class. We define the probability function as:

$$ \rho_{\theta}(x) = \sigma(\theta_0 + \sum_{i = 1}^n x_i \theta_i) $$

where $\sigma$ is defined as the sigmoid function:

$$ \sigma(x) = \frac{1}{1 + e^{-x}} $$

This prediction function is used for the final decision in the prediction function $f_{\theta}$ using a theshold (here 0.5):

$$ 
f_{\theta}(x) = \left\{\begin{matrix}
0, \rho_{\theta}(x) < 0.5 \\
1, \rho_{\theta}(x) \geq 0.5\\
\end{matrix}\right.
$$

Now to define the cost function $J(\theta)$, instead of using the prediction function to calculate the error like we did in [linear regression](https://therayquaza.github.io/machine_learning/linear/linear_regression.html), we use the probability function $\rho_{\theta}$. The loss function of the logistic regression is called [log-loss](https://therayquaza.github.io/fundamentals/metrics_and_losses.html) and is defined as:

$$
J(\theta) = - \frac{1}{N} \sum_{i = 1}^n y_i \log(\rho_{\theta}(x_i)) + (1 - y_i) \log(1 - \rho_{\theta}(x_i))
$$

This cost functions makes sense for our task:
- When target is positive ($(1 - y_i) = 0$), the cost of one obsveration becomes $ \log(\rho_{\theta}(x_i))$ which tends to 0 when $\rho_{\theta}(x_i)$ tends to 1
- When target is negative ($y_i = 0$), the cost of one obsveration becomes $ \log(1 - \rho_{\theta}(x_i))$ which tends to 0 when $\rho_{\theta}(x_i)$ tends to 0

The gradient can be calculated using the chain rule:

$$
\nabla_{\theta}J(\theta) = \frac{1}{N} (\rho_{\theta}(x) - y) \cdot x
$$

## Implementation in python

Here a naive implementation of the logistic regression using the classic gradient descent:

```python
class LogisticRegression:
    def __init__(self, lr=1e-3, it=10):
        self.it = it
        self.lr = lr

    def predict_proba(self, X):
        if self.weights.shape[0] != X.shape[1]:
            raise ValueError("Invalid shape between weights and features")
        return 1 / (1 + np.exp(- X @ self.weights))

    def fit(self, X, y):
        self.weights = np.random.randn(X.shape[1] + 1)
        X = np.hstack([X, np.ones((X.shape[0], 1))])
        for _ in range(self.it):
            self.weights -= (self.lr / X.shape[0]) * (self.predict_proba(X) - y) @ X
        return self

    def predict(self, X):
        return np.where(self.predict_proba(np.hstack([X, np.ones((X.shape[0], 1))])) > 0.5, 1, 0)
```

```{note}
Note the synthetic 1 feature added in the observation for the bias !
```

## Binary classification on banana quality

Using a dataset for binary classification from [Kaggle](https://www.kaggle.com/datasets/l3llff/banana) (amazing website by the way !), I trained the logistic model:

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

df = pd.read_csv("banana_quality.csv")

X = df[["Size","Weight","Sweetness","Softness","HarvestTime","Ripeness","Acidity"]].to_numpy()
y = np.where(df[["Quality"]].to_numpy() == "Bad", 0, 1).flatten()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

model = LogisticRegression(lr=1, it=30)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Accuracy = ", accuracy(y_test, y_pred))
>>> Accuracy =  0.88
```

```{figure} https://raw.githubusercontent.com/TheRayquaza/therayquaza.github.io/main/images/machine_learning/linear/Logistic_loss_GD.png
Loss on 30 iterations with 1 as learning rate
```

We get 87% accuracy with this naive model. Using other techniques like Mini Batch GD can help increase the accuracy of the model. I also found out that using a high learning rate performs better for my model.