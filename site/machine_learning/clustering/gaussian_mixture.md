# Gaussian Mixture
![Clustering](https://raw.githubusercontent.com/TheRayquaza/therayquaza.github.io/main/images/badges/Clustering.svg)

Gaussian mixture (GM) is an interesting distribution based technique that uses gaussian distributions (normal distribution) to represent each clusters. It is based on probability and tries to identify to which distribution each observations should be assigned to using probabilities.

## Some probabilistic math

This section is focused on probabilistic math to provide all required math concepts to describe the mechanism behind GM. I will dive really into the subject: I will just describe roughly the basic to understand how works Gaussian Mixture.

For a more detailled description on the topic, you can follow those links:
- Multivaraite probability distributions: [Brown University](https://www.dam.brown.edu/people/huiwang/classes/am165/Prob_ch5_2007.pdf)
- Multivariate gaussians: [CS 229 - Standford](https://cs229.stanford.edu/section/gaussians.pdf)

```{note}
**Gaussian** is stricltly equivalent to **Normal** in the context of probability distribution
```

### Random Variable

A random variable $X$ is simply a mapping from the sample space $\Omega$ to a measurable space $E$ (thus $X(\Omega) = E$). In other words, $X$ maps each outcome to a value (typically $\mathbb{R}$).

$$ X: \Omega \longrightarrow E $$

The expected value of a random variable $\mathbb{E}(X)$ is the arithmetic mean of the outcomes of $X$.

| Discrete | Continuous |
|----------|-------------|
| $ \mathbb{E}(X) = \sum_{x_i \in E} x_iP(X = x_i) $ | $ \mathbb{E}(X) = \int_{-\infty}^{+\infty} xf_X(x) dx $ |

### Probability Distribution

A distribution is a function that gives the probability of an **event** to happen given an **experiment**. ([wikipedia](https://en.wikipedia.org/wiki/Probability_distribution)).
Here is a list of popular probability function used to modelize problems: [wikipedia list](https://en.wikipedia.org/wiki/List_of_probability_distributions)

### PDF & PMF

PDF (probability density function) and PMF (probability density function) are similar notions denoted as $f_X$.

They describe *the behavior* of the distribution for different kind of random variables. More exactly, they provide the relative probability that the value of a random variable would be equal to that sample.

[PDF](https://en.wikipedia.org/wiki/Probability_density_function) and [PMF](https://en.wikipedia.org/wiki/Probability_mass_function) provide the likelihood that the value of a random variable is equal to a sample.

PMF is used for discrete random variables while PDF is its continuous *version*.

### CDF

[CDF](https://en.wikipedia.org/wiki/Cumulative_distribution_function) (Cumulative distribution function $F_X$) gives the probability of an event to occur whithin a range of values.

| Discrete | Continuous |
|----------|-------------|
| $ F_X(x) = \sum_{t \leq x} f_X(t) $ | $ F_X(x) = \int_{-\infty}^x f_X(t) dt $ |

We can then calculate the probability that X is whithin a range using the CDF:

$$ \forall x \in E,P(X \leq x) = F_X(x) $$

### Univariate Gaussian distribution

A gaussian distribution (also called normal distribution) is a popular distribution with characterized by its probability density function known as the **bell-curve**:

$$ \forall x \in \mathbb{R}, f_X(x) = \frac{exp(-\frac{1}{2}(\frac{x - \mu}{\sigma})^2)}{\sigma \sqrt{2\pi}} $$

The parameters are:
- $\mu$: the expectation value ($\mathbb{E}(X)$)
- $\sigma$: the standard deviation ($\sigma(X) = \sqrt{Var(X)} = \sqrt{\mathbb{E}(X^2) - \mathbb{E}(X)^2}$)

```{figure} https://raw.githubusercontent.com/TheRayquaza/therayquaza.github.io/main/images/machine_learning/clustering/gaussian.png
A pretty gaussian $\mathcal{N}(10, 2)$
```

```{note}
$\mu$ indicates the center of the distribution and $\sigma$ indicates how spread the data are from the mean.
```

### Generaliaztion to multivariate distributions

The generalization of distribution is multivariate distribution. Multivariate distribution is a distribution involving more than one random variable: we use **random vector** (a list of random variables).

```{figure} https://raw.githubusercontent.com/TheRayquaza/therayquaza.github.io/main/images/machine_learning/clustering/Multivariate_normal_sample.svg
Multivariate normal PDF on 2 random variables from wikipedia
```

A multivariate distribution represents a distribution involving multiple random variables. Each random variable has (potentially) its own set of parameters within the same distribution family.

We can define the covariance matrix $\Sigma$ and the mean vector $M$ of a multivariate gaussian distrubtion as we did earlier for the univariate gaussian distribution.

Given a random vector $X = [X_0, ..., X_n]$, we can say that $X$ follows a gaussian distribution $\mathcal{N}(M, \Sigma)$: 

$$ X \sim \mathcal{N}(M, \Sigma) $$

where 

$$ M_i = \mathbb{E}(X_i) $$
$$ \Sigma_{i,j} = cov(X_i, X_j) = \mathbb{E}(X_iX_j) - \mathbb{E}(X_i)\mathbb{E}(X_j) $$

The PDF function of a multivariate gaussian distribution $ \mathcal{N}(M, \Sigma) $ with $k$ random variables is given by:

$$ \forall x \in \mathbb{R}^k, f_X(x) = \frac{exp(-\frac{1}{2}(x-M)^T\Sigma^{-1}(x-M))}{\sqrt{det(\Sigma) (2\pi)^k}} $$

## How works Gaussian Mixture ?

Now we have defined everything needed, we can describe the mechanism of Gaussian Mixture.

Gaussian mixture uses a multivariate gaussian distribution where each cluster will have its own multivariate gaussian distribution with its specific parameters. Each cluster has its own random vector where each random variable associate one feature.

### Problem

Let's denote:
- $N$ the number of obsevrations
- $d$ the number of features
- $K$ the number of clusters given by the user

In a GMM of $K$ clusters with $N$ observations $X$, we want to maximize $P(X|\pi,M,\Sigma)$ where 
- **$\pi$**: vector of **mixture** coefficient speciying the probability of each cluster being chosen when assigning a sample (shape is $K$)
- **$M$**: matrix where each column represents the expected value $\mathbb{E}(X)$ of each clusters (shape is $K \times d$)
- **$\Sigma$**: list of covariance matrix where each covariance matrix is associated to a cluster (shape is $K \times d \times d$)

In other words, GMM tries to find the best PDF functions $p_k$ and $\pi$ for $K$ clusters (with $X_k \sim \mathcal{N}(M_k, \Sigma_k)$ a random vector of $d$ random variables) to maximize:

$$ P(X|\pi,M,\Sigma) = \prod_{n = 1}^N[\sum_{k=1}^K \pi_k p_k(X_n)]$$

```{note}
The $p_k$ function is nothing but the PDF function ($f_X$) of a multivariate gaussian distribution with parameters $M_k$ and $\Sigma_k$
```

### Expectation-Maximization (EM) algorithm

One way to solve this maximization problem is by using the EM algorithm.
EM algorithm, as described in his name, has two steps: 
- The model calculates the expected value of the log-likelihood function given the current parameters: **E-step**
- The parameters are updated to maximize the log-likelihood function: **M-step**

We repeat these steps until convergence.

#### Parameter initalization

We intialize our parameters:
- The mixture coefficients are set uniformly: $ \pi_k = \frac{1}{K} $
- The mean parameter $M$ of each distributions can be defined by the mean on each features of the observations
- There are different strategies to compute the first $\Sigma$. Each of those strategies assign the same $\Sigma$ on all distributions
    * Full covariance matrix: compute the whole covariance matrix of the dataset 
    * Diagonal covariance matrix: compute the variance of observations on each features

#### E-Step

During the E-step we compute parameters $\gamma_{kn}$ for each clusters and each values. These gammas values represent the probability that the $X_i$ observation belongs to cluster $k$.
$\gamma_{kn}$ can be found using with the **bayes theorem**:

$$ \gamma_{ki} = P(X_i \in C_k) = \frac{\pi_k p_k(X_i)}{\sum_{j=1}^K \pi_j p_j(X_i)} $$

These gammas will be used in the M-step to re-calculate the parameters.

#### M-Step

We can then update the parameters of each gaussian distributions using the gammas:

- $\pi_k$ becomes the average sum of probabilities that a point belongs to cluster $k$

$$ \pi_k = \frac{1}{N} \sum_{i=1}^N \gamma_{ki} $$

- $M_k$ becomes the weighted average of all observations with weight on cluster $k$

$$ M_k = \frac{\sum_{i = 1}^N \gamma_{ki} X_i}{\sum_{i = 1}^N \gamma_{ki}}$$

- $\Sigma_k$ becomes the weighted variance of all observations with weight on cluster $k$

$$ \Sigma_k = \frac{\sum_{i = 1}^N \gamma_{ki} (X_i - M_k)(X_i - M_k)^T}{\sum_{i=1}^N \gamma_{ki}} $$

## Assumption

GM assumes that the best clusters are distributed in multivariate gaussian distributions.

## Implementation

Here an implementation of a gaussian mixture model using numpy. This code is far from perfect and performs sometimes badly due to the randomized intialization.

```python
import numpy as np
from scipy.stats import multivariate_normal

class GMM:
    
    def __init__(self, k=2, it=100):
        self.k = k
        self.it = it

    def __m_step(self, X):
        scalars = np.sum(self.g, axis=1)
        
        self.weights_ = scalars / self.k
        self.means_ = ((self.g @ X).T / scalars).T

        for i in range(self.k):
            self.covariances_[i] = np.sum(self.g[i, :, None] * np.square(X - self.means_[i])) / scalars[i]
           
    def __e_step(self, X):
        for i in range(self.k):
            self.g[i] = multivariate_normal.pdf(X, mean=self.means_[i], cov=self.covariances_[i])

    def fit(self, X):
        self.N, self.d = X.shape
        self.weights_ = np.ones((self.k)) / self.k
        self.means_ = np.zeros((self.k, self.d))
        for i in range(self.k):
            self.means_[i] = X[np.random.choice(self.N, size=1)]

        if self.d != 1:
            self.covariances_ = np.transpose(np.cov(X, rowvar=False)[:,:,None] * np.ones(self.d, self.d, self.k), axes=(2, 0, 1))
        else:
            self.covariances_ = np.cov(X, rowvar=False) * np.ones((self.k, self.d, self.d))

        self.g = np.zeros((self.k, self.N))
        
        for it in range(self.it):
            if self.verbose:
                print("######### step", it)
            self.__e_step(X)
            self.__m_step(X)
        return self
```

I struggled a lot when I tried to create this model. 
- One issue was computing the covariance matrices made them invalid (either not positive or not symetric).
- Another issue was the *underflow* problem: it appears that computing the product of probabilities (< 1) creates 0 values as iterations continue. It led to the degradation of the parameters until **NaN** values occur.

## Experimenting GMM

Now we can experiment the GMM on two distributions, here: $\mathcal{N}(2, 1)$ and $\mathcal{N}(-2, 1)$

```python
import matplotlib.pyplot as plt

N = 100
k = 2
d = 1
X1 = np.random.multivariate_normal(np.array([2]),  np.array([[1]]), N//2)
X2 = np.random.multivariate_normal(np.array([-2]),  np.array([[1]]), N//2)
X = np.concatenate((X1, X2))
y = np.concatenate((np.zeros(N//2), np.ones(N//2)))

model = GMM(k=2, it=10).fit(X)

print(model.covariances_)
print(model.means_)
print(model.weight_)
```

We can retrieve the covariance and mean matrices from the model.
Finally, we can plot the predicted gaussians generated by our model and compare it to the expected gaussians $\mathcal{N}(2, 1)$ and $\mathcal{N}(-2, 1)$:

```python
from scipy.stats import norm

plt.figure(figsize=(8, 6))
for i in range(k):
    X_cluster = X[y == i]
    plt.hist(X_cluster, bins=30, density=True, alpha=0.5, label=f'Cluster {i}', color=['red', 'blue'][i])

for i in range(k):
    mu = model.means_[i]
    sigma = np.sqrt(np.diag(model.covariances_[i]))
    x = np.linspace(-6, 6, 1000)
    plt.plot(x, norm.pdf(x, mu, sigma), label=f'GMM Component {i}', linestyle='--')

plt.xlabel('X')
plt.ylabel('Density')
plt.title('Original Gaussians vs GMM Components')
plt.legend()
plt.grid(True)
plt.show()
```

```{figure} https://raw.githubusercontent.com/TheRayquaza/therayquaza.github.io/main/images/machine_learning/clustering/gaussian_mixture_estimation.png
GMM Components
```