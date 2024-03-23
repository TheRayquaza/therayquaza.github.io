# Activations

Activation funtions in Neural Networks are special functions used to transform output of a layer using the input and weights associated to the layer. Activation functions are characterized by three aspects:
- non linearity
- range
- derivability

Depending on the task we want to achieve some activation functions may be more appropriate.

I will draw a non-exhaustive list of activation functions used and provide an example where using this activation function is appropriate.

## Sigmoid

Sigmoid is a well known activation function used to transform its input to a value included in $[0, 1]$:

$$ \sigma(x) = \frac{1}{1 + e^{-x}} $$

Its derivative is easy to calculate (a proof can be found [here](https://towardsdatascience.com/derivative-of-the-sigmoid-function-536880cf918e)):

$$ \frac{\partial \sigma(x)}{\partial x} = \sigma(x) (1 - \sigma(x)) $$

```{figure} https://raw.githubusercontent.com/TheRayquaza/therayquaza.github.io/main/images/fundamentals/activations/sigmoid.png
Sigmoid
```

Sigmoid can be used for binary classification where the output of the neural network is a probability between 0 and 1.
It suffers from the **vanishing gradient problem**. This problem causes the neural network to become harder to train: squishes output into a small range (here $[0, 1]$) reduces large change in the output and thus reducing the gradient as the change is propagating through the deep neural networks.

## Relu

Relu is another popular activation function fast to calculate and providing effective result. Relu solves the vanishing gradient problem because its output space is not restricted.

$$ Relu(x) = max(0, x) $$

Its derivative is not defined for $x = 0$ (the function is continuous but its derivative is not):

$$ \frac{\partial Relu(x)}{\partial x} = \left\{\begin{matrix} 
0, x < 0 \\
1, x > 0\\
\end{matrix}\right.
$$

```{figure} https://raw.githubusercontent.com/TheRayquaza/therayquaza.github.io/main/images/fundamentals/activations/relu.png
Relu
```

Relu is appropriate for most of the problem because it is fast but suffers from the **dead relu problem** which can occurs in some configuration (). Dead Relu outputs the same value (here 0 for negative values) which leads to value stuck at 0.

## Leaky Relu

Leaky relu addresses the issue of **dead relu** by allowing negative values with a very small coefficient applied.

$$ 
LRelu_{\alpha}(x) = \left\{\begin{matrix} 
\alpha x, x < 0\\
x, x \geq 0\\
\end{matrix}\right.
$$

$$ 
\frac{\partial LRelu_{\alpha}(x)}{\partial x} = \left\{\begin{matrix} 
\alpha, x < 0\\
1, x > 0\\
\end{matrix}\right.
$$

We usually set $\alpha = 0.01$.

```{figure} https://raw.githubusercontent.com/TheRayquaza/therayquaza.github.io/main/images/fundamentals/activations/leaky_relu.png
Leaky Relu
```

## Tanh

Tanh is a zero-centered activation functions. Its formula is given by

$$ Tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} $$

$$ \frac{\partial Tanh(x)}{\partial x} =  1 - Tanh(x)^2 $$


```{figure} https://raw.githubusercontent.com/TheRayquaza/therayquaza.github.io/main/images/fundamentals/activations/tanh.png
Tanh
```

Tanh performs really well in RNN and LSTM models due to its non linearity and its centered property. Tanh is however globally less used than Relu or sigmoid.

## Elu (Exponential Linear Units)

In 2015, Elu was presented in by Clevert et al. in their paper [Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)](https://arxiv.org/pdf/1511.07289v5.pdf)

Elu activation performs better generalization and leads to faster convergence.

$$ Elu_{\alpha}(x) = \left\{\begin{matrix} 
x, x > 0\\
\alpha (e^x -1), x \leq 0\\
\end{matrix}\right. $$

```{figure} https://raw.githubusercontent.com/TheRayquaza/therayquaza.github.io/main/images/fundamentals/activations/elu.png
Elu
```

## Gelu (Gaussian Error Linear Units)

Gelu is a new activation function proposed in 2016 by Dan Hendrycks & Kevin Gimpel in their paper [Gaussian Error Linear Units (GELUs)](https://arxiv.org/pdf/1606.08415v5.pdf) which performs really well and even better than Relu and Elu for large range of task.

Gelu definition can be confusing at the beginning, an approximation using $tanh$ has been given in the paper:

$$ Gelu(x) = \frac{x}{2} (1 + tanh(\sqrt(\frac{2}{\pi})(x + 044715x^3))) $$

Gelu is derivable everywhere unlike Relu and performs really well for image recognition and NLP tasks.

```{figure} https://raw.githubusercontent.com/TheRayquaza/therayquaza.github.io/main/images/fundamentals/activations/gelu.png
Gelu
```

## Softmax

Softmax is an activation function using multiple variables as input, it is a multiclass generalization of the **sigmoid** function:

$$ Softmax(x)_i = \frac{e^{x_i}}{\sum_{j = 1}^N e^{x_j}} $$

The particularity of the softmax function is that: $ \sum_{i = 1}^N Softmax(x)_i = 1 $

Softmax can be used in multi class prediction. It is usually used in the last layer to compute the probabilities of an obsevration to be of each classes.

## References
1. Deep Learning & Activations: Deep Learning by Ian Goodfellow and Yoshua Bengio and Aaron Courville, MIT Press
2. Activations: Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, 3rd Edition
3. Dying Relu [Stack Exchange](https://datascience.stackexchange.com/questions/5706/what-is-the-dying-relu-problem-in-neural-networks)
4. Vanishing gradient [Towards Data Science](https://towardsdatascience.com/the-vanishing-gradient-problem-69bf08b15484)
5. Elu: Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs) [arxiv](https://arxiv.org/pdf/1511.07289v5.pdf)
5. Gelu: Gaussian Error Linear Units (GELUs) [arxiv](https://arxiv.org/pdf/1606.08415v5.pdf)