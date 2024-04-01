# Naive Bayes

```{tableofcontents}
```

Naive Bayes regroups all supervised classification techniques based on the Bayes Theorem and the assumption that all features are independent of each other.

## Bayes Theorem

Bayes theorem is a fundamental theorem of conditional probability.
The fundamental idea behind bayes' theorem is: $ Posterior = \frac{Likelihood \times Prior}{Evidence} $

Given an independent feature vector $X=(x_1,x_2,...,x_n)$ and its class $y$, we have:

$$ P(y | X) = \frac{P(X|y) \times P(y)}{P(X)} $$

Because all features $x_i$ are independent to each other, we have:

$$
\begin{equation}
\begin{split}
P(X|y) &= P(x_1,...,x_n | y) \\
&= P(x_1|x_2,...,x_n,y) \times P(x_2|x_3,...,x_n,y) \times ... \times P(x_n|y) \\
&= P(x_1|y) \times P(x_2|y) \times ... \times P(x_n|y) \\
&= \prod_{i=1}^N P(x_i|y) \\
\end{split}
\end{equation}
$$

and the formula can be rewritten as:

$$ P(y | X) = \frac{\prod_{i=1}^n P(x_i|y) \times P(y)}{P(X)}  $$

## Pros & Cons

| Pros | Cons |
|------|------|
| Requires a small amount of training data, training is fast | Assumption of independent predictors/features, all attributes are mutually independent |
| Handles continuous and discrete data, not sensitive to irrelevant features | Zero Frequency problem |
| Simple, fast, and easy to implement | Not adapted to continuous features |
| Binary and multi-class classification ||
| Highly scalable ||

## Assumption

Naive bayes technique assumes that all features are independent of each other.

## References
1. Naive Bayes Algorithm [Medium](https://medium.com/analytics-vidhya/na%C3%AFve-bayes-algorithm-5bf31e9032a2)
2. Bayes' Theorem [Wikipedia](https://en.wikipedia.org/wiki/Bayes%27_theorem)
3. Great video on Bayes' theorem [3Blue1Brown Youtube](https://www.youtube.com/watch?v=HZGCoVF3YvM)
4. Multinomial bayes classification [Towards Data Science](https://towardsdatascience.com/multinomial-na%C3%AFve-bayes-for-documents-classification-and-natural-language-processing-nlp-e08cc848ce6)
5. NLP preprocessing [Medium](https://medium.com/@maleeshadesilva21/preprocessing-steps-for-natural-language-processing-nlp-a-beginners-guide-d6d9bf7689c9)
