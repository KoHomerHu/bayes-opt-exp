# bayes-opt-exp

This is the implementation and visualisation of various techniques introduces in [A Tutorial on Bayesian Optimization](https://arxiv.org/abs/1807.02811).

## Gaussian process

We use the RBF kernel and constant mean. The hyperparameters (i.e. mean, sigma, alpha, and length) maximize the negative log-likelihood. 

Run ```python gaussian_process.py```, use the sliders to change the alpha and length to see the impacts of these two hyperparameters.

## Bayesian optimisation

An animation to optimize a function with only one peak (but not only one global optimum) is demonstrated by running ```python bayes_optimisation.py```.

By default it uses the expected improvement as the acquisition function, which tends to exploit more, hence prone to get trapped in a local maxima. 

Use ```python bayes_optimisation.py -h``` argument to see what acquisition functions are implemented.

## TO-DO

- [ ] Implementation of more efficient knowledge gradient
- [ ] Implementation of predicted entropy search
