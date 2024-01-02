# bayes-opt-exp

This is the implementation and visualisation of various techniques introduces in [A Tutorial on Bayesian Optimization](https://arxiv.org/abs/1807.02811).

## Gaussian process

We use the RBF kernel and constant mean. The hyperparameters (i.e. mean, sigma, alpha, and length) maximize the negative log-likelihood. 

Run ```gaussian_process.py```, use the sliders to change the alpha and length to see the impacts of these two hyperparameters.

## Bayesian optimisation

An animation to optimize a function with only one peak (but not only one global optimum) is demonstrated by running ```bayes_optimisation.py```.
