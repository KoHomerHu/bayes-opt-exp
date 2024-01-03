import numpy as np
from scipy.stats import norm
from gaussian_process import GaussianProcess

"""
Since we only take 100 points, we do not need to 
compute the exact expected improvement function
or use the scipy.optimize package to find the maximum.
"""
def expected_improvement(mu, std, best, X = None, y = None, x_ast = None):
    lst = []
    for i in range(len(mu)):
        if std[i] != 0:
            z = (best - mu[i]) / std[i]
            D = mu[i] - best
            lst.append(max(D, 0) + std[i] * norm.pdf(z) - abs(D) * norm.cdf(z))
        else:
            lst.append(0)
    lst = np.array(lst)
    return np.argmax(lst), lst

"""
Similarly, we use simulation to compute the knowledge gradient.
This implementation is quite inefficient.
"""
def knowledge_gradient(mu, std, best, X = None, y = None, x_ast = None, J = 5):
    delta = np.zeros((J, len(x_ast)))
    for j in range(J):
        predicted_y = np.random.normal(mu, std)
        for i in range(len(x_ast)):
            gp = GaussianProcess(np.append(X, x_ast[i]), np.append(y, predicted_y[i]), x_ast)
            predicted_mu, _ = gp.conditional_dist()
            predicted_best = np.max(predicted_mu)
            delta[j][i] = predicted_best - best
    KG = np.mean(delta, axis = 0)
    return np.argmax(KG), KG

    
