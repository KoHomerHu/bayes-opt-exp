import numpy as np
from scipy import optimize
from scipy.stats import norm

"""
Since we only take 100 points, we do not need to 
compute the exact expected improvement function
or use the scipy.optimize package to find the maximum.
"""
def expected_improvement(mu, std, best):
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
    
