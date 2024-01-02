import numpy as np

def cholesky_decomposition(A):
    L = np.zeros_like(A)
    for i in range(len(A)):
        for j in range(i + 1):
            if i == j:
                L[i, j] = np.sqrt(A[i, i] - np.sum(L[i, :] ** 2))
            else:
                L[i, j] = (A[i, j] - np.sum(L[i, :] * L[j, :])) / L[j, j]
    return L

def forward_substitution(L, b):
    n = len(b)
    y = np.zeros(n)
    for i in range(n):
        y[i] = (b[i] - np.dot(L[i, :i], y[:i])) / L[i, i]
    return y

def backward_substitution(R, b):
    n = len(b)
    x = np.zeros(n)
    for i in reversed(range(n)):
        x[i] = (b[i] - np.dot(R[i, i + 1:], x[i + 1:])) / R[i, i]
    return x

def cholesky_solve(L, b):
    x = np.zeros_like(b)
    if len(b.shape) == 1:
        y = forward_substitution(L, b)
        x = backward_substitution(L.T, y)
    else:
        for i in range(b.shape[1]):
            y = forward_substitution(L, b[:, i])
            x[:, i] = backward_substitution(L.T, y)
    return x