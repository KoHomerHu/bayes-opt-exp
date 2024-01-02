import numpy as np
from scipy import optimize

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

"""Gaussian Process with constant mean and RBF kernel"""
class GaussianProcess():
    def __init__(self, X, y, x_ast, mean = 0, 
                 alpha = 4.0, length = 1.0, sigma = 0.05):
        self.mean = mean
        self.alpha = alpha
        self.length = length
        self.sigma = sigma

        self.kernel = lambda x, y: self.alpha * np.exp(-np.sum((x - y) ** 2) / (2 * self.length ** 2))

        self.X = X
        self.y = y
        self.x_ast = x_ast

    def conditional_dist(self):
        A = np.zeros((len(self.X), len(self.X)))
        for i in range(len(self.X)):
            for j in range(len(self.X)):
                A[i, j] = self.kernel(self.X[i], self.X[j])
        A = A + self.sigma ** 2 * np.eye(len(self.X))
        K = np.zeros((len(self.x_ast), len(self.X)))
        for i in range(len(self.x_ast)):
            for j in range(len(self.X)):
                K[i, j] = self.kernel(self.x_ast[i], self.X[j])
        B = np.zeros((len(self.x_ast), len(self.x_ast)))
        for i in range(len(self.x_ast)):
            for j in range(len(self.x_ast)):
                B[i, j] = self.kernel(self.x_ast[i], self.x_ast[j])
        L = cholesky_decomposition(A)
        mu = self.mean + np.dot(K, cholesky_solve(L, self.y - self.mean))
        cov = B - np.dot(K, cholesky_solve(L, K.T))
        return mu, cov
    
    "Compute the negative log likelihood of the Gaussian Process ignoring the constant term"
    def neg_log_likelihood(self, pars):
        mean, alpha, length, sigma = pars
        kernel_theta = lambda x, y: alpha * np.exp(-np.sum((x - y) ** 2) / (2 * length ** 2))
        A = np.zeros((len(self.X), len(self.X)))
        for i in range(len(self.X)):
            for j in range(len(self.X)):
                A[i, j] = kernel_theta(self.X[i], self.X[j])
        A = A + sigma ** 2 * np.eye(len(self.X))
        L = cholesky_decomposition(A)
        return 0.5 * np.dot(y - mean, cholesky_solve(L, self.y - mean)) + np.sum(np.log(np.diag(L)))

    """Obtain the MLE estimation of the hyperparameters mean, alpha, length and sigma"""
    def get_hyperparameter(self):
        pars = np.array([self.mean, self.alpha, self.length, self.sigma])
        result = optimize.minimize(self.neg_log_likelihood, pars, method='Powell', 
                                   bounds = ((None, None), (1e-5, None), (1e-5, None), (1e-5, None)))
        return result.x


class GPPlot:
    def __init__(self, alpha, length, X, y, x_ast, f):
        self.X = X
        self.y = y
        self.x_ast = x_ast
        self.f = f
        self.alpha = alpha
        self.length = length

        self.fig, self.ax = plt.subplots()
        plt.subplots_adjust(bottom=0.25)
        self.alpha_slider_ax = plt.axes([0.25, 0.15, 0.65, 0.03])
        self.length_slider_ax = plt.axes([0.25, 0.1, 0.65, 0.03])

        self.alpha_slider = Slider(self.alpha_slider_ax, 'Alpha', 1e-5, 2 * alpha, valinit=self.alpha)
        self.length_slider = Slider(self.length_slider_ax, 'Length', 1e-5, 2 * length, valinit=self.length)

        self.alpha_slider.on_changed(self.update_alpha)
        self.length_slider.on_changed(self.update_length)

        self.update_plot()

    def update_alpha(self, val):
        self.alpha = val
        self.update_plot()

    def update_length(self, val):
        self.length = val
        self.update_plot()

    def update_plot(self, event=None):
        if event is None or event.name == 'button_release_event':
            gp = GaussianProcess(X = self.X, y = self.y, x_ast = self.x_ast, alpha=self.alpha, length=self.length)
            mu, cov = gp.conditional_dist()
            std = np.sqrt(np.diag(cov))
            lower_bound = mu - 1.96 * std # 95% confidence interval
            upper_bound = mu + 1.96 * std

            self.ax.clear()
            self.ax.plot(x_disp, y_disp, label="f(x)") 
            self.ax.plot(self.X, self.y, 'o', label="data points")
            self.ax.plot(self.x_ast, mu, label="Predicted mean")
            self.ax.fill_between(self.x_ast, lower_bound, upper_bound, alpha=0.2)
            self.ax.legend()
            self.ax.set_ylim(-30, +30)
            plt.draw()


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider

    f = lambda x: (-(x-2)**2 + 1) * np.cos(x) + 5 * np.sin(2*x)
    x_disp = np.linspace(-5, 5, 100)
    y_disp = f(x_disp)
    X = np.random.uniform(-5, 5, 5) # initial data points
    y = f(X) + np.random.normal(0, 0.7, len(X)) # ground truth sigma = 0.7
    x_ast = np.linspace(-5, 5, 100)

    gp = GaussianProcess(X, y, x_ast)
    mean, alpha, length, sigma = gp.get_hyperparameter()

    print("Initial hyperparameters:")
    print("mean = ", mean)
    print("alpha = ", alpha)
    print("length = ", length)
    print("sigma = ", sigma)

    gp_plot = GPPlot(alpha, length, X, y, x_ast, f)
    plt.show()

