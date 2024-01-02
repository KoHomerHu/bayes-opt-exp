import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from gaussian_process import GaussianProcess
from acquisition_function import expected_improvement

def plot_gp(ax, gp, X, y, x_ast, f, new_x, new_y):
    x_disp = np.linspace(-5, 5, 100)
    y_disp = f(x_disp)

    mu, cov = gp.conditional_dist()
    std = np.sqrt(np.diag(cov))
    lower_bound = mu - 1.96 * std  # 95% confidence interval
    upper_bound = mu + 1.96 * std

    ax.clear()
    ax.plot(x_disp, y_disp, label="f(x)")
    ax.plot(X, y, 'o', label="data points")
    ax.plot(x_ast, mu, '--', label="predicted mean")
    if new_x is not None:
        ax.plot(new_x, new_y, 'o', label=" next point")
    ax.fill_between(x_ast, lower_bound, upper_bound, alpha=0.2)
    ax.legend()
    ax.set_ylim(-30, +30)

def update(frame):
    global X, y, x_ast, gp, ax

    mu, cov = gp.conditional_dist()
    std = np.sqrt(np.diag(cov))
    best = np.max(y)
    idx, _ = expected_improvement(mu, std, best)
    new_x = x_ast[idx]
    new_y = f(new_x)
    X = np.append(X, new_x)
    y = np.append(y, new_y)
    x_ast = np.linspace(-5, 5, 100)

    gp = GaussianProcess(X, y, x_ast)
    print(X)
    plot_gp(ax, gp, X, y, x_ast, f, new_x, new_y)

if __name__ == '__main__':
    f = lambda x: (-(x - 2) ** 2 + 1) * np.cos(x)
    X = np.random.uniform(-5, 5, 1)  # initial data points
    y = f(X) + np.random.normal(0, 0.7, len(X))  # ground truth with sigma = 0.7
    x_ast = np.linspace(-5, 5, 100)

    gp = GaussianProcess(X, y, x_ast)

    fig, ax = plt.subplots()

    def init():
        plot_gp(ax, gp, X, y, x_ast, f, None, None)
        return ax,

    ani = FuncAnimation(fig, func = update, frames=100, interval=1000)
    
    plt.show()
