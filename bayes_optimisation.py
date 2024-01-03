import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from gaussian_process import GaussianProcess
from acquisition_function import expected_improvement, knowledge_gradient

def plot_gp(ax, gp, X, y, x_ast, f, new_x, new_y, af):
    x_disp = np.linspace(-5, 5, 50)
    y_disp = f(x_disp)

    mu, cov = gp.conditional_dist()
    std = np.sqrt(np.diag(cov))
    lower_bound = mu - 1.96 * std  # 95% confidence interval
    upper_bound = mu + 1.96 * std

    ax.clear()
    ax.plot(x_disp, np.zeros_like(x_disp), color='black')
    ax.plot(x_disp, y_disp, label="f(x)")
    ax.plot(X, y, 'o', label="data points")
    ax.plot(x_ast, mu, '--', label="predicted mean")
    if new_x is not None:
        ax.plot(new_x, new_y, 'o', color='red', label=" next point")
    ax.plot(x_ast, af, label=af_label, color='purple')
    ax.fill_between(x_ast, lower_bound, upper_bound, alpha=0.2)
    ax.legend()
    ax.set_ylim(-30, +30)

def update(frame):
    global X, y, x_ast, gp, ax

    mu, cov = gp.conditional_dist()
    std = np.sqrt(np.diag(cov))
    best = np.max(y)
    idx, af = acquisition_function(mu, std, best, X, y, x_ast)
    new_x = x_ast[idx]
    new_y = f(new_x)
    X = np.append(X, new_x)
    y = np.append(y, new_y)
    x_ast = np.linspace(-5, 5, 50)

    gp = GaussianProcess(X, y, x_ast)
    plot_gp(ax, gp, X, y, x_ast, f, new_x, new_y, af)

    if abs(new_y - best) < 1e-8:
        print("Optimisation stopped!!!")
        ani.event_source.stop()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--expected_improvement', '-ei', action='store_true', help='Use expected improvement as the acquisition function')
    parser.add_argument('--knowledge_gradient', '-kg', action='store_true', help='Use knowledge gradient as the acquisition function')
    args = parser.parse_args()
    if args.knowledge_gradient:
        acquisition_function = knowledge_gradient
        af_label = "knowledge gradient"
    else:
        acquisition_function = expected_improvement # default method is expected improvement
        af_label = "expected improvement"


    f = lambda x: (-(x - 2) ** 2 + 1) * np.cos(x)
    X = np.random.uniform(-5, 5, 1)  # initial data points
    y = f(X) + np.random.normal(0, 0.7, len(X))  # ground truth with sigma = 0.7
    x_ast = np.linspace(-5, 5, 50)

    gp = GaussianProcess(X, y, x_ast)
    mu, cov = gp.conditional_dist()
    std = np.sqrt(np.diag(cov))
    _, af = acquisition_function(mu, std, np.max(y), X, y, x_ast)

    fig, ax = plt.subplots()
    
    plot_gp(ax, gp, X, y, x_ast, f, None, None, af)

    ani = FuncAnimation(fig, func = update, frames=100, interval=1000)
    
    plt.show()
