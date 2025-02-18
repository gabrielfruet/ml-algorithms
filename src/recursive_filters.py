import jax
import tqdm
from jax import errors, numpy as jnp, vmap
from matplotlib.axes import Axes
from wiener import wiener_apply, wiener_filter_inputs_sampling
import matplotlib
from matplotlib import pyplot as plt
import scienceplots

@jax.jit
def corr(x, y):
    return x.T @ y / len(x)

def steepest_descent(w, x, d, lr=0.01):
    p_xd = corr(x, d)
    Rx = corr(x, x)
    wn = w - 2*lr*(Rx @ w - p_xd)
    return wn

def newton_algorithm(w, x, d, lr=0.01):
    p_xd = corr(x, d)
    Rx = corr(x, x)
    wn = w - lr * jnp.linalg.solve(Rx, Rx @ w - p_xd)
    return wn

def stocastic_gradient_descent(w, xi, di, lr=0.01):
    e = di - xi @ w
    wn = w + 2*lr*xi*e
    return wn

def normalized_lms(w, xi, di, lr=0.01):
    e = di - xi @ w
    μ = lr/(xi@xi.T + 1e-1)
    wn = w + 2*μ*xi*e
    return wn

def moving_average(x,w):
    return jnp.convolve(x, jnp.ones(w), mode='valid')/w



# %
if __name__ == '__main__':
    plt.style.use('science')
    matplotlib.rcParams.update({'font.size': 18})

    key = jax.random.PRNGKey(0)
    noise = jax.random.normal(key, (10_001,))
    d = noise[:-1]
    x = noise[1:] + 1.6 * noise[:-1]
    x_train, d_train = wiener_filter_inputs_sampling(x,d, filter_size=2)

    w_opt = newton_algorithm(jnp.zeros((2,)), x_train, d_train, lr=1.)
    d_hat_opt = wiener_apply(x, w_opt)

    def param_error(w, w_opt=w_opt):
        return jnp.mean((w_opt - w)**2)

    def steepest_descent_training(lr=0.2, w0=None, iterations=100, x=x, d=d, x_train=x_train, d_train=d_train):
        if w0 is None:
            w0 = jnp.zeros((2,))

        w = w0
        W = [w]
        errors = []
        with tqdm.tqdm(total=iterations) as pbar:
            for _ in range(iterations):
                wn = steepest_descent(w, x_train, d_train, lr=lr)
                d_hat = wiener_apply(x, wn)
                error = jnp.mean((d_hat - d)**2)
                pbar.set_postfix(error=error)
                pbar.update(1)
                w = wn
                W += [w]
                errors += [error]


        return jnp.array(W), jnp.array(errors)

    def newton_algorithm_training(lr=0.2, w0=None, iterations=100, x=x, d=d, x_train=x_train, d_train=d_train):
        if w0 is None:
            w0 = jnp.zeros((2,))

        w = w0
        W = [w]
        errors = []
        with tqdm.tqdm(total=iterations) as pbar:
            for _ in range(iterations):
                wn = newton_algorithm(w, x_train, d_train, lr=lr)
                d_hat = wiener_apply(x, wn)
                error = jnp.mean((d_hat - d)**2)
                pbar.set_postfix(error=error)
                pbar.update(1)
                w = wn
                W += [w]
                errors += [error]

        return jnp.array(W), jnp.array(errors)

    def stocastic_gradient_descent_training(lr=0.2, w0=None, x=x, d=d, x_train=x_train, d_train=d_train):
        if w0 is None:
            w0 = jnp.zeros((2,))

        w = w0
        W = [w]
        errors = []
        with tqdm.tqdm(total=x_train.shape[0]) as pbar:
            for xi, di in zip(x_train, d_train):
                wn = stocastic_gradient_descent(w, xi, di, lr=lr)
                d_hat = wiener_apply(x, wn)
                error = jnp.mean((d_hat - d)**2)
                pbar.set_postfix(error=error)
                pbar.update(1)
                w = wn
                W += [w]
                errors += [error]

        return jnp.array(W), jnp.array(errors)


    def normalized_lms_training(lr=0.01, w0=None, x=x, d=d, x_train=x_train, d_train=d_train):
        if w0 is None:
            w0 = jnp.zeros((2,))

        w = w0
        W = [w]
        errors = []
        with tqdm.tqdm(total=x_train.shape[0]) as pbar:
            for xi, di in zip(x_train, d_train):
                wn = normalized_lms(w, xi, di, lr=lr)
                d_hat = wiener_apply(xi, wn)
                error = jnp.mean((w_opt - wn)**2)
                pbar.set_postfix(error=error)
                pbar.update(1)
                w = wn
                W += [w]
                errors += [error]

        return jnp.array(W), jnp.array(errors)



    def plot_descent(W, ax: Axes, w_opt=w_opt):
        w_opt0, w_opt1 = w_opt
        x,y = jnp.meshgrid(jnp.linspace(w_opt0-1, w_opt0+1, 100), jnp.linspace(w_opt1-1, w_opt1+1, 100))
        xy = jnp.stack([x, y]).T
        z = vmap(vmap(param_error))(xy)
        # cs = ax.contourf(x, y, z, cmap="plasma", levels=15, alpha=0.75)
        # ax.contour(cs)
        ax.contour(x,y,z, levels=15, colors='tab:blue')
        ax.scatter(w_opt0, w_opt1, color='tab:red', label='$w_{opt}$', alpha=0.75)
        Wx = W[:,0]
        Wy = W[:,1]
        ax.plot(Wx,Wy, label='Iterations of $w$')

    W_sd, errors = steepest_descent_training(lr=0.02)
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    plot_descent(W_sd, ax[0])
    ax[1].plot(errors)
    fig.legend()
    fig.show()

    W_na, errors = newton_algorithm_training(lr=0.1)
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    plot_descent(W_na, ax[0])
    ax[1].plot(errors)
    fig.legend()
    fig.show()

    W_sg, errors_sg = stocastic_gradient_descent_training(lr=.001)
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    plot_descent(W_sg, ax[0])
    ax[1].plot(errors_sg)
    fig.legend()
    fig.show()

    W_nl, errors_nl = normalized_lms_training(lr=0.001)
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    plot_descent(W_nl, ax[0])
    ax[1].plot(errors_nl)
    fig.legend()
    fig.show()

    plt.plot(x, label='$x$')
    plt.plot(d, label='$d$')
    plt.plot(d_hat, label='$\\hat{d}$')
    plt.legend()
    plt.show()
