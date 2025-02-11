import jax.numpy as jnp
import jax
from jax import vmap
from functools import partial

def correlate(x,y=None, p=None, cov=False):
    if y is None:
        y = x

    n = x.shape[-1]

    if p is None:
        p = n


    assert len(x.shape) == len(y.shape), "X and Y must have the same shape"

    assert x.shape[-1] == y.shape[-1], "X and Y must have the same length"

    if p > n:
        raise ValueError("p must be less than or equal to the number of columns in X and Y")

    if len(x.shape) > 1:
        return jnp.mean(vmap(correlate, in_axes=(0,0,None))(x, y, p), axis=0)


    if cov:
        X = cov_lagged_matrix(x, p)
        Y = cov_lagged_matrix(y, p)

        return (Y.T @ X)/(n-p+1)
    else:
        X = lagged_matrix(x, p)
        Y = lagged_matrix(y, p)

        return (Y.T @ X)/n


def lagged_matrix(X, p=3):
    n = X.shape[0]
    zero_padded = jnp.zeros(n + p - 1)
    def get_lagged(i):
        return jax.lax.dynamic_update_slice(zero_padded, X, (i,))

    M = jax.vmap(get_lagged)(jnp.arange(p))

    return M.T

def cov_lagged_matrix(X, p=3):
    n = X.shape[0]
    indices = jnp.arange(p)

    def get_lagged(i):
        return jnp.flip(jax.lax.dynamic_slice(X, (p - i - 1,), (n-p,)))

    M = jax.vmap(get_lagged)(indices)
    return M.T

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    n_samples = 1000
    seed = jax.random.PRNGKey(42)
    base_signal = jax.random.normal(seed, (n_samples,))
    correlated_signal = 0.7 * base_signal \
        + 0.7 * jnp.roll(base_signal, 5)\
        + 0.5 * jnp.roll(base_signal, 4)\
        + 0.3 * jnp.roll(base_signal, 3)

    uncorrelated_signal = jax.random.normal(seed, (n_samples,))

    c1 = correlate(uncorrelated_signal, correlated_signal, 10, cov=True)
    plt.imshow(c1)
    plt.show()


