import jax.numpy as jnp
import jax
from jax import vmap

def correlate(x,y=None, p=None, cov=False):
    if y == None:
        y = x
    
    n = x.shape[0]

    assert len(x.shape) == len(y.shape), "X and Y must have the same shape"

    if len(x.shape) > 1:
        return jnp.mean(vmap(correlate, in_axes=(0,0,None))(x, y, p), axis=0)

    assert x.shape[0] == y.shape[0], "X and Y must have the same length"

    if p is None:
        p = n

    assert p <= n, "p must be less than or equal to the number of columns in X\
    and Y"

    X = lagged_matrix(x - x.mean(), p)
    Y = lagged_matrix(y - y.mean(), p)

    if cov:
        return (Y.T @ X)/(n-p+1)
    return (Y.T @ X)/n

def lagged_matrix(X,p=3):
    n = len(X)
    M = jnp.zeros((p,n+p-1))

    for i in range(p):
        M = M.at[i, i:n+i].set(X)

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

    c1 = correlate(uncorrelated_signal, correlated_signal, 10)
    plt.imshow(c1)
    plt.show()


