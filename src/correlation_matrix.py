import jax.numpy as jnp
import jax

def correlate(x,y, p=None):
    assert len(x.shape) == 1, "X must be a 1D array"
    assert len(y.shape) == 1, "Y must be a 1D array"

    n = x.shape[0]

    assert x.shape[0] == y.shape[0], "X and Y must have the same length"

    if p is None:
        p = n

    assert p <= n, "p must be less than the number of columns in X and Y"

    X = jnp.zeros((p,n))
    Y = jnp.zeros((p,n))

    for i in range(p):
        X = X.at[i].set(jnp.roll(x, shift=i))
        Y = Y.at[i].set(jnp.roll(y, shift=i))

    return (X @ Y.T) / n



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


