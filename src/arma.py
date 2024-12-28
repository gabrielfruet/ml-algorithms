import jax
from jax import Array, jit
import jax.numpy as jnp
from jax import lax
from newton_raphson import newton_raphson
from bfgs import bfgs
from least_squares import least_squares
import matplotlib.pyplot as plt

jax.config.update("jax_traceback_filtering", "off")


def arma(data, theta, phi, var, n, seed=1):
    """
    Simulates an ARMA process given parameters and initial data.
    
    Parameters:
    - data: jnp.ndarray, initial data to seed the simulation.
    - theta: jnp.ndarray, coefficients for the MA part.
    - phi: jnp.ndarray, coefficients for the AR part.
    - var: float, variance of the noise.
    - n: int, number of steps to simulate.
    - seed: int, random seed for reproducibility.

    Returns:
    - jnp.ndarray: Simulated ARMA process of length `n`.
    """
    key = jax.random.PRNGKey(seed)
    order_ar = phi.shape[0] 
    order_ma = theta.shape[0]  

    Xt = jnp.zeros(n + order_ar)  
    Xt = Xt.at[:order_ar].set(data[-order_ar:])  

    noise = jax.random.normal(key, (n + order_ma,)) * var

    def step(carry, i):
        Xt, noise = carry
        ar_terms = lax.dynamic_slice(Xt, (i,), (order_ar,))
        ma_terms = lax.dynamic_slice(noise, (i,), (order_ma,))
        
        xt = jnp.dot(phi, ar_terms) \
            + jnp.dot(theta, ma_terms) \
            + noise[i + order_ma]

        return (Xt.at[i + order_ar].set(xt), noise), None

    init = (Xt, noise)
    (Xt, _), _ = lax.scan(step, init, jnp.arange(0, n))

    return Xt[order_ar:]

@jit
def log_likelihood(residuals, var):
    n = residuals.shape[0]
    epsilon = 1e-8
    return -n/2 * jnp.log(2 * jnp.pi) \
        - n/2 * jnp.log(var + epsilon) \
        - 1/(2 * (var + epsilon)) * jnp.sum(residuals**2)

def arma_log_likelihood(data, p, q, seed=1):
    def fn(params, p=p, q=q, data=data):
        var = params[0]
        theta = params[1:q+1]
        phi = params[q+1:q+1+p]
        prediction = arma(data[-p:], theta, phi, var, len(data) - p, seed)
        return -log_likelihood(prediction - data[len(phi):], var)
    return fn

def fit_arma_ols(data, p, q, iter, tol=1e-9):
    n = len(data)
    residuals = jnp.zeros(data.shape[0] - p)

    coef: None | jnp.ndarray = None

    for i in range(iter):
        x = []
        y = []
        for lag in range(p):
            x.append(data[lag:n-p+lag])

        y.append(data[p:])

        for lag in range(q):
            x.append(residuals)

        x = jnp.array(x).T
        y = jnp.squeeze(jnp.array(y))

        coef = least_squares(x,y)
        new_residuals = y - jnp.dot(x, coef)
    
        if jnp.allclose(new_residuals, residuals, atol=tol):
            print("Converged at iteration", i)
            print("Diff between residuals and new_residuals", jnp.sum(jnp.abs(new_residuals - residuals)))
            break 

        residuals = new_residuals

    if coef is None:
        raise ValueError("iter should be greater than 0")
    print(f'{coef=}')

    return coef[:p], coef[p:], jnp.sum(residuals ** 2) / (n - p - q)

phi = jnp.array([0.3, 0.2, 0.1])
theta = jnp.array([0.5, 0.5])
var = 1
arma_seed = jax.random.normal(jax.random.PRNGKey(1), (len(phi), )) * var

data = arma(arma_seed, phi, theta, var, 10_000)
plt.hist(data, bins=50)
plt.show()

theta, phi, var = fit_arma_ols(data,len(phi), len(theta), 10)
print(f'{theta=}',f'{phi=}', f'{var=}')


