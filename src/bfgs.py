import jax
from jax import vjp, jacobian, grad, jit, Array
from jax.typing import ArrayLike
import jax.numpy as jnp
from typing import Callable
import logging


logging.basicConfig()
logger = logging.getLogger('bfgs')
logger.setLevel(logging.DEBUG)

def bfgs(fun: Callable[[jnp.ndarray], Array],
                 num_parameters: int,
                 epsilon: float = 1e-1,
                 maxiter = 10,
                 seed: int = 1):
    x = jax.random.normal(jax.random.PRNGKey(seed), (num_parameters,))
    x /= jnp.linalg.norm(x)
    B_inv = jnp.eye(num_parameters, num_parameters)  # Start with identity as the inverse
    c1 = 1e-4
    c2 = 0.9
    dfun = grad(fun)
    iter = 0

    while jnp.linalg.norm(dfun(x)) > epsilon and iter < maxiter:
        iter += 1
        logger.debug(f'{jnp.linalg.norm(dfun(x))=}')
        p = -B_inv @ dfun(x)  # Use inverse directly for direction

        # Line search to find step size
        a = 1
        logger.debug('Line search starting')
        while fun(x + a * p) > fun(x) + c1 * a * p.T @ dfun(x) or \
              -p.T @ dfun(x + a * p) > -c2 * p.T @ dfun(x):
            logger.debug(f'{a=}')
            a *= 0.5
            if a < 1e-8:
                raise ValueError('Line search failed, cannot proceed. Try to lower maxiter')

        logger.debug(f'Line search satisfied at {a=}')
        
        # Update step
        s = a * p
        xn = x + s
        y = dfun(xn) - dfun(x)
        logger.debug(f'{x=}')

        # Safeguard for y.T @ s
        ys = y.T @ s
        if ys <= 1e-8:
            raise ValueError('y.T @ s is too small, cannot procced. Try to lower maxiter')
        else:
            rho = 1 / ys
            B_inv_y = B_inv @ y
            B_inv = B_inv + rho * jnp.outer(s, s) - rho * jnp.outer(B_inv_y, B_inv_y) / (y.T @ B_inv_y)
        
        logger.debug(f'{B_inv=}')
        logger.debug(f'{xn=}')
        x = xn  # Update x to the new point

    logger.debug(f'Final gradient {jnp.linalg.norm(dfun(x))=}')

    return x
