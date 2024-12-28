import jax.numpy as jnp
import jax
from jax import grad, jit, jacobian, Array
from jax.typing import ArrayLike
from typing import Callable
import logging

logging.basicConfig()
logger = logging.getLogger('newton_raphson')
logger.setLevel(logging.DEBUG)

def newton_raphson(
        fun: Callable[[jnp.ndarray], Array],
        x_or_n: jnp.ndarray | int,
        epsilon: float = 1e-6, 
        maxiter: int = 10):
    x = None
    n = None

    if isinstance(x_or_n, jnp.ndarray):
        x = x_or_n
    if isinstance(x_or_n, int):
        n = x_or_n

    if x is None:
        x = jnp.zeros(n)

    gradient = grad(fun)
    hessian = jacobian(gradient)

    logger.debug(f'Initial guess: {x}')
    logger.debug(f'Gradient {jnp.linalg.norm(gradient(x))=}')

    for i in range(maxiter):
        h_val = hessian(x)
        g_val = gradient(x)
        logger.debug(f"{h_val=}")

        step = jnp.linalg.inv(h_val) @ g_val
        logger.debug(f"Step size {jnp.linalg.norm(step)=}")

        x = x - step

        logger.debug(f"Step size {jnp.linalg.norm(step)=}")
        if jnp.linalg.norm(step) < epsilon:
            logger.debug(f"Converged in {i + 1} iterations.")
            return x, True

    return x, False
