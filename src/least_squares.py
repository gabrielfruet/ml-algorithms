import jax
import jax.numpy as jnp

def least_squares(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    x_pinv = jnp.linalg.pinv(x)
    return x_pinv @ y

