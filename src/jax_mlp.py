import jax
import jax.numpy as jnp
from jax import Array
from jax import grad, jit, vmap
import time
import logging

jax.config.update("jax_traceback_filtering", "off")


logging.basicConfig()
logging.root.setLevel(logging.WARNING)
logger = logging.getLogger('jax_mlp')
logger.setLevel(logging.INFO)

xor_dataset = [
    ([0,0], 0),
    ([0,1], 1),
    ([1,0], 1),
    ([1,1], 0)
]

and_dataset = [
    ([0,0], 0),
    ([0,1], 0),
    ([1,0], 0),
    ([1,1], 1)
]

x_and_dataset = jnp.array([x for x, _ in and_dataset]).reshape((-1, 2))
y_and_dataset = jnp.array([y for _, y in and_dataset]).reshape((-1, 1))

x_xor_dataset = jnp.array([x for x, _ in xor_dataset]).reshape((-1, 2))
y_xor_dataset = jnp.array([y for _, y in xor_dataset]).reshape((-1, 1))

def sigmoid(x: jnp.ndarray) -> jnp.ndarray:
    return 1 / (1 + jnp.exp(-x))

# Double vmap to map a NxM array element-wise differentiation
d_sigmoid = vmap(vmap(grad(sigmoid)))

def foward_pass(x: jnp.ndarray, w: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    logger.debug(f'{jnp.shape(w)=}')
    logger.debug(f'{jnp.shape(b)=}')
    return (x @ w) + b

def foward_propagation(x: jnp.ndarray,
                       w: list[jnp.ndarray],
                       b: list[jnp.ndarray]
                       ) -> tuple[list[jnp.ndarray], list[jnp.ndarray]]:
    # Store the activations of each neuron on each layer
    activations = list[Array]()
    # Store the activation after the activation function of each neuron on each layer
    activated = list[Array]()

    for wi, bi in zip(w,b):
        if not activations:
            # If we are on the first layer, we use the input as the x value
            neuron_activation = foward_pass(x, wi, bi)
        else:
            # Otherwise, we use the last layer activation
            neuron_activation = foward_pass(activated[-1], wi, bi)

        activations.append(neuron_activation)
        activated.append(sigmoid(neuron_activation))

    return activations, activated

def compute_error(y_true: jnp.ndarray, y_hat: jnp.ndarray) -> jnp.ndarray:
    return jnp.mean((y_true - y_hat) ** 2)

compute_error_grad = jax.grad(compute_error, argnums=1)

def backpropagation_pass(weights: list[jnp.ndarray],
                         biases: list[jnp.ndarray],
                         activations: list[jnp.ndarray],
                         activated: list[jnp.ndarray],
                         x_input: jnp.ndarray,
                         y_true: jnp.ndarray,
                         learning_rate: float):
    y_hat = activated[-1]
    logger.debug(f'{jnp.shape(y_hat)=}')
    logger.debug(f'{jnp.shape(y_true)=}')
    # Compute the error gradient.
    # The error gradient is the derivative of the error function with respect to the output of the last layer.
    # The error function is the mean squared error.
    # The derivative of the mean squared error is 2*(y_true - y_hat).
    # This yields a vector for each feature of the output, which is the gradient for
    # each feature of the output.
    delta = compute_error_grad(y_true, y_hat) * d_sigmoid(activations[-1])

    for i in reversed(range(len(weights))):
        logger.debug(f'iteration {i}')
        # If we are not on the first layer,
        # we need to use the previous activation as the delta
        if i > 0:
            prev = activated[i-1]
        else:
            prev = x_input

        logger.debug(f'Delta dimension. \n{jnp.shape(delta)=}')
        logger.debug(f'Prev dimension. \n{jnp.shape(prev)=}')
        
        # The gradient of the weights is the dot product of the previous
        # activation after the activation function and the delta
        # The delta is the error gradient of the current layer
        # Example: In the special case of the last layer, gradient is how much
        # each activation influeced in the error, this is shown by the delta,
        # that is the error gradient for each neuron based on its weights.
        grad_w = prev.T @ delta  
        logger.debug(f'Calculating weight gradient. \n{jnp.shape(grad_w)=}')


        # The gradient of the bias is the sum of the delta
        # This is becasue the bias is a scalar that is added to each neuron
        # So the gradient is the sum of the error gradient of each neuron
        grad_b = jnp.sum(delta, axis=0) 
        logger.debug(f'Calculating bias gradient. \n{jnp.shape(grad_w)=}')

        # Applying the gradient in reverse (i.e the positive gradient show
        # the direction the function GROWS, the negative gradient show the direction function
        # DECREASES.
        logger.debug(f'Applying weight gradient.')
        weights[i] -= learning_rate * grad_w

        logger.debug(f'Applying bias gradient.')
        biases[i] -= learning_rate * grad_b

        if i > 0:
            diff = d_sigmoid(activations[i - 1])
            logger.debug(f'{jnp.shape(diff)=}')
            # The delta of the next layer is the delta of the current layer
            # pondered by how much each neuron contributes to it, in reverse order.
            # So, thinking of an specific neuron i,
            # for each delta of the next layer, we multiply by the weight of the neuron_i
            # to the respective delta_j
            # This is done for all neurons in the next layer.
            # After that, we have to multiply by the derivative of the activation function.
            # This is due to the chain rule.
            delta = (delta @ weights[i].T)*diff
            logger.debug(f'Delta shape at {i}: {jnp.shape(delta)=}')

key = jax.random.key(1)

w: list[jnp.ndarray] = [
    jax.random.normal(key, (2, 10)),
    jax.random.normal(key, (10, 100)),
    jax.random.normal(key, (100,1))
]

b: list[jnp.ndarray] = [
    jnp.zeros(10),
    jnp.zeros(100),
    jnp.zeros(1),
]

BATCH_SIZE = 2_048

def training_loop():
    start = time.perf_counter()

    learning_rate = 1
    epochs = 1000

    repeater = lambda v: jnp.repeat(v, repeats=BATCH_SIZE//4, axis=0)

    x = repeater(x_and_dataset)
    y = repeater(y_and_dataset)
    logger.info(f'{x.device}')

    for epoch in range(epochs):
        activations, activated = foward_propagation(x, w, b)
        error = compute_error(y, activated[-1])
        logger.info(f'Current error: {error}')
        backpropagation_pass(w, b, activations, activated, x, y, learning_rate)

        if epoch % 100 == 0:
            logger.info(f"Epoch {epoch}, Error: {error}")

    end = time.perf_counter()
    logger.info(f'Training took {end-start} seconds')

training_loop()
# Final prediction
_, predictions = foward_propagation(x_and_dataset, w, b)
print("Predictions:")
print(jnp.round(predictions[-1]))
