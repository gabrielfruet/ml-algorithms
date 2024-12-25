from collections.abc import Callable
from functools import partial
import jax
from abc import ABC
from jax import grad, vmap
from jax.typing import ArrayLike
import jax.numpy as jnp
from jax.numpy import ndarray
from tqdm import tqdm
import logging
import time

import os
os.environ['KERAS_BACKEND'] = 'jax'

import keras

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data(path="mnist.npz")

x_train = jax.device_put(jnp.array(x_train), jax.devices()[0])

jax.config.update("jax_traceback_filtering", "off")

logging.basicConfig()
logging.root.setLevel(logging.WARNING)
logger = logging.getLogger('jax_mlp2')
logger.setLevel(logging.INFO)

ActivationFunction = Callable[[ndarray], ndarray]

def seed_generator() -> Callable[[], int]:
    curr = 0
    def seed(curr=curr):
        curr += 1
        return curr

    return seed

seeder = seed_generator()

class Layer(ABC):
    def forward(self, x: ndarray) -> ndarray:
        raise NotImplementedError

    def backpropagate(self, delta: ndarray, learning_rate: float = 0.1) -> ndarray:
        raise NotImplementedError

class Activation(Layer):
    def __init__(self, function: ActivationFunction):
        self.fn: ActivationFunction = function
        self.dfn = vmap(vmap(grad(self.fn)))
        self.prev: ArrayLike | None = None

    def forward(self, x: ndarray) -> ndarray:
        self.prev = x
        return self.fn(x)

    def backpropagate(self, delta: ndarray, learning_rate: float = 0.1) -> ndarray:
        if self.prev is None:
            raise ValueError('No forward pass has been made yet')
        return delta * self.dfn(self.prev)

class SoftmaxActivation(Layer):
    def __init__(self):
        self.prev: ndarray | None = None
        self.activation = partial(jax.nn.softmax, axis=-1)

    def forward(self, x: ndarray) -> ndarray:
        self.prev = x
        return self.activation(x)

    def backpropagate(self, delta: ndarray, learning_rate: float = 0.1) -> ndarray:
        if self.prev is None:
            raise ValueError('No forward pass has been made yet')

        # The softmax activation has contribution of not only the xi 
        # but all the xj in the last layers, so when computing
        # the derivative of the loss with respect to xi we have to consider
        # the contributions of the xj to the process.

        # This is why we have to use the jacobian of the softmax function.

        # The new delta_i would be the summation with k varying of delta_k(i.e
        # error term on the k neuron) times the derivative of the softmax s_k
        # in relation to the x_i input(i.e how x_i affects s_k).

        _, vjp_fn = jax.vjp(self.activation, self.prev)
        return vjp_fn(delta)[0]

class Dense(Layer):
    def __init__(self, 
                 in_features: int,
                 out_features: int,
                 seed: int | None = None):
        if seed is None:
            seed = seeder()

        self.w = jax.random.normal(jax.random.PRNGKey(seed), (in_features, out_features)) * jnp.sqrt(2 / in_features)
        self.b = jnp.zeros((out_features,))
        self.prev: ArrayLike | None = None

    def forward(self, x: ndarray) -> ndarray:
        w, b = self.w, self.b
        self.prev = x
        activation = x @ w + b
        return activation

    def backpropagate(self, delta: ndarray, learning_rate: float = 0.1) -> ndarray:
        if self.prev is None:
            raise ValueError('No forward pass has been made yet')
        x = self.prev
        grad_w = jnp.transpose(x) @ delta

        logger.debug(f"{jnp.mean(grad_w, axis=-1).shape=}")
        logger.debug(f"{grad_w.shape=}")

        # Gradient centralization technique
        # grad_w = grad_w - jnp.mean(grad_w, axis=-1)[..., jnp.newaxis]

        grad_b = jnp.sum(delta, axis=0)
        self.w -= learning_rate * grad_w
        self.b -= learning_rate * grad_b
        return delta @ self.w.T

def mse_loss(y_true: ndarray, y_hat: ndarray) -> ArrayLike:
    return jnp.mean((y_true - y_hat) ** 2)

type LossFunction = Callable[[ndarray, ndarray], ArrayLike]

class MLP:
    def __init__(self, layers: list[Layer], loss: LossFunction = mse_loss):
        self.layers = layers
        self.loss = loss

    def __call__(self, x: ndarray) -> ndarray:
        return self.forward(x)

    def forward(self, x: ndarray) -> ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backpropagate(self,
                      y_true: ndarray,
                      y_hat: ndarray,
                      learning_rate: float = 0.1):

        delta = grad(self.loss)(y_hat, y_true)
        for layer in reversed(self.layers):
            delta = layer.backpropagate(delta, learning_rate)

    def train(self,
              x: ndarray,
              y: ndarray,
              x_valid: ndarray,
              y_valid: ndarray,
              learning_rate: float = 0.1):
        start = time.perf_counter()
        epochs = 20
        logger.info(f'{x.device=} {y.device=}')

        for epoch in range(epochs):
            loss_total = 0
            for xi, yi in tqdm(zip(x,y), total=len(x), bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'):
                y_hat = self.forward(xi)
                loss = self.loss(yi, y_hat)
                loss_total += loss
                logger.debug(f"Epoch {epoch}, Loss: {loss}")
                self.backpropagate(yi, y_hat, learning_rate)

            y_hat_valid = self.forward(x_valid)
            valid_loss = self.loss(y_valid, y_hat_valid)
            logger.info(f'Epoch {epoch}, Train Loss: {loss_total/x.shape[1]} Valid Loss: {valid_loss}')



        end = time.perf_counter()
        logger.info(f'Training took {end - start} seconds')

xor_dataset = [
    ([0, 0], 0),
    ([0, 1], 1),
    ([1, 0], 1),
    ([1, 1], 0)
]

x_xor_dataset = jnp.array([x for x, _ in xor_dataset], dtype=jnp.float32)
y_xor_dataset = jnp.array([y for _, y in xor_dataset], dtype=jnp.float32).reshape((-1, 1))

model1 = MLP([
    Dense(2, 10),
    Activation(jax.nn.sigmoid),
    Dense(10, 10),
    Activation(jax.nn.sigmoid),
    Dense(10, 10),
    Activation(jax.nn.sigmoid),
    Dense(10, 1),
    Activation(jax.nn.sigmoid),
])

model2 = MLP([
    Dense(784, 512),
    Activation(jax.nn.sigmoid),
    Dense(512, 64),
    Activation(jax.nn.sigmoid),
    Dense(64, 10),
    SoftmaxActivation(),
])

x_train_ok = x_train.reshape(200, 300, 28*28)
y_train_ok = jax.nn.one_hot(y_train, num_classes=10).reshape(200, 300, 10)
model2.train(x_train_ok, y_train_ok, x_test.reshape(-1, 28*28),
             jax.nn.one_hot(y_test, num_classes=10), learning_rate=0.5)

def compare_predictions(n: int):
    result = model2.forward(x_test.reshape((-1, 28*28)))
    print(jnp.argmax(result[:n], axis=-1))
    print(y_test[:n])

compare_predictions(10)
