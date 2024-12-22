import numpy as np
import numpy.typing as npt
import logging

logger = logging.getLogger('mlp')
logger.setLevel(logging.DEBUG)

and_dataset = [
    ([0,0], 0),
    ([0,1], 0),
    ([1,0], 0),
    ([1,1], 1)
]

x_and_dataset = np.array([x for x, _ in and_dataset]).reshape((-1, 2))
print(x_and_dataset)
y_and_dataset = np.array([y for _, y in and_dataset]).reshape((-1, 1))

type FArray = npt.NDArray[np.floating]

def sigmoid(x: FArray) -> FArray:
    return 1 / (1 + np.exp(-x))

def d_sigmoid(x: FArray) -> FArray:
    s = sigmoid(x)
    return s * (1 - s)

def foward_pass(x: FArray, w: FArray, b: FArray) -> FArray:
    logger.debug(f'{w.shape=}')
    logger.debug(f'{b.shape=}')
    return (x @ w) + b

def foward_propagation(x: FArray, w: list[FArray], b: list[FArray]) -> tuple[list[FArray], list[FArray]]:
    activations = list[FArray]()
    activated = list[FArray]()

    for wi, bi in zip(w,b):
        if not activations:
            neuron_activation = foward_pass(x, wi, bi)
        else:
            neuron_activation = foward_pass(activated[-1], wi, bi)

        activations.append(neuron_activation)
        activated.append(sigmoid(neuron_activation))

    return activations, activated

def compute_error(y: FArray, y_hat: FArray) -> np.floating:
    return np.mean((y - y_hat) ** 2)

def error_contrib(y_true: FArray, y_hat: FArray) -> FArray:
    return 2*(y_hat - y_true)

def backpropagation_pass(weights: list[FArray],
                         biases: list[FArray],
                         activations: list[FArray],
                         activated: list[FArray],
                         x_input: FArray,
                         y_true: FArray,
                         learning_rate: float):
    y_hat = activated[-1]
    logger.debug(f'{y_hat.shape=}')
    logger.debug(f'{y_true.shape=}')
    delta = error_contrib(y_true, y_hat) * d_sigmoid(activations[-1])

    for i in reversed(range(len(weights))):
        logger.debug(f'iteration {i}')
        if i > 0:
            prev = activations[i-1]
        else:
            prev = x_and_dataset

        logger.debug(f'Delta dimension. \n{delta.shape=}')
        
        grad_w = prev.T @ delta  
        logger.debug(f'Calculating weight gradient. \n{grad_w.shape=}')


        grad_b = np.sum(delta, axis=0) 
        logger.debug(f'Calculating bias gradient. \n{grad_w.shape=}')


        logger.debug(f'Applying weight gradient.')
        weights[i] -= learning_rate * grad_w

        logger.debug(f'Applying bias gradient.')
        biases[i] -= learning_rate * grad_b

        if i > 0:
            diff = d_sigmoid(activations[i - 1])
            logger.debug(f'{diff.shape=}')
            delta = (delta @ weights[i].T)*diff
            logger.debug(f'Delta shape at {i}: {delta.shape=}')

    

w: list[FArray] = [
    np.random.randn(20).reshape((2, 10)),
    np.random.randn(100).reshape((10, 10)),
    np.random.randn(100).reshape((10, 10)),
    np.random.randn(10).reshape((10, 1))
]
b: list[FArray] = [
    np.random.rand(10).reshape((-1)),
    np.random.rand(10).reshape((-1)),
    np.random.rand(10).reshape((-1)),
    np.random.randn(1).reshape((-1))
]


def training_loop():
    learning_rate = 0.1
    epochs = 10_000

    for epoch in range(epochs):
        activations, activated = foward_propagation(x_and_dataset, w, b)
        error = compute_error(y_and_dataset, activated[-1])
        print(error)
        backpropagation_pass(w, b, activations, activated, x_and_dataset, y_and_dataset, learning_rate)

        if epoch % 100 == 0:
            logger.info(f"Epoch {epoch}, Error: {error}")

training_loop()
# Final prediction
_, predictions = foward_propagation(x_and_dataset, w, b)
print("Predictions:")
print(np.round(predictions[-1]))
