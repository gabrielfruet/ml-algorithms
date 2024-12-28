# ML Algorithm Implementations using JAX

<img src="https://upload.wikimedia.org/wikipedia/commons/8/86/Google_JAX_logo.svg" width="256">

This repository contains implementations of various machine learning
algorithms and optimization techniques. Each file focuses on a specific
method, demonstrating its core functionality.

## Files and Descriptions

- [**arma.py**](src/arma.py): Autoregressive Moving Average (ARMA) model
  implementation.
- [**bfgs.py**](src/bfgs.py): BFGS optimization algorithm.
- [**jax_mlp2.py**](src/jax_mlp2.py): Multi-layer Perceptron using JAX
  with OOP.
- [**jax_mlp.py**](src/jax_mlp.py): Multi-layer Perceptron using JAX.
- [**least_squares.py**](src/least_squares.py): Linear regression using
  least squares.
- [**np_mlp.py**](src/np_mlp.py): Multi-layer Perceptron using NumPy.
- [**newton_raphson.py**](src/newton_raphson.py): Newton-Raphson method
  for optimization.

## Usage

Each file can be run independently to explore the respective algorithm.
Clone the repository and navigate to the `src` directory to start
experimenting.

``` bash
git clone <repository-url>
cd <repository-name>/src
python arma.py
```

## Requirements

Install dependencies using `pyproject.toml`:

``` bash
pip install -r requirements.txt
```
