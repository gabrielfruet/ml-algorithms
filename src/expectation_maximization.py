from typing import Any
from correlation_matrix import correlate
import jax.numpy as jnp
import pandas as pd
from jax.scipy.optimize import minimize

def multivariate_gaussian_pdf(x, mu, sigma):
    d = x.shape[-1]  # Dimensão do vetor x
    sigma_inv = jnp.linalg.inv(sigma)  # Inversa da matriz de covariância
    sigma_det = jnp.linalg.det(sigma)  # Determinante da matriz de covariância
    
    norm_const = 1 / ((2 * jnp.pi) ** (d / 2) * jnp.sqrt(sigma_det))
    exponent = -0.5 * jnp.sum((x - mu) @ sigma_inv * (x - mu), axis=-1)
    
    return norm_const * jnp.exp(exponent)

def em(X, k: int, maxiter=100, seed=42):
    n = len(X)
    classes = []

    for i in range(k):
        add_1 = 1 if n % k > i else 0
        start = (n // k) * i + add_1
        end = (n // k) * (i + 1) + add_1
        classes.append(X[start:end])

    means = jnp.zeros((k,2))
    covariances = jnp.zeros((k,2,2))

    probs = jnp.zeros((k,n))

    for iter in range(maxiter):
        # E-step
        for i, sample in enumerate(classes):
            mu = sample.mean(axis=0)
            sigma = (sample - mu).T @ (sample - mu) / len(sample)
            # sigma = correlate(sample - mu)
            means = means.at[i].set(mu)
            covariances = covariances.at[i].set(sigma)

        # M-step
        for i in range(k):
            p = multivariate_gaussian_pdf(X, means[i], covariances[i])
            probs = probs.at[i].set(p)

        choices = probs.argmax(axis=0)
        classes = []

        for i in range(k):
            idx = choices == i
            classes.append(X[idx])

        yield classes, means, covariances


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    from matplotlib.patches import Ellipse
    import matplotlib.animation as animation
    data = pd.read_csv('~/Downloads/2d-em.csv', header=None)
    A = jnp.array(data[0])
    B = jnp.array(data[1])
    X = jnp.c_[A,B]

    fig, ax = plt.subplots(figsize=(6,6))

    def update(frame) -> Any:
        ax.clear()
        classes, means, covariances = frame
        for i, (sample, mean, cov) in enumerate(zip(classes, means, covariances)):
            eigenvalues, eigenvectors = jnp.linalg.eigh(cov)
            angle = float(jnp.degrees(jnp.arctan2(*eigenvectors[:, 0][::-1])))
            width, height = 2 * jnp.sqrt(eigenvalues)
            sc = ax.scatter(*sample.T, alpha=0.2, label=f'Class {i}')
            color = sc.get_facecolor()[0]
            ellipse_1std = Ellipse(xy=mean, width=width, height=height, angle=angle,
                              edgecolor='black', facecolor=color, alpha=0.5)
            ellipse_2std = Ellipse(xy=mean, width=2*width, height=2*height, angle=angle,
                              edgecolor='black', facecolor=color, alpha=0.2)
            ax.add_patch(ellipse_1std)
            ax.add_patch(ellipse_2std)
            ax.scatter(*mean, color=color, label=f'Mean {i}')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.legend()
        ax.grid()

    frames = list(em(X, 2, maxiter=5))
    ani = animation.FuncAnimation(fig, update, frames=frames, repeat=False)

    ani.save("em_animation.gif", writer="pillow", fps=1)

    print("GIF salvo como 'em_animation.gif'")
