import jax
import jax.numpy as jnp
from matplotlib import pyplot as plt
from scipy.io import wavfile
import numpy as np
import soundfile as sf
import sounddevice as sd

@jax.jit
def cov(x, y):
    μ_x = jnp.mean(x, axis=0, keepdims=True)
    μ_y = jnp.mean(y, axis=0, keepdims=True)
    x_bar = x - μ_x
    y_bar = y - μ_y
    return x_bar.T @ y_bar / len(x)

def wiener_fit(x, d):
    # Explicação p_xd Isso é a correlação entre cada x de entrada do filtro com
    # o valor d esperado. Ou seja, como cada x influencia em d. Essa correlação
    # cruzada vai ser um vetor de tamanho n, sendo n o tamanho do filtro. Se x
    # tem 990 amostras de tamanho 10, e d tem 990 amostras(o resultado da
    # aplicação do filtro nas 10 amostras de x) então a correlação terá tamanho
    # 10, que vai ser a correlação media da iésima amostra com a saída d.
    p_xd = cov(x, d)
    Rx = cov(x, x)
    w_opt = jnp.linalg.solve(Rx, p_xd)
    return jnp.squeeze(w_opt)

def wiener_sample_indexes(sample_size, filter_size):
    filter_padding = jnp.arange(filter_size)
    sample_window = jnp.arange(sample_size - filter_size)
    # samples_indexes é uma matriz de tamanho (n_samples, filter_size) onde
    # cada linha é um conjunto de índices que representam uma janela de
    # amostras de tamanho filter_size
    samples_indexes = sample_window[:, None] + filter_padding
    return samples_indexes

def wiener_apply(x, w_opt):
    x = x[None, None, :]  # Add batch and channel dimensions for convolution
    w_opt = w_opt[::-1]  # Reverse the filter for correct convolution behavior
    w_opt = w_opt[None, None, :]  # Add batch and channel dimensions

    result = jax.lax.conv_general_dilated(
        x, w_opt,
        window_strides=(1,),  # Stride of 1
        padding="VALID",  # Equivalent to 'same' filtering behavior
    )

    return result[0, 0, :]  # Remove extra dimensions

def sample_wiener_filter_inputs(x, d, filter_size):
    n = len(x)

    samples_indexes = wiener_sample_indexes(n, filter_size)

    x_new = x[samples_indexes]
    d_new = d[filter_size:]  # Only next-step predictions

    return x_new, d_new

def main1():
    prng_key = jax.random.PRNGKey(0)
    n_samples = 1000
    filter_size = 512
    t = jnp.linspace(0, 10, n_samples)
    d = jnp.sin(t) * 10
    noise = jax.random.normal(prng_key, (n_samples,)) * 0.5
    x = d + noise
    x_train, d_train = sample_wiener_filter_inputs(x, d, filter_size=filter_size)
    w_opt = wiener_fit(x_train, d_train)

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].plot(t, d, label='Desired Signal')
    ax[0].set_title('Desired Signal')
    ax[1].plot(t, x, label='Noisy Signal')
    ax[1].set_title('Noisy Signal')

    d_hat = wiener_apply(x, w_opt)
    ax[2].plot(t[:len(d_hat)], d_hat, label='Filtered Signal')
    ax[2].set_title('Filtered Signal')

    for a in ax:
        a.legend()
    plt.show()

def main2():
    audio, sr = sf.read('./bachfugue.wav')
    audio_hat = []
    filter_size = 32
    noise = jax.random.normal(jax.random.PRNGKey(0), (len(audio), )) * 0.1
    for i,(d,x) in enumerate(zip(audio.T, audio.T + noise)):
        print(f'{i}th channel')
        x_train, d_train = sample_wiener_filter_inputs(x, d, filter_size=filter_size)
        w_opt = wiener_fit(x_train, d_train)

        d_hat = wiener_apply(x, w_opt)
        audio_hat.append(d_hat)

    audio_hat = jnp.stack(audio_hat).T
    sf.write('./bachfugue_filtered.wav', audio_hat, sr)
    sf.write('./bachfugue_noisy.wav', (audio.T + noise).T, sr)


if __name__ == '__main__':
    main2()
