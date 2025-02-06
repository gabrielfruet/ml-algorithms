import jax
import tqdm
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

def wiener_fit(x, d, filter_size=10):
    x,d = wiener_filter_inputs_sampling(x, d, filter_size)

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
    x = x
    w_opt = w_opt[::-1]  # Reverse the filter for correct convolution behavior

    # result = jax.lax.conv_general_dilated(
    #     x, w_opt,
    #     window_strides=(1,),  # Stride of 1
    #     padding="SAME",  # Equivalent to 'same' filtering behavior
    # )
    result = jnp.convolve(x, w_opt, mode='same')

    return result

def wiener_filter_inputs_sampling(x, d, filter_size):
    n = len(x)

    samples_indexes = wiener_sample_indexes(n, filter_size)

    x_new = x[samples_indexes]
    d_new = d[filter_size:]  # Only next-step predictions

    return x_new, d_new

def main1():
    key = jax.random.PRNGKey(0)
    n_samples = 1000
    filter_size = 100
    t = jnp.linspace(0, 10, n_samples)
    d = jnp.sin(t) * 30
    key, subkey = jax.random.split(key)
    v1 = jax.random.normal(subkey, (n_samples,)) * 0.5
    key, subkey = jax.random.split(key)
    v2 = jax.random.normal(subkey, (n_samples,)) * 0.5
    x = d + v1
    w_opt = wiener_fit(v2, x)

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].plot(t, d, label='Desired Signal')
    ax[0].set_title('Desired Signal')
    ax[1].plot(t, x, label='Noisy Signal')
    ax[1].set_title('Noisy Signal')

    v2_hat = wiener_apply(x, w_opt)
    ax[2].plot(t[:len(v2_hat)], x - v2_hat, label='Filtered Signal')
    ax[2].set_title('Filtered Signal')

    for a in ax:
        a.legend()
    plt.show()

FILTER_SIZE = 32
BATCH_SIZE=2 ** 14

def main2(filter_size=FILTER_SIZE, batch_size=BATCH_SIZE):
    audio, sr = sf.read('./bachfugue.wav')
    audio_hat = []
    noise = jax.random.normal(jax.random.PRNGKey(0), (len(audio), )) * 0.1
    number_of_batches = len(audio) // batch_size
    print('Number of batches: ',number_of_batches)
    def slicer(x, batch_size=batch_size):
        for i in range(0, len(x), batch_size):
            yield jax.lax.dynamic_slice(x, (i*batch_size,), (batch_size,))

    device = jax.devices()[0]

    for i,(desired_signal,noisy_signal) in enumerate(zip(audio.T, audio.T + noise)):
        print(f'Processing {i}th channel')
        with tqdm.tqdm(total=number_of_batches) as pbar:
            w_opt = jnp.zeros(filter_size)

            for x,d in zip(slicer(noisy_signal), slicer(desired_signal)):
                w_opt_j = wiener_fit(x, d, filter_size)
                w_opt += w_opt_j * len(x)/len(desired_signal)
                pbar.update(1)
                cuda_mem = device.memory_stats()
                assert cuda_mem is not None
                pbar.set_postfix(mem=f'{cuda_mem["bytes_in_use"] / 1024 ** 2:.2f} MB')
                pbar.refresh()

            d_hat = wiener_apply(noisy_signal, w_opt)
            pbar.set_postfix(error=jnp.sqrt(jnp.mean((d_hat - desired_signal)**2)))
            audio_hat.append(d_hat)

    audio_hat = jnp.stack(audio_hat).T
    sf.write('./bachfugue_filtered.wav', audio_hat, sr)
    sf.write('./bachfugue_noisy.wav', (audio.T + noise).T, sr)


def main3(filter_size=FILTER_SIZE, batch_size=BATCH_SIZE, noise_gain: float = 0.1):
    audio, sr = sf.read('./bachfugue.wav')
    audio_hat = []
    key = jax.random.PRNGKey(0)
    def generate_noise(shape, key):
        key, subkey = jax.random.split(key)
        return jax.random.normal(subkey, shape) * noise_gain, key

    noise, key = generate_noise((len(audio),2), key)
    number_of_batches = len(audio) // batch_size

    print('Number of batches: ',number_of_batches)
    def slicer(x, batch_size=batch_size):
        for i in range(0, len(x), batch_size):
            yield jax.lax.dynamic_slice(x, (i*batch_size,), (batch_size,))

    device = jax.devices()[0]

    for i,(desired_signal, noise_sample) in enumerate(zip(audio.T, noise.T)):
        print(f'Processing {i}th channel')
        with tqdm.tqdm(total=number_of_batches) as pbar:
            w_opt = jnp.zeros(filter_size)

            for d,v1 in zip(slicer(desired_signal), slicer(noise_sample)):
                v2, key = generate_noise(len(d), key)
                x = d + v1
                w_opt_j = wiener_fit(v2, x, filter_size)
                w_opt += w_opt_j * len(x)/len(desired_signal)

                # pbar logic
                pbar.update(1)
                cuda_mem = device.memory_stats()
                assert cuda_mem is not None
                pbar.set_postfix(mem=f'{cuda_mem["bytes_in_use"] / 1024 ** 2:.2f} MB')
                pbar.refresh()

            gen_noise, key = generate_noise((len(desired_signal), ), key)

            d_hat = wiener_apply(gen_noise, w_opt)
            pbar.set_postfix(error=jnp.sqrt(jnp.mean((noise_sample - d_hat)**2)))
            audio_hat.append(desired_signal + noise_sample - d_hat)
            # audio_hat.append(d_hat)

    audio_hat = jnp.stack(audio_hat).T
    sf.write('./bachfugue_filtered2.wav', audio_hat, sr)
    sf.write('./bachfugue_noisy2.wav', audio + noise, sr)

if __name__ == '__main__':
    main2()
