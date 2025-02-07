import jax
import tqdm
import jax.numpy as jnp
from matplotlib import pyplot as plt
import soundfile as sf
from src.correlation_matrix import correlate

@jax.jit
def corr(x, y):
    return x.T @ y / len(x)

def wiener_fit(x, d, filter_size=10, method='custom'):
    xn,d = wiener_filter_inputs_sampling(x, d, filter_size)

    # Explicação p_xd Isso é a correlação entre cada x de entrada do filtro com
    # o valor d esperado. Ou seja, como cada x influencia em d. Essa correlação
    # cruzada vai ser um vetor de tamanho n, sendo n o tamanho do filtro. Se x
    # tem 990 amostras de tamanho 10, e d tem 990 amostras(o resultado da
    # aplicação do filtro nas 10 amostras de x) então a correlação terá tamanho
    # 10, que vai ser a correlação media da iésima amostra com a saída d.
    p_xd = corr(xn, d)
    if method == 'custom':
        Rx = correlate(x,x, p=filter_size)
    else:
        Rx = corr(xn, xn)
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

    result = jnp.convolve(x, w_opt, mode='same')

    return result

def wiener_filter_inputs_sampling(x, d, filter_size):
    n = len(x)

    samples_indexes = wiener_sample_indexes(n, filter_size)

    x_new = x[samples_indexes]
    d_new = d[filter_size:]  # Only next-step predictions

    return x_new, d_new

def main1(sin_amp=5, filter_size=100):
    key = jax.random.PRNGKey(0)
    n_samples = 1000
    t = jnp.linspace(0, 10, n_samples)
    d = jnp.sin(t) * sin_amp
    key, subkey = jax.random.split(key)
    v1 = jax.random.normal(subkey, (n_samples,)) * 0.5
    key, subkey = jax.random.split(key)
    v2 = jax.random.normal(subkey, (n_samples,)) * 0.5
    x = d + v1
    w_opt = wiener_fit(v2, x, filter_size)

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].plot(t, d, label='Desired Signal')
    ax[0].set_title('Desired Signal')
    ax[1].plot(t, x, label='Noisy Signal')
    ax[1].set_title('Noisy Signal')
    breakpoint()

    v2_hat = wiener_apply(x, w_opt)
    ax[2].plot(t[:len(v2_hat)], x - v2_hat, label='Filtered Signal')
    ax[2].set_title('Filtered Signal')

    for a in ax:
        a.legend()
    plt.show()

FILTER_SIZE = 32
BATCH_SIZE=2 ** 14

def sin_audio(sr=4800, duration_seconds=5, freq=440):
    t = jnp.linspace(0, duration_seconds, sr * duration_seconds)
    audio_sing_channel = jnp.sin(2 * jnp.pi * freq * t)
    return jnp.stack([audio_sing_channel, audio_sing_channel]).T, sr


def batchfuge_audio():
    audio, sr = sf.read('./bachfugue.wav')
    return audio, sr

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
                # w_opt += w_opt_j * len(x)/len(desired_signal)
                w_opt += w_opt_j / number_of_batches
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


def main3(
        filter_size=FILTER_SIZE,
        batch_size=BATCH_SIZE,
        noise_gain: float = 0.1,
        audio_sampler=batchfuge_audio,
        method='custom'):
    audio, sr = audio_sampler()
    audio_hat = []

    if batch_size == -1:
        batch_size = len(audio)
    key = jax.random.PRNGKey(0)
    def generate_noise(shape, key,freq=440):
        key, subkey = jax.random.split(key)
        # t = jnp.linspace(0, shape[0]/sr, shape[0]*shape[1])
        return jax.random.normal(subkey, shape) * noise_gain, key
        # return jnp.sin(2 * jnp.pi * freq * t).reshape(shape) * noise_gain, key

    v1_noise, key = generate_noise((len(audio),2), key)
    v2_noise, key = generate_noise((len(audio),2), key)
    number_of_batches = len(audio) // batch_size

    print('Number of batches: ',number_of_batches)
    def slicer(x, batch_size=batch_size):
        length = x.shape[0]
        for i in range(0, length, batch_size):
            size = min(batch_size, length - i)
            yield jax.lax.dynamic_slice(x, (i,), (size,))

    device = jax.devices()[0]

    for i,(desired_signal, v1_noise_sample, v2_noise_sample) in enumerate(zip(audio.T, v1_noise.T, v2_noise.T)):
        print(f'Processing {i}th channel')

        with tqdm.tqdm(total=number_of_batches) as pbar:
            w_opt = []
            noisy_signal = desired_signal + v1_noise_sample


            for x,v2 in zip(slicer(noisy_signal), slicer(v2_noise_sample)):
                w_opt_j = wiener_fit(v2, x, filter_size, method=method)
                w_opt.append(w_opt_j)


                # pbar logic
                pbar.update(1)
                cuda_mem = device.memory_stats()
                if cuda_mem is not None:
                    pbar.set_postfix(mem=f'{cuda_mem["bytes_in_use"] / 1024 ** 2:.2f} MB')
                pbar.refresh()

            w_opt = jnp.array(w_opt).mean(axis=0)


            v1_hat = wiener_apply(noisy_signal, w_opt)
            # fig, ax = plt.subplots(1, 3, figsize=(15, 5))
            # ax[0].plot((desired_signal)[0:1000], label='original signal')
            # ax[1].plot((desired_signal + v1_noise_sample - v1_hat)[0:1000], label='estimated signal')
            # ax[2].plot((desired_signal + v1_noise_sample)[0:1000], label='noisy signal')
            # fig.legend()
            # fig.show()
            pbar.set_postfix(error=jnp.sqrt(jnp.mean((v1_noise_sample - v1_hat)**2)))
            audio_hat.append(noisy_signal - v1_hat)

    def normalize(x):
        return (x - x.min()) / (x.max() - x.min())*2 - 1

    audio_hat = normalize(jnp.stack(audio_hat).T)
    noisy_audio = normalize(audio + v1_noise)
    sf.write('./bachfugue_filtered2.wav', audio_hat, sr)
    sf.write('./bachfugue_noisy2.wav', noisy_audio, sr)


if __name__ == '__main__':
    main2()
