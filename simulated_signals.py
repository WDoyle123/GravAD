import os
import csv
import numpy as np
from gravad import params, gen_waveform, SEED, preprocess, frequency_series
from pycbc.catalog import Merger
import jax
from jax import random
import jax.numpy as jnp

SAMPLING_RATE = 4096

def write_waveform(mass1, mass2, waveform):
    file_path = "simulated_signals/"
    file_name = f"{mass1}_{mass2}_noisy_waveform_time_domain_{SEED}.csv"

    os.makedirs(file_path, exist_ok=True)
    with open(file_path + file_name, 'w', newline='') as file:
        writer = csv.writer(file)
        for value in waveform:
            writer.writerow([str(value)])

def sim_signal(mass1, mass2):
    key = jax.random.PRNGKey(SEED)
    _, delta_f, psd_jax, _, _ = preprocess("GW150914", "H1", 0)
    freqs = frequency_series(delta_f)
    waveform_template_freq_domain = gen_waveform(mass1, mass2, freqs, params)
    noisy_waveform_time_domain = np.real(waveform_template_freq_domain)
    """  
    # Add noise
    sqrt_psd = np.sqrt(psd_jax)
    real_part = random.normal(key, shape=(len(psd_jax),))
    noise_freq_domain = sqrt_psd * real_part

    # Add the noise to the signal in the frequency domain
    noisy_waveform_freq_domain = waveform_template_freq_domain + noise_freq_domain
    noisy_waveform_freq_domain = np.array(noisy_waveform_freq_domain)

    # Transform the noisy signal to the time domain
    noisy_waveform_time_domain = np.fft.ifft(noisy_waveform_freq_domain)

    # Take the real part of the output, assuming the imaginary part is negligible
    noisy_waveform_time_domain = np.real(noisy_waveform_time_domain)
    """
    
    write_waveform(mass1, mass2, noisy_waveform_time_domain)

def main():
    mass1s = mass2s = [i for i in range(10, 121, 10)]

    for mass1 in mass1s:
        for mass2 in mass2s:
            sim_signal(mass1, mass2)

if __name__ == "__main__":
    main()

