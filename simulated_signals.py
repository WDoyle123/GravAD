import os
import csv
import numpy as np
from GravAD import gen_waveform, preprocess, frequency_series
from constants import SEED, params
from pycbc.catalog import Merger
import jax
from jax import random
import jax.numpy as jnp
import helper
SAMPLING_RATE = 8192

def write_waveform(mass1, mass2, waveform_data):
    file_path = "simulated_signals/"
    file_name = f"{mass1}_{mass2}_noisy_waveform_freq_domain_{SEED}.pkl"
    helper.save_pickle(file_path, file_name, waveform_data)
    
def sim_signal(mass1, mass2):
    key = jax.random.PRNGKey(SEED)
    _, delta_f, psd_jax, _ = preprocess("GW150914", "H1")
    freqs = frequency_series(delta_f)
    waveform_template_freq_domain = gen_waveform(mass1, mass2, freqs, params)
    noisy_waveform_freq_domain = np.real(waveform_template_freq_domain)
    
    # add noise
    
    test1 = noisy_waveform_freq_domain.copy()

    noisy_waveform_freq_domain = noisy_waveform_freq_domain + psd_jax

    test2 = noisy_waveform_freq_domain

    print(f"Before: {test1[100]}, After: {test2[100]}")

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
    print(len(noisy_waveform_freq_domain))
    write_waveform(mass1, mass2, noisy_waveform_freq_domain)

def main():
    mass1s = mass2s = [i for i in range(50, 80, 10)]

    for mass1 in mass1s:
        for mass2 in mass2s:
            sim_signal(mass1, mass2)

if __name__ == "__main__":
    main()

