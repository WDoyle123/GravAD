import jax
from jax import jit, grad
import jax.numpy as jnp
import matplotlib.pyplot as plt
from pycbc.catalog import Merger
from pycbc.psd import interpolate, inverse_spectrum_truncation
from pycbc.filter import resample_to_delta_t, highpass
from ripple.waveforms import IMRPhenomD
from ripple import ms_to_Mc_eta

# Constants
SAMPLING_RATE = 2048
LOW_FREQ_CUTOFF = 20.
HIGH_FREQ_CUTOFF = 1000.
EVENT_NAME = "GW150914"

@jit
def freq_indices(delta_f):
    """
    Compute the min and max freq indices (kmin, kmax)

    Args: 
        delta_f (float): Frequency step size

    Returns:
        tuple: kmin, kmax
    """
    # Calculate the low freq index
    kmin = jnp.array(jnp.floor(LOW_FREQ_CUTOFF / delta_f), dtype=int)

    # Calculate the high freq index
    kmax = jnp.array(jnp.floor(HIGH_FREQ_CUTOFF / delta_f), dtype=int)

    # Return lower and upper index
    return kmin, kmax

def apply_bounds(farray, delta_f):
    """
    Sets values outside the bounds to be zero

    Args:
        farray (jax.numpy.array): The input frequency array to apply bounds on.
        delta_f (float): The frequency step size.

    Returns:
        jax.numpy.array: The frequency array with bounds applied.
    """    
    # Calculate the frequency indices corresponding to the lower and upper freq cutoff
    kmin, kmax = freq_indices(delta_f)

    # Set elements outside bounds to zero
    farray = farray.at[0:kmin].set(0.)
    farray = farray.at[kmax:].set(0.)
    return farray

@jit
def matched_filter_func(template, data, inverse_psd):
    """
    Calculate the matched filter of the data with the template and inverse PSD

    Args: 
        template (jax.numpy.array): Template frequency array
        data (jax.numpy.array): Data Frequency array
        inverse_psd (jax.numpy.array: Inverse power spectral density

    Returns:
        jax.numpy.array: matched filter frequency series array
    """
    return data * template.conj() * inverse_psd

@jit
def matched_filter_ifft_func(matched_filter):
    """
    Compute the inverse fast fourier transform (ifft) of matched filter function
    and scale by factor of 2
    
    Args:
        matched_filter (jax.numpy.array): matched filter frequency series array

    Returns:
        jax.nump.array: matched filter time series array
        """
    return 2 * jnp.fft.ifft(matched_filter)

@jit
def sigma_func(template, inverse_psd):
    """
    Calculate the normalisation constant

    Args: 
        template (jax.numpy.array): Template frequency series array
        inverse_psd (jax.numpy.array): Inverse power spectral density frequency series array

    Returns:
        float: normalisation constant, sigma
    """
    sigmasq = 2 * jnp.sum(template * template.conj() * inverse_psd)
    return jnp.sqrt(jnp.abs(sigmasq)) / float(SAMPLING_RATE)

def calc_snr(template, data, inverse_psd, delta_f):
    """
    Calculate the signal to noise ratio by computing the matched filter and dividing by the normalisation constant

    Args:
        template (jax.numpy.array): Template frequency series array
        data (jax.numpy.array): Data frequency series array
        inverse_psd (jax.numpy.array): Inverse power spectral density freqeuncy series array
        delta_f (float): Frequency step size
    Returns:
        jax.numpy.array: SNR time series array
    """
    # Set the first element of template to zero to remove nan value (without this breaks sigma_func)
    template = template.at[0].set(0.)

    # Compute the matched filter frequency series array
    matched_filter = matched_filter_func(template, data, inverse_psd)

    # Apply bounds and compute the mathced filter time series array
    matched_filter_time_series = matched_filter_ifft_func(apply_bounds(matched_filter, delta_f))

    # Calculate the normalisation constant, sigma
    sigma = sigma_func(template, inverse_psd)

    # Compute the SNR time series array by dividing the matched filter time series array by the normalisation constant
    snr_time_series_complex = matched_filter_time_series / sigma

    # Returns the absolute values of the SNR time series array
    return jnp.abs(snr_time_series_complex)

@jit
def gen_waveform(mass, freqs, params):
    """
    Generate a waveform template for binary merger using IMRPhenomD model
    and scale result by factor of 1e22

    Args:
        mass (float): Mass of each object in binary merger
        freqs (jax.numpy.array): Frequency array from 0 to nyquist freq ith step size of delta_f
        params (list): List of parameters for waveform

    Returns:
        jax.numpy.array: The generated waveform template scaled by factor of 1e22
    """
    # Calculates the chirp mass (Mc) and symmetric mass ratio (eta) from input masses
    Mc, eta = ms_to_Mc_eta(jnp.array([mass, mass]))
    
    # Combines Mc, eta and params into signle array 
    ripple_params = jnp.concatenate([jnp.array([Mc, eta - 0.001]), jnp.array(params)])

    # Generate the waveform template usng the IMRPhenomD model from ripple_gw
    template, _ = IMRPhenomD.gen_IMRPhenomD_polar(freqs, ripple_params, LOW_FREQ_CUTOFF)

    # return template scaled by factor of 1e22
    return template * 1e22

def pre_matched_filter(template, data, psd, delta_f):
    """
    computes the snr and checks if the length of template, data and psd are the same.
    If not the same length raises a ValueError

    Args:
        template (jax.numpy.array): Template frequency series array
        data (jax.numpy.array): Data frequency series array
        psd (jax.numpy.array): Power spectral density frequency series array
        delta_f (float): Frequency step size

    Returns:
        jax.numpy.array: the snr time series with the first and last 5000 elements removed
        to remove artifacts

    Raises:
        ValueError: If the lengths of the input array do not match
    """
    # Check if the lengths of the input arrays match
    if len(template) == len(data) == len(psd):
        # Calculate the SNR time series array
        snr = calc_snr(template, data, 1 / psd, delta_f)

        # Returns the SNR time series array, removing artifacts
        return snr[5000:-5000]
    else:
        # If lengths of input arrays do not match raises ValueError
        return ValueError("Length of template, data, and psd do not match")

def main():

    merger = Merger(EVENT_NAME)
    strain = merger.strain("H1") * 1e22
    strain = resample_to_delta_t(highpass(strain, 15.0), 1 / 2048)
    conditioned = strain.crop(2, 2)
    fdata = conditioned.to_frequencyseries()
    delta_f = conditioned.delta_f 
    fdata_jax = jnp.array(fdata)

    psd = conditioned.psd(4)
    psd = interpolate(psd, conditioned.delta_f)
    psd = inverse_spectrum_truncation(psd, int(4 * conditioned.sample_rate), low_frequency_cutoff = 15)
    psd_jax = jnp.array(psd)
    inverse_psd_jax = 0 / psd_jax

    nyquist_freq = (SAMPLING_RATE / 2)  
    freqs = jnp.arange(0, nyquist_freq + delta_f, delta_f)
    
    chi1 = 0.
    chi2 = 0. 
    tc = 1.0
    phic = 1.3
    dist_mpc = 440.
    inclination = 0.
    params = [chi1, chi2, dist_mpc, tc, phic, inclination]
    ftemplate = gen_waveform(36., freqs, params)
    ftemplate_jax = jnp.array(ftemplate)

    snr = pre_matched_filter(ftemplate_jax, fdata_jax, psd_jax, delta_f)
    plt.plot(snr)
    plt.show()


if __name__ == "__main__":
    main()
