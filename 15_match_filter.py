import jax
from jax import jit, grad, lax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from pycbc.catalog import Merger
from pycbc.psd import interpolate, inverse_spectrum_truncation
from pycbc.filter import resample_to_delta_t, highpass
from ripple.waveforms import IMRPhenomD
from ripple import ms_to_Mc_eta
import math

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
    kmax = jnp.array(jnp.floor(HIGH_FREQ_CUTOFF / delta_f) + 1, dtype=int)

    # Return lower and upper index
    return kmin, kmax

@jit
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

    # Create a boolean mask for the allowed frequency range
    mask = (jnp.arange(farray.shape[0]) >= kmin) & (jnp.arange(farray.shape[0]) < kmax)

    # Apply the mask to the frequency array
    farray_masked = farray * mask
        
    farray_masked_no_nan = jnp.where(jnp.isnan(farray_masked), 0, farray_masked)

    return farray_masked_no_nan

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
    return jnp.sqrt(jnp.abs(sigmasq) + 1e-9) / float(SAMPLING_RATE)

@jit
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
    # +1 fudge factor removes nan from dsnr. (higher factor reduces snr value and increases dsnr values)
    freqs = freqs + 1

    # Calculates the chirp mass (Mc) and symmetric mass ratio (eta) from input masses
    Mc, eta = ms_to_Mc_eta(jnp.array([mass, mass]))
    
    # Combines Mc, eta and params into signle array 
    ripple_params = jnp.concatenate([jnp.array([Mc, eta - 0.001]), jnp.array(params)])

    # Generate the waveform template usng the IMRPhenomD model from ripple_gw
    template, _ = IMRPhenomD.gen_IMRPhenomD_polar(freqs, ripple_params, LOW_FREQ_CUTOFF)

    # Replace NaN values with zeros in the template
    template_no_nan = jnp.where(jnp.isnan(template), 0, template)

    # return template_no_nan scaled by factor of 1e22
    return template_no_nan * 1e22

@jit
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
        psd = psd + 1e-9
        snr = calc_snr(template, data, 1 / psd, delta_f)

        # Returns the SNR time series array, removing artifacts
        return snr[5000:-5000] * 10
    else:
        # If lengths of input arrays do not match raises ValueError
        return ValueError("Length of template, data, and psd do not match")

def make_function(data, psd, freqs, params, delta_f):
    """
    Creates a function to compute the max snr for a given mass

    Args:
    data (jax.numpy.array): Data frequency series array
    psd (jax.numpy.array): Power spectral density frequecy series array
    freqs (jax.numpy.array): Frequency array from 0 to nyquist frequency with step size of delta_f
    params (list): List of parameters for waveform
    delta_f (float): Frequency step size

    Returns:
    function: The function takes an input mass and returns the max snr
    """
    def snr(mass):
        ftemplate_jax = jnp.array(gen_waveform(mass, freqs, params))
        snr = pre_matched_filter(ftemplate_jax, data, psd, delta_f)
        return snr.max()
    return jit(snr)

def psd_func(conditioned):
    """
    Calculate the power spectral density of the conditioned strain data

    Args:
    conditioned (pycbc.types.timeseries.TimeSeries): conditioned strain data

    Returns:
    jax.numpy.array: Power spectral density frequency series array 
    """
    psd = conditioned.psd(4)
    psd = interpolate(psd, conditioned.delta_f)
    psd = inverse_spectrum_truncation(psd, int(4 * conditioned.sample_rate), low_frequency_cutoff = 15)
    return jnp.array(psd)

def main():

    # Load the data for event
    merger = Merger(EVENT_NAME)
    strain = merger.strain("H1") * 1e22

    # High-pass filter the data and resample
    strain = resample_to_delta_t(highpass(strain, 15.0), 1 / 2048)

    # Crop the data to remove filter artifacts
    conditioned = strain.crop(2, 2)

    # Convert the condtioned time series data to frequency domain
    fdata = conditioned.to_frequencyseries()
    delta_f = conditioned.delta_f 
    fdata_jax = jnp.array(fdata)

    # Calculate the PSD of the conditioned data
    psd_jax = psd_func(conditioned)

    # Calculate the inverse PSD
    inverse_psd_jax = 1 / psd_jax

    # Create a frequency array from 0 to Nyquist frequency with a step size of delta_f
    nyquist_freq = (SAMPLING_RATE / 2)  
    freqs = jnp.arange(0, nyquist_freq + delta_f, delta_f)

    # Apply frequency bounds to the frequency array
    freqs_cropped = apply_bounds(freqs, delta_f)

    # Define waveform parameters
    chi1 = 0.
    chi2 = 0. 
    tc = 1.0
    phic = 1.3
    dist_mpc = 440.
    inclination = 0.
    params = [chi1, chi2, dist_mpc, tc, phic, inclination]

    # Create the SNR function with the given data and parameters
    snr_func = make_function(fdata_jax, psd_jax, freqs, params, delta_f)

    # Compute the gradient of the SNR function
    dsnr = jit(grad(snr_func))
   
    # Iterate through a range of masses and compute the SNR and its gradient for each mass
    for i in range(60):
        mass = 20. + (i)
        snr_val = snr_func(mass)
        grad_val = dsnr(mass)
        snr_str = "{:.12f}".format(snr_val) if not math.isnan(snr_val) else " " * 12 + "NaN"
        grad_str = "{:.12f}".format(grad_val) if not math.isnan(grad_val) else " " * 12 + "NaN"
        print("mass: {:>6.1f}, snr: {:>12}, grad: {:>6}".format(mass, snr_str, grad_str))
    
    # SNR function used for plotting
    def my_snr(mass):
        template = gen_waveform(mass, freqs, params)
        snr = pre_matched_filter(template, fdata_jax, psd_jax, delta_f)
        return snr.max()

    # Calculate the SNRs for a range of masses and plot the result
    snrs = [float(my_snr(m)) for m in jnp.linspace(20, 80, 500)]
    plt.scatter(jnp.linspace(20, 80, 500), snrs)
    plt.show()
    
      
if __name__ == "__main__":
    main()
