from ripple.waveforms import IMRPhenomD
from ripple import ms_to_Mc_eta
import matplotlib.pyplot as plt
from pycbc.catalog import Merger
from pycbc.psd import interpolate, inverse_spectrum_truncation
from pycbc.types import TimeSeries, FrequencySeries
from pycbc.filter import resample_to_delta_t, highpass
import jax
from jax import jit, lax, grad
import jax.numpy as jnp
import math

@jit
def get_kmin_kmax(f_l, f_u, delta_f):
    """
    Calculate the minimum and maximum frequency bin indices given the lower and upper
    frequency bounds, frequency resolution, and total number of frequency bins.

    Args:
    f_l: a float representing the lower frequency bound
    f_u: a float representing the upper frequency bound
    delta_f: a float representing the frequency resolution

    Returns:
    two integers representing the minimum and maximum frequency bin indices.
    """
    kmin = jnp.array(jnp.floor(f_l / delta_f), dtype=int)
    kmax = jnp.array(jnp.floor(f_u / delta_f), dtype=int)
    return kmin, kmax


def apply_kmin_kmax(template, f_l, f_u, total_bins, kmin, kmax):
    """
    Zero out frequency components in a frequency domain template or signal outside of the
    specified frequency bounds.

    Args:
    template: a Jax array representing the frequency signal
    f_l: a float representing the lower frequency bound
    f_u: a float representing the upper frequency bound
    total_bins: an integer representing the total number of frequency bins
    kmin: an integer representing the minimum frequency bin index
    kmax: an integer representing the maximum frequency bin index

    Returns:
    A Jax array with frequency components outside of the specified bounds set to zero.
    """
    total_bins = int(total_bins)
    tmp = jnp.zeros(total_bins, dtype=template.dtype)
    tmp = tmp.at[0:(int(total_bins) // 2) + 1].set(template)
    tmp = tmp.at[0:kmin].set(0.)
    tmp = tmp.at[kmax:].set(0.)
    return tmp

@jit
def matched_filter(template_fs, data_fs,  psd):
    """
    Applies matched filtering to the input data and template, given the power spectral density and
    the sampling frequency.

    Args:
    data_fs: a Jax array representing the frequency domain data to be filtered
    template_fs: a Jax array representing the frequency domain template for the matched filter
    psd: a Jax array representing the power spectral density
    fs: a Jax array representing the sampling frequency
    delta_f: a float representing the frequency resolution

    Returns:
    An array of real values representing the signal-to-noise ratio (SNR)

    Source: https://www.gw-openscience.org/s/events/GW150914/LOSC_Event_tutorial_GW150914.html
    """
    # Calculate the optimal filter
    optimal = data_fs * template_fs.conj() * psd
    
    # Compute the matched filter in the time domain
    optimal_time = 2 * jnp.fft.ifft(optimal)

    # Compute the normalisation constant
    sigmasq = 2 * jnp.sum(template_fs * template_fs.conj() * psd)
    sigma = jnp.sqrt(jnp.abs(sigmasq)) / 2048. 
    
    # Compute the signal-to-noise ratio
    snr_complex = optimal_time / sigma 
    snr = jnp.abs(snr_complex)
    
    # Return the signal-to-noise ratio array
    return snr

@jit
def waveform_template(mass, fs, params, f_ref, psd, fdata):
    """
    Generates a waveform template using the given parameters, sampling frequency, power spectral density,
    and frequency domain data.
    
    Args:
    mass: a float representing the mass of a black hole
    fs: a Jax array representing the sampling frequency
    params: a list of parameters that are used to generate the waveform template
    f_ref: a float representing the reference frequency
    psd: a Jax array representing the power spectral density
    fdata: a Jax array representing the frequency domain data
    
    Returns:
    A Jax array representing the waveform template
    """
    Mc, eta = ms_to_Mc_eta(jnp.array([mass,mass]))
    ripple_params = jnp.concatenate([jnp.array([Mc, eta]), jnp.array(params)])
    template, _ = IMRPhenomD.gen_IMRPhenomD_polar(fs, ripple_params, f_ref)
    return template * 1e22

def make_function(freqs, f_ref, psd_jax, inverse_psd_jax_cropped, fdata_jax, fdata_jax_cropped, delta_f, params, low_freq, high_freq, total_bins, kmin,kmax):
    def snr(mass):
        template_freqs = waveform_template(mass, freqs, params, f_ref, psd_jax, fdata_jax)
        template_freqs = apply_kmin_kmax(template_freqs, low_freq, high_freq, total_bins, kmin, kmax)
        template_freqs = template_freqs[kmin:kmax]
        snr = matched_filter(template_freqs, fdata_jax_cropped, inverse_psd_jax_cropped)
        
        # remove snr artifact
        tmp = jnp.zeros(len(snr))
        tmp = tmp.at[0:len(snr)].set(snr)
        tmp = tmp.at[0:5000].set(0)
        snr = tmp.at[len(snr)-5000:].set(0)
        snr = snr * 10
        return snr.max()

    return jit(snr)

def main():

    # Get gw wave data
    eventname = "GW150914"
    merger = Merger(eventname)
    strain = 1e22 * merger.strain("H1")
    sampling_rate = int(2048)
    delta_f = 1.0 / sampling_rate
    strain = resample_to_delta_t(highpass(strain, 15.0), delta_f)
    conditioned = strain.crop(2, 2)
    fdata = conditioned.to_frequencyseries()
    fdata_jax = jnp.array(fdata)

    # Get psd
    psd = conditioned.psd(4)
    psd = interpolate(psd, conditioned.delta_f)
    psd = inverse_spectrum_truncation(psd, int(4 * conditioned.sample_rate), low_frequency_cutoff = 15)
    psd_jax = jnp.array(psd)
    inverse_psd_jax = 1 / psd_jax

    # make sure data has the sampe delta_f
    npts_data = len(conditioned)
    nyquist_freq_data = 1.0 / (delta_f * 2)
    freqs = jnp.linspace(0, nyquist_freq_data, int(1 + 28 * nyquist_freq_data))
    delta_f = freqs[1]-freqs[0]
    freq_bins = int(1 / delta_f)
    
    # Get template
    chi1 = 0.
    chi2 = 0.
    tc = 1.0
    phic = 1.3
    dist_mpc = 440.
    inclination = 0.
    params = [chi1, chi2, dist_mpc, tc, phic, inclination]
    mass = 36.
    low_freq = 20.
    high_freq = 1000.
    f_ref = low_freq

    # Matched filter
    total_bins = freq_bins * sampling_rate
    kmin, kmax = get_kmin_kmax(low_freq, high_freq, delta_f)

    fdata_jax_cropped = apply_kmin_kmax(fdata_jax, low_freq, high_freq, total_bins, kmin, kmax)
    psd_jax_cropped = apply_kmin_kmax(psd_jax, low_freq, high_freq, total_bins, kmin, kmax)
    
    fdata_jax_cropped = fdata_jax_cropped[kmin:kmax]
    psd_jax_cropped = psd_jax_cropped[kmin:kmax]
    inverse_psd_jax_cropped = 1 / psd_jax_cropped

    snr_func = make_function(freqs, f_ref, psd_jax, inverse_psd_jax_cropped,fdata_jax, fdata_jax_cropped, delta_f, params, low_freq, high_freq, total_bins,kmin, kmax)
    dsnr = jit(grad(snr_func))

    for i in range(30):
        mass = 36. + (i/10)
        snr_val = snr_func(mass)
        grad_val = dsnr(mass)
        snr_str = "{:.12f}".format(snr_val) if not math.isnan(snr_val) else " " * 12 + "NaN"
        grad_str = "{:.12f}".format(grad_val) if not math.isnan(grad_val) else " " * 12 + "NaN"
        print("mass: {:>6.1f}, snr: {:>12}, grad: {:>6}".format(mass, snr_str, grad_str))
   
    line = "~" * (75)
    print(line)
    debug_template = waveform_template(36.5, freqs, params, f_ref, psd_jax, fdata_jax)
    print(len(debug_template))

    counts = 0
    for i in abs(debug_template):
        if math.isnan(i):
            #print(i)
            counts += 1
    print(counts)
    
    if len(debug_template) == counts:
        print(f"all values in debug template are NaN")

if __name__ =="__main__":
    main()

