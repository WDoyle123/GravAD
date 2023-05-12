import jax
from jax import jit, grad, lax
import jax.numpy as jnp
from jax import random
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib import cm
from matplotlib.cm import ScalarMappable
from pycbc.catalog import Merger
from pycbc.psd import interpolate, inverse_spectrum_truncation
from pycbc.filter import resample_to_delta_t, highpass
from ripple.waveforms import IMRPhenomD
from ripple import ms_to_Mc_eta
import math
import time

# Constants
SAMPLING_RATE = 2048
LOW_FREQ_CUTOFF = 20.
HIGH_FREQ_CUTOFF = 1000.
#EVENT_NAME = "GW150914"
#EVENT_NAME = "GW151012"
#EVENT_NAME = 'GW151226'
#EVENT_NAME = 'GW170814'
EVENT_NAME = 'GW170104' 
MAX_ITERS = 1000
SEED = 0

def gen_init_mass(rng_key):
    """
    Generate an initial mass value uniformly between 20 and 80

    Args:
    rng_key (jax.random.PRNGKey): A JAX random key used to generate random numbers

    Returns:
    tuple: The generated initial mass value and an updated random key
    """
    rng_key, subkey = random.split(rng_key)
    init_mass = random.uniform(subkey, minval = 20., maxval = 80.)
    return init_mass, rng_key

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
    farray (jax.numpy.array): The input frequency array to apply bounds on
    delta_f (float): The frequency step size

    Returns:
    jax.numpy.array: The frequency array with bounds applied
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
    jax.numpy.array: matched filter time series array
    """
    return 2 * jnp.fft.ifft(matched_filter) * float(SAMPLING_RATE)

@jit
def sigma_func(template, inverse_psd, delta_f):
    """
    Calculate the normalisation constant

    Args: 
    template (jax.numpy.array): Template frequency series array
    inverse_psd (jax.numpy.array): Inverse power spectral density frequency series array

    Returns:
    float: normalisation constant, sigma
    """
    sigmasq = jnp.sum(template * template.conj() * inverse_psd) * delta_f
    return jnp.sqrt(jnp.abs(sigmasq) + 1e-9) / 0.6

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
    sigma = sigma_func(template, inverse_psd, delta_f)

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
        return snr[5000:-5000] 
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

def get_optimal_mass(init_mass, freqs, params, data, psd, delta_f):
    """
    Perform a simulated annealing optimization to find the mass value
    that maximizes the signal-to-noise ratio (SNR) for a gravitational wave event

    Args:
    init_mass (float): Initial mass value to start the optimisation
    freqs (jax.numpy.array): Frequency array from 0 to Nyquist frequency with step size of delta_f
    params (list): List of parameters for waveform generation
    data (jax.numpy.array): Data frequency series array
    psd (jax.numpy.array): Power spectral density frequency series array
    delta_f (float): Frequency step size

    Returns:
    dict: A dictionary containing the results of mass, SNR, and gradient for each iteration
    """    
    results = {}
    mass = init_mass
    snr_func = make_function(data, psd, freqs, params, delta_f)
    dsnr = jit(grad(snr_func))
    rng_key = random.PRNGKey(SEED)

    # Set initial temperature and the rate of temperature decrease
    temperature = 10
    annealing_rate = 0.995

    for i in range(MAX_ITERS):
        
        snr = float(snr_func(mass))
        gradient = float(dsnr(mass))
        
        rng_key, subkey = random.split(rng_key)
        learning_rate = random.uniform(subkey, minval=0.5, maxval=1.5)

        # Add random perturbation to the mass update step
        perturbation = random.normal(subkey) * temperature

        results[i] = {"mass": mass, "snr": snr, "gradient": gradient, "iter": i}

        mass = mass + (learning_rate * gradient) + perturbation
        if mass < 11.:
            mass = 11. + abs(perturbation)
        if mass > 120.:
            mass = 120. - abs(perturbation)
        if math.isnan(mass):
            break

        # Update temperature
        temperature *= annealing_rate

    return results

def plot_results(results):
    n = len(results)

    mass_values = [result["mass"] for result in results.values()]
    snr_values = [result["snr"] for result in results.values()]
    iterations = list(results.keys())
    
    colormap = cm.jet
    norm = plt.Normalize(vmin=min(mass_values), vmax=max(mass_values))
    
    plt.figure(figsize=(18, 12))
    plt.scatter(iterations, snr_values, c=mass_values, cmap=colormap, norm=norm)
    
    mappable = ScalarMappable(norm=norm, cmap=colormap)
    cbar = plt.colorbar(mappable, ax=plt.gca())
    cbar.set_label('Mass Index', fontsize=24)
    cbar.ax.tick_params(labelsize=24)  # Change font size of colorbar numbers
    
    plt.tick_params(axis='both', which='major', labelsize=24)
    
    plt.tight_layout(pad=5.0)  # Add padding for title
    plt.grid(True)
    
    plt.xlabel("Iteration", fontsize=24)
    plt.ylabel("SNR", fontsize=24)
    plt.title(f"SNR vs Iteration with Mass as colour {EVENT_NAME}", fontsize=24, pad=20)  # Add padding for title
    
    plt.show()

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

    # Define waveform parameters
    chi1 = 0.
    chi2 = 0.
    tc = 1.0
    phic = 1.3
    dist_mpc = 1000.
    inclination = 0.
    params = [chi1, chi2, dist_mpc, tc, phic, inclination]

    rng_key = random.PRNGKey(SEED)
    init_mass, rng_key = gen_init_mass(rng_key)

    start_time = time.time()
    results = get_optimal_mass(init_mass, freqs, params, fdata_jax, psd_jax, delta_f)
    total_time = time.time() - start_time

    max_snr = max(results.values(), key=lambda x: x["snr"])
    print(f"Max SNR values: mass={max_snr['mass']}, snr={max_snr['snr']}, gradient={max_snr['gradient']}") 

    # Extract mass and snr values from the results dictionary
    mass_values = [result["mass"] for result in results.values()]
    snr_values = [result["snr"] for result in results.values()]
    iter_values = [result["iter"] for result in results.values()]

    # Get the last key in the dictionary
    last_key = list(results.keys())[-1]

    # Access the "iter" value using the last key
    iters = results[last_key]["iter"] + 1

    # Create a plot of the SNR values in the order they were computed
    plt.figure(figsize=(18, 12))
    n = len(mass_values)
    colors = cm.jet(jnp.linspace(-1, 1, n))
    norm = plt.Normalize(-1, n-1)
    mappable = ScalarMappable(norm=norm, cmap=cm.jet)

    # Add "X" markers for initial, peak, and final points
    plt.scatter(mass_values[0], snr_values[0], marker='x', color=colors[0], s=400, linewidth=3, label='Initial')
    plt.scatter(max_snr["mass"], max_snr["snr"], marker='x', color=colors[(max_snr['iter'])], s=400, linewidth=3, label='Peak')
    plt.scatter(mass_values[-1], snr_values[-1], marker='x', color=colors[-1], s=400, linewidth=3, label='Final')

    for i in range(n - 1):
        plt.plot(mass_values[i:i+2], snr_values[i:i+2], "o", color=colors[i], alpha=0.05, markersize=5)

    # Add a custom legend to show the relation between the X markers and the stats box
    custom_legend = [
            plt.Line2D([0], [0], marker='x', color=colors[0], label=(f"Initial: mass: {mass_values[0]:.2f}, SNR: {snr_values[0]:.2f}, iter: {iter_values[0]}"), linestyle='None', markersize=10, markeredgewidth=2),
        plt.Line2D([0], [0], marker='x', color=colors[(max_snr['iter'])], label=(f"Peak: mass: {max_snr['mass']:.2f}, SNR: {max_snr['snr']:.2f}, iter: {max_snr['iter']}"), linestyle='None', markersize=10, markeredgewidth=2),
        plt.Line2D([0], [0], marker='x', color=colors[-1], label=(f"Final: mass: {mass_values[-1]:.2f}, SNR: {snr_values[-1]:.2f}, iter: {iter_values[-1]}"), linestyle='None', markersize=10, markeredgewidth=2)
    ]
    plt.legend(handles=custom_legend, fontsize=21)

    plt.xlabel(f"Mass (solar mass)", fontsize=24)
    plt.ylabel("SNR", fontsize=24)
    plt.title(f"SNR vs Mass {EVENT_NAME} (Time taken: {total_time:.2f} seconds), iterations: {iters}", fontsize=24)
    plt.tick_params(axis='both', which='major', labelsize=24)
    plt.grid(True)

    # Create the colorbar
    cbar = plt.colorbar(mappable, ax=plt.gca())
    cbar.set_label('SNR Index', fontsize=24)
    plt.tight_layout()
    plt.savefig('plot.png', bbox_inches='tight')
    cbar.ax.tick_params(labelsize=24)  # Change font size of colorbar numbers

    plt.show()    
    best_template = gen_waveform(max_snr["mass"], freqs, params)
    best_snr = pre_matched_filter(best_template, fdata_jax, psd_jax, delta_f)
    plt.xlabel(f"SNR Index", fontsize=24)
    plt.ylabel(f"SNR", fontsize=24)
    plt.title(f"SNR Time Series {EVENT_NAME}", fontsize=24)
    plt.tick_params(axis='both', which="major", labelsize=24)

    plt.plot(best_snr)
    plt.show()
    print(delta_f)

    plot_results(results)

if __name__ == "__main__":
    main()
