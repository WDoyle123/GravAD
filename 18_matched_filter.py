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
MAX_ITERS = 500
TEMPERATURE = 1
ANNEALING_RATE = 0.99
LRU = 1.5 # Learning Rate Upper
LRL = 5.5 # Learning Rate Lower
SEED = 1

def gen_init_mass(rng_key):
    """
    Generate an initial mass value uniformly between 20 and 80

    Args:
    rng_key (jax.random.PRNGKey): A JAX random key used to generate random numbers

    Returns:
    tuple: The generated initial mass value and an updated random key
    """
    rng_key, subkey1 = random.split(rng_key)
    rng_key, subkey2 = random.split(rng_key)
    init_mass1 = random.uniform(subkey1, minval = 20., maxval = 80.)
    init_mass2 = random.uniform(subkey2, minval = 20., maxval= 80.)
    init_mass1 = 11.
    init_mass2 = 11.
    return init_mass1, init_mass2, rng_key

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
    sigmasq = 4 * jnp.sum(template * template.conj() * inverse_psd) * delta_f
    return jnp.sqrt(jnp.abs(sigmasq) + 1e-9)

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
def gen_waveform(mass1, mass2, freqs, params):
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
    Mc, eta = ms_to_Mc_eta(jnp.array([mass1, mass2]))
    
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
    def snr(mass1, mass2):
        ftemplate_jax = jnp.array(gen_waveform(mass1, mass2, freqs, params))
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

def get_optimal_mass(init_mass1, init_mass2, freqs, params, data, psd, delta_f):
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
    mass1 = init_mass1
    mass2 = init_mass2
    snr_func = make_function(data, psd, freqs, params, delta_f)
    dsnr1 = jit(grad(lambda m1, m2: snr_func(m1, m2), argnums=0))
    dsnr2 = jit(grad(lambda m1, m2: snr_func(m1, m2), argnums=1))
    rng_key = random.PRNGKey(SEED)

    # Set initial temperature and the rate of temperature decrease
    temperature = TEMPERATURE
    annealing_rate = ANNEALING_RATE

    for i in range(MAX_ITERS):
        
        snr = float(snr_func(mass1, mass2))
        gradient1 = float(dsnr1(mass1, mass2))
        gradient2 = float(dsnr2(mass1, mass2))

        rng_key, subkey1 = random.split(rng_key)
        rng_key, subkey2 = random.split(rng_key)

        learning_rate1 = random.uniform(subkey1, minval=LRL, maxval=LRU)
        learning_rate2 = random.uniform(subkey2, minval=LRL, maxval=LRU)

        # Add random perturbation to the mass update step
        perturbation1 = random.normal(subkey1) * temperature
        perturbation2 = random.normal(subkey2) * temperature

        results[i] = {"mass1": mass1, "mass2": mass2, "snr": snr, "gradient1": gradient1, "gradient2": gradient2, "iter": i}

        mass1 = mass1 + (learning_rate1 * gradient1) + perturbation1
        if mass1 < 11.:
            mass1 = 11. + abs(perturbation1)
        if mass1 > 120.:
            mass1 = 120. - abs(perturbation1)
        if math.isnan(mass1):
            break
        mass2 = mass2 + (learning_rate2 * gradient2) + perturbation2
        if mass2 < 11.:
            mass2 = 11. + abs(perturbation2)
        if mass2> 120.:
             mass2 = 120. - abs(perturbation2)
        if math.isnan(mass2):
            break

        # Update temperature
        temperature *= annealing_rate

    return results

def plot_snr_mass(results, total_time, EVENT_NAME):
    
    mass1_values, mass2_values, snr_values, iter_values, combined_mass = sort_results(results)
    max_snr = get_max_snr_array(results, EVENT_NAME)
    plt.figure(figsize=(18, 12))
    n = len(mass1_values)
    color_name = cm.jet
    colors = color_name(jnp.linspace(-1, 1, n))
    norm = plt.Normalize(-1, n-1)
    mappable = ScalarMappable(norm=norm, cmap=color_name)
    last_key = list(results.keys())[-1]
    iters = results[last_key]["iter"] + 1

    peak_combined_mass = max_snr['mass1'] + max_snr['mass2']
    print(peak_combined_mass)

    initial_marker = plt.scatter(combined_mass[0], snr_values[0], marker='x', color=colors[0], s=400, linewidth=3, label=f"Initial: mass1: {mass1_values[0]:.2f}, mass2: {mass2_values[0]:.2f}, SNR: {snr_values[0]:.2f}, iter: {iter_values[0]}")
    peak_marker = plt.scatter(peak_combined_mass, max_snr["snr"], marker='x', color=colors[(max_snr['iter'])], s=400, linewidth=3, label=f"Peak: mass1: {max_snr['mass1']:.2f}, mass2: {max_snr['mass2']:.2f}, SNR: {max_snr['snr']:.2f}, iter: {max_snr['iter']}")
    final_marker = plt.scatter(combined_mass[-1], snr_values[-1], marker='x', color=colors[-1], s=400, linewidth=3, label=f"Final: mass1: {mass1_values[-1]:.2f}, mass2: {mass2_values[-1]:.2f}, SNR: {snr_values[-1]:.2f}, iter: {iter_values[-1]}")

    for i in range(n - 1):
        plt.plot(combined_mass[i:i+2], snr_values[i:i+2], "o", color=colors[i], alpha=0.15, markersize=5)

    plt.legend(fontsize=21)

    plt.xlabel(f"Mass (solar mass)", fontsize=24)
    plt.ylabel("SNR", fontsize=24)
    plt.title(f"SNR vs Mass {EVENT_NAME} (Time taken: {total_time:.2f} seconds), iterations: {iters}", fontsize=24)
    plt.tick_params(axis='both', which='major', labelsize=24)
    plt.grid(True)

    # Create the colorbar
    cbar = plt.colorbar(mappable, ax=plt.gca())
    cbar.set_label('SNR Index', fontsize=24)
    plt.tight_layout()
    cbar.ax.tick_params(labelsize=24)  # Change font size of colorbar numbers

    # Save the plot with a filename consisting of the graph title, event name, temperature, annealing_rate, and MAX_ITERS
    filename = f"SNR_vs_Combined_Mass_for_{EVENT_NAME}_T_{TEMPERATURE:.2f}_AR_{ANNEALING_RATE:.3f}_MI_{MAX_ITERS}_{LRL}_{LRU}_SEED{SEED}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

    ################
    # SECOND GRAPH #
    ################
    colormap = cm.jet
    norm = plt.Normalize(vmin=min(combined_mass), vmax=max(combined_mass))

    plt.figure(figsize=(18, 12))
    plt.scatter(iter_values, snr_values, c=combined_mass, cmap=colormap, norm=norm)

    mappable = ScalarMappable(norm=norm, cmap=colormap)
    cbar = plt.colorbar(mappable, ax=plt.gca())
    cbar.set_label('Mass Index', fontsize=24)
    cbar.ax.tick_params(labelsize=24)  # Change font size of colorbar numbers

    plt.tick_params(axis='both', which='major', labelsize=24)

    plt.tight_layout(pad=5.0)  # Add padding for title
    plt.grid(True)

    plt.xlabel("Iteration", fontsize=24)
    plt.ylabel("SNR", fontsize=24)
    plt.title(f"SNR vs Iteration {EVENT_NAME} (Time taken: {total_time:.2f} seconds), iterations: {iters}", fontsize=24)
    filename = f"SNR_vs_Iterations_for_{EVENT_NAME}_T_{TEMPERATURE:.2f}_AR_{ANNEALING_RATE:.3f}_MI_{MAX_ITERS}_{LRL}_{LRU}_SEED{SEED}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def get_max_snr_array(results, EVENT_NAME):
    max_snr = max(results.values(), key=lambda x: x['snr'])
    max_snr['mass1'] = jnp.asarray(max_snr['mass1']).item()
    max_snr['mass2'] = jnp.asarray(max_snr['mass2']).item()
    max_snr['event_name'] = EVENT_NAME

    filename = f"results_for_{EVENT_NAME}_T_{TEMPERATURE:.2f}_AR_{ANNEALING_RATE:.3f}_MI_{MAX_ITERS}_{LRL}_{LRU}_SEED{SEED}.txt"
    with open(filename, "w") as f:
        f.write(str(max_snr))

    return max_snr


def sort_results(results):
    mass1_values = []
    mass2_values = []
    snr_values = []
    iter_values = []
    combined_mass = []

    for result in results.values():
        mass1_values.append(result['mass1'])
        mass2_values.append(result['mass2'])
        snr_values.append(result['snr'])
        iter_values.append(result['iter'])
        combined_mass.append(result['mass1'] + result['mass2'])

    return mass1_values, mass2_values, snr_values, iter_values, combined_mass
    
def main():

    events = [
    "GW150914", "GW151012", "GW151226", "GW170104", "GW170608", "GW170729",
    "GW170809", "GW170814", "GW170817", "GW170818", "GW170823",
    "GW190403_051519", "GW190408_181802", "GW190412", "GW190413_052954",
    "GW190413_134308", "GW190421_213856", "GW190424_180648", "GW190425",
    "GW190426_152155", "GW190426_190642", "GW190503_185404",
    "GW190512_180714", "GW190513_205428", "GW190514_065416",
    "GW190517_055101", "GW190519_153544", "GW190521", "GW190521_074359",
    "GW190527_092055", "GW190602_175927", "GW190620_030421",
    "GW190630_185205", "GW190701_203306", "GW190706_222641",
    "GW190707_093326", "GW190708_232457", "GW190719_215514",
    "GW190720_000836", "GW190725_174728", "GW190727_060333",
    "GW190728_064510", "GW190731_140936", "GW190803_022701",
    "GW190805_211137", "GW190814", "GW190828_063405", "GW190828_065509",
    "GW190909_114149", "GW190910_112807", "GW190915_235702",
    "GW190916_200658", "GW190917_114630", "GW190924_021846",
    "GW190925_232845", "GW190926_050336", "GW190929_012149",
    "GW190930_133541", "GW191103_012549", "GW191105_143521",
    "GW191109_010717", "GW191113_071753", "GW191126_115259",
    "GW191127_050227", "GW191129_134029", "GW191204_110529",
    "GW191204_171526", "GW191215_223052", "GW191216_213338",
    "GW191219_163120", "GW191222_033537", "GW191230_180458",
    "GW200112_155838", "GW200115_042309", "GW200128_022011",
    "GW200129_065458", "GW200202_154313", "GW200208_130117",
    "GW200208_222617", "GW200209_085452", "GW200210_092254",
    "GW200216_220804", "GW200219_094415", "GW200220_061928",
    "GW200220_124850", "GW200224_222234", "GW200225_060421",
    "GW200302_015811", "GW200306_093714", "GW200308_173609",
    "GW200311_115853", "GW200316_215756", "GW200322_091133"
    ]


    all_results = []

    for event in events:
        EVENT_NAME = event
    
        try:
            # Load the data for event
            merger = Merger(EVENT_NAME)
        except:
            continue

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
        dist_mpc = 440.
        inclination = 0.
        params = [chi1, chi2, dist_mpc, tc, phic, inclination]

        rng_key = random.PRNGKey(SEED)
        init_mass1, init_mass2, rng_key = gen_init_mass(rng_key)

        start_time = time.time()
        results = get_optimal_mass(init_mass1, init_mass2, freqs, params, fdata_jax, psd_jax, delta_f)
        total_time = time.time() - start_time
    
        plot_snr_mass(results, total_time, EVENT_NAME)
        
        all_results.append(get_max_snr_array(results, EVENT_NAME))

    filename = f"all_results_for_T_{TEMPERATURE:.2f}_AR_{ANNEALING_RATE:.3f}_MI_{MAX_ITERS}_{LRL}_{LRU}_SEED{SEED}.txt"
    with open(filename, "w") as f:
        f.write(str(all_results))


if __name__ == "__main__":
    main()
