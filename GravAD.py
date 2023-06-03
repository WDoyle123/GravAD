import jax
from jax import jit, grad, lax, vmap, random
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib import cm
from matplotlib.cm import ScalarMappable
from pycbc.catalog import Merger
from pycbc.psd import interpolate, inverse_spectrum_truncation
from pycbc.filter import resample_to_delta_t, highpass
from pycbc.waveform import get_td_waveform
from pycbc.filter import matched_filter
import pylab
from pycbc.filter import sigma
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
STRAINS = ['H1', 'L1']
EVENTS = [
    "GW150914", "GW151012", "GW151226", "GW170104", "GW170608", "GW170729",
    "GW170809", "GW170814", "GW170817", "GW170818", "GW170823"]

# Currently set to min template 
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
    init_mass1 = 11. # Remove for random mass
    init_mass2 = 11. # ^^^^^^^^^^^^^^^^^^^^^^
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
        return snr[5000:len(snr)-5000]
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
    @jit
    def get_snr(mass1, mass2):
        ftemplate_jax = jnp.array(gen_waveform(mass1, mass2, freqs, params))
        snr = pre_matched_filter(ftemplate_jax, data, psd, delta_f)
        return snr.max()
    return get_snr

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

@jit
def one_step(state, _):
    current_i, mass1, mass2, rng_key, temperature, annealing_rate, data, psd, freqs, params, delta_f = state
    snr_func = make_function(data, psd, freqs, params, delta_f)
    dsnr1 = jit(grad(lambda m1, m2: snr_func(m1, m2), argnums=0))
    dsnr2 = jit(grad(lambda m1, m2: snr_func(m1, m2), argnums=1))

    snr = jax.lax.convert_element_type(snr_func(mass1, mass2), jnp.float32)
    gradient1 = jax.lax.convert_element_type(dsnr1(mass1, mass2), jnp.float32)
    gradient2 = jax.lax.convert_element_type(dsnr2(mass1, mass2), jnp.float32)

    rng_key, subkey1 = random.split(rng_key)
    rng_key, subkey2 = random.split(rng_key)

    learning_rate1 = random.uniform(subkey1, minval=LRL, maxval=LRU)
    learning_rate2 = random.uniform(subkey2, minval=LRL, maxval=LRU)

    perturbation1 = random.normal(subkey1) * temperature
    perturbation2 = random.normal(subkey2) * temperature

    new_mass1 = mass1 + (learning_rate1 * gradient1) + perturbation1
    new_mass1 = jax.lax.cond(new_mass1 < 11., lambda _: 11. + jnp.abs(perturbation1), lambda _: new_mass1, None)
    new_mass1 = jax.lax.cond(new_mass1 > 120., lambda _: 120. - jnp.abs(perturbation1), lambda _: new_mass1, None)
    new_mass1 = jax.lax.cond(jnp.isnan(new_mass1), lambda _: jnp.nan, lambda _: new_mass1, None)

    new_mass2 = mass2 + (learning_rate2 * gradient2) + perturbation2
    new_mass2 = jax.lax.cond(new_mass2 < 11., lambda _: 11. + jnp.abs(perturbation2), lambda _: new_mass2, None)
    new_mass2 = jax.lax.cond(new_mass2 > 120., lambda _: 120. - jnp.abs(perturbation2), lambda _: new_mass2, None)
    new_mass2 = jax.lax.cond(jnp.isnan(new_mass2), lambda _: jnp.nan, lambda _: new_mass2, None)

    temperature *= annealing_rate
    new_state = (current_i+1, new_mass1, new_mass2, rng_key, temperature, annealing_rate, data, psd, freqs, params, delta_f)
    return new_state, (snr, mass1, mass2)

def get_optimal_mass(init_mass1, init_mass2, freqs, params, data, psd, delta_f):
    rng_key = random.PRNGKey(SEED)
    temperature = TEMPERATURE
    annealing_rate = ANNEALING_RATE
    state = (0, init_mass1, init_mass2, rng_key, temperature, annealing_rate, data, psd, freqs, params, delta_f)

    final_state, (snr_hist, mass1_hist, mass2_hist) = lax.scan(one_step, state, jnp.arange(MAX_ITERS))
    return snr_hist, mass1_hist, mass2_hist

def plot_snr_mass(results, total_time, EVENT_NAME, STRAIN, freqs, params, fdata_jax, psd_jax, delta_f):
    
    mass1_values, mass2_values, snr_values, iter_values, combined_mass = sort_results(results)
    max_snr = get_max_snr_array(results, EVENT_NAME, STRAIN, total_time)

    # SNR vs Mass plot
    plt.figure(figsize=(18, 12))
    n = len(mass1_values)
    color_name = cm.jet
    colors = color_name(jnp.linspace(-1, 1, n))
    norm = plt.Normalize(-1, n-1)
    mappable = ScalarMappable(norm=norm, cmap=color_name)
    last_key = list(results.keys())[-1]
    iters = list(range(MAX_ITERS))

    peak_combined_mass = max_snr['mass1'] + max_snr['mass2']

    initial_marker = plt.scatter(combined_mass[0], snr_values[0], marker='x', color=colors[0], s=400, linewidth=3, label=f"Initial: mass1: {mass1_values[0]:.2f}, mass2: {mass2_values[0]:.2f}, SNR: {snr_values[0]:.2f}, iter: {iter_values[0]}")
    peak_marker = plt.scatter(peak_combined_mass, max_snr["snr"], marker='x', color=colors[(max_snr['iter'])], s=400, linewidth=3, label=f"Peak: mass1: {max_snr['mass1']:.2f}, mass2: {max_snr['mass2']:.2f}, SNR: {max_snr['snr']:.2f}, iter: {max_snr['iter']}")
    final_marker = plt.scatter(combined_mass[-1], snr_values[-1], marker='x', color=colors[-1], s=400, linewidth=3, label=f"Final: mass1: {mass1_values[-1]:.2f}, mass2: {mass2_values[-1]:.2f}, SNR: {snr_values[-1]:.2f}, iter: {iter_values[-1]}")

    for i in range(n - 1):
        plt.plot(combined_mass[i:i+2], snr_values[i:i+2], "o", color=colors[i], alpha=0.15, markersize=5)

    plt.legend(fontsize=21)

    plt.xlabel(f"Mass (solar mass)", fontsize=24)
    plt.ylabel("SNR", fontsize=24)
    plt.title(f"SNR vs Mass {EVENT_NAME}({STRAIN}) (Time taken: {total_time:.2f} seconds), iterations: {MAX_ITERS}", fontsize=24)
    plt.tick_params(axis='both', which='major', labelsize=24)
    plt.grid(True)

    # Create the colorbar
    cbar = plt.colorbar(mappable, ax=plt.gca())
    cbar.set_label('SNR Index', fontsize=24)
    plt.tight_layout()
    cbar.ax.tick_params(labelsize=24)  # Change font size of colorbar numbers

    # Save the plot with a filename consisting of the graph title, event name, temperature, annealing_rate, and MAX_ITERS
    filename = f"SNR_vs_Combined_Mass_for_{EVENT_NAME}_{STRAIN}_T_{TEMPERATURE:.2f}_AR_{ANNEALING_RATE:.3f}_MI_{MAX_ITERS}_{LRL}_{LRU}_SEED{SEED}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

    # SNR vs Iteration plot
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
    plt.title(f"SNR vs Iteration {EVENT_NAME}({STRAIN}) (Time taken: {total_time:.2f} seconds), iterations: {MAX_ITERS}", fontsize=24)
    filename = f"SNR_vs_Iterations_for_{EVENT_NAME}_{STRAIN}_T_{TEMPERATURE:.2f}_AR_{ANNEALING_RATE:.3f}_MI_{MAX_ITERS}_{LRL}_{LRU}_SEED{SEED}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

    # SNR time-series plot
    mass1 = max_snr['mass1']
    mass2 = max_snr['mass2']
    snrp = max_snr['snr']
    template = gen_waveform(mass1, mass2, freqs, params)
    snr = pre_matched_filter(template, fdata_jax, psd_jax, delta_f)
    plt.plot(snr)
    plt.grid()
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.tight_layout()
    plt.xlabel("SNR Index", fontsize=18)
    plt.ylabel("SNR", fontsize=24)
    plt.title(f"Optimal Template {EVENT_NAME}({STRAIN}) SNR:{snrp:.2f}", fontsize=18)
    filename = f"Optimal_Template_for_{EVENT_NAME}_{STRAIN}_T_{TEMPERATURE:.2f}_AR_{ANNEALING_RATE:.3f}_MI_{MAX_ITERS}_{LRL}_{LRU}_SEED{SEED}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


def pycbc_plots(EVENT_NAME, STRAIN, conditioned, results, total_time):
    
    # From the PyCBC tutorial 3: https://colab.research.google.com/github/gwastro/pycbc-tutorials/blob/master/tutorial/3_WaveformMatchedFilter.ipynb
    mass1_values, mass2_values, snr_values, iter_values, combined_mass = sort_results(results)
    max_snr = get_max_snr_array(results, EVENT_NAME, STRAIN, total_time)
    merger = Merger(EVENT_NAME)

    # GravAD calculated Masses
    mass1 = max_snr['mass1']
    mass2 = max_snr['mass2']

    hp, hc = get_td_waveform(approximant="SEOBNRv4_opt",
                            mass1=mass1,
                            mass2=mass2,
                            delta_t=1.0/2048,
                            f_lower=15)

    hp.resize(len(conditioned))

    psd = conditioned.psd(4)
    psd = interpolate(psd, conditioned.delta_f)
    psd = inverse_spectrum_truncation(psd, int(4 * conditioned.sample_rate),
                                  low_frequency_cutoff=15)
    template = hp.cyclic_time_shift(hp.start_time)
    snr = matched_filter(template, conditioned,
                     psd=psd, low_frequency_cutoff=20)
    snr = snr.crop(4 + 4, 4)

    fig, axs = plt.subplots(2, 1, figsize=[15, 7], sharex=True)  # Share the x-axis

    peak = abs(snr).numpy().argmax()
    snrp = snr[peak]
    time = snr.sample_times[peak]

    dt = time - conditioned.start_time
    aligned = template.cyclic_time_shift(dt)

    # scale the template so that it would have SNR 1 in this data
    aligned /= sigma(aligned, psd=psd, low_frequency_cutoff=20.0)

    # Scale the template amplitude and phase to the peak value
    aligned = (aligned.to_frequencyseries() * snrp).to_timeseries()
    aligned.start_time = conditioned.start_time

    # We do it this way so that we can whiten both the template and the data
    white_data = (conditioned.to_frequencyseries() / psd**0.5).to_timeseries()

    # apply a smoothing of the turnon of the template to avoid a transient
    # from the sharp turn on in the waveform.
    tapered = aligned.highpass_fir(30, 512, remove_corrupted=False)
    white_template = (tapered.to_frequencyseries() / psd**0.5).to_timeseries()

    white_data = white_data.highpass_fir(30., 512).lowpass_fir(300, 512)
    white_template = white_template.highpass_fir(30, 512).lowpass_fir(300, 512)

    # Select the time around the merger
    start_time = merger.time - 0.2
    end_time = merger.time + 0.1

    # Index the SNR plot and template plot to the selected time range
    snr_index = (snr.sample_times >= start_time) & (snr.sample_times <= end_time)
    template_index = (white_template.sample_times >= start_time) & (white_template.sample_times <= end_time)

    # First plot
    axs[0].plot(snr.sample_times[snr_index], abs(snr)[snr_index])
    axs[0].set_ylabel('Signal-to-noise', fontsize=18)
    axs[0].set_title(f"SNR Time-series for Peak Template, {EVENT_NAME}({STRAIN})", fontsize=21)
    axs[0].tick_params(axis='both', which='major', labelsize=14)

    # Second plot
    axs[1].plot(white_template.sample_times[template_index], white_data[template_index], label="Data")
    axs[1].plot(white_template.sample_times[template_index], white_template[template_index], label="Template")
    axs[1].set_xlabel('Time (s)', fontsize=18)
    axs[1].set_ylabel('Strain', fontsize=18)
    axs[1].set_title(f"Template against GW signal, {EVENT_NAME}({STRAIN})", fontsize=21)
    axs[1].legend()

    filename = f"snr_and_aligned_{EVENT_NAME}_{STRAIN}_T_{TEMPERATURE:.2f}_AR_{ANNEALING_RATE:.3f}_MI_{MAX_ITERS}_{LRL}_{LRU}_SEED{SEED}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

    subtracted = conditioned - aligned

    # Plot the original data and the subtracted signal data
    fig, axs = plt.subplots(2, 1, figsize=[15, 8], sharex=True)  # Here we make x-axes shared

    # Data for the two plots
    data_list = [(conditioned, 'Original_{}_Data_({})'.format(STRAIN, EVENT_NAME)),
                 (subtracted, 'Signal_Subtracted_from_{}_Data_({})'.format(STRAIN, EVENT_NAME))]

    # Loop over the data and create the subplots
    for i, (data, title) in enumerate(data_list):
        t, f, p = data.whiten(4, 4).qtransform(.001,
                                               logfsteps=100,
                                               qrange=(8, 8),
                                               frange=(20, 512))

        # Specify the plot to be modified
        ax = axs[i]
        ax.set_title(title, fontsize=21)
        img = ax.pcolormesh(t, f, p**0.5, vmin=1, vmax=6)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.set_yscale('log')
        ax.set_ylabel('Frequency (Hz)', fontsize=18)
        ax.set_xlim(merger.time - 2, merger.time + 1)

    # Label x-axis at the end (as x-axis is shared)
    axs[1].set_xlabel('Time (s)', fontsize=18)

    # Adjusting the spaces between the plots
    plt.subplots_adjust(hspace=0.5)

    filename = "contours_{}_{}_T_{:.2f}_AR_{:.3f}_MI_{}_{}_SEED{}.png".format(EVENT_NAME, STRAIN, TEMPERATURE, ANNEALING_RATE, MAX_ITERS, LRL, LRU, SEED)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def sort_results(results):
    mass1_values = []
    mass2_values = []
    snr_values = []
    combined_mass = []

    for result in results.values():
        mass1_values.append(jnp.array(result['mass1s'], dtype=jnp.float32).item())
        mass2_values.append(jnp.array(result['mass2s'], dtype=jnp.float32).item())
        combined_mass.append(jnp.array(result['mass1s'] + result['mass2s'], dtype=jnp.float32))
        snr_values.append(jnp.array(result['snr'], dtype=jnp.float32).item())

    iter_values = list(range(MAX_ITERS))

    return mass1_values, mass2_values, snr_values, iter_values, combined_mass

def preprocess(event, strain):

    try:
        merger = Merger(event)
        strain = merger.strain(strain) * 1e22
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

        return fdata_jax, delta_f, psd_jax, conditioned

    except Exception as e:
        print(f"Error processing event {event} and strain {strain}: {str(e)}")
        return None, None, None

def precompile():

    print("Beginning Compilation...")
    fdata_jax, delta_f, psd_jax, _ = preprocess("GW150914", "H1")
    chi1 = 0.
    chi2 = 0.
    tc = 1.0
    phic = 1.
    dist_mpc = 440.
    inclination = 0.
    params = [chi1, chi2, dist_mpc, tc, phic, inclination]
    freqs = frequency_series(delta_f)
    rng_key = random.PRNGKey(SEED)
    init_mass1, init_mass2, rng_key = gen_init_mass(rng_key)
    results = get_optimal_mass(init_mass1, init_mass2, freqs, params, fdata_jax, psd_jax, delta_f)
    print("Compiled! \n")

    return None

def frequency_series(delta_f):
    nyquist_freq = (SAMPLING_RATE / 2)
    return jnp.arange(0, nyquist_freq + delta_f, delta_f)

def get_max_snr_array(results, EVENT_NAME, STRAIN, total_time):
    max_snr = max(results.values(), key=lambda x: x['snr'])
    max_snr['iter'] = jnp.asarray(max_snr['iters']).item()
    max_snr['mass1'] = jnp.asarray(max_snr['mass1s']).item()
    max_snr['mass2'] = jnp.asarray(max_snr['mass2s']).item()
    max_snr['final_mass'] = (max_snr['mass1'] + max_snr['mass2'])
    max_snr['event_name'] = EVENT_NAME
    max_snr['strain'] = STRAIN
    max_snr['time'] = total_time

    filename = f"results_for_{EVENT_NAME}_{STRAIN}_T_{TEMPERATURE:.2f}_AR_{ANNEALING_RATE:.3f}_MI_{MAX_ITERS}_{LRL}_{LRU}_SEED{SEED}.txt"
    with open(f"{filename}", "w") as f:
        f.write(str(max_snr))

    return max_snr

def main():

    # compile the code --> allows for accurate timings
    precompile()

    # Test every GW event with each detector
    all_results = []

    for event in EVENTS:
        for strain in STRAINS:

            print(f"Analysing Event: {event}, Strain: {strain}")

            fdata_jax, delta_f, psd_jax, conditioned = preprocess(event, strain)

            # Calculate the inverse PSD
            inverse_psd_jax = 1 / psd_jax

            freqs = frequency_series(delta_f)
            time_series = abs(jnp.fft.ifft(freqs))

            # Define waveform parameters
            chi1 = 0.
            chi2 = 0.
            tc = 1.0
            phic = 1.
            dist_mpc = 440.
            inclination = 0.
            params = [chi1, chi2, dist_mpc, tc, phic, inclination]

            rng_key = random.PRNGKey(SEED)
            init_mass1, init_mass2, rng_key = gen_init_mass(rng_key)

            make_function(fdata_jax, psd_jax, freqs, params, delta_f)
            start_time = time.time()
            snr,  mass1, mass2 = get_optimal_mass(init_mass1, init_mass2, freqs, params, fdata_jax, psd_jax, delta_f)
            total_time = time.time() - start_time

            snr = jnp.array(snr)
            mass1 = jnp.array(mass1)
            mass2 = jnp.array(mass2)

            results = {}
            iterations = list(range(MAX_ITERS+1))

            for i in range(MAX_ITERS):
                results[i] = {'snr': snr[i], 'mass1s': mass1[i], 'mass2s': mass2[i], 'iters': iterations[i]}
            
            # tracing type plots aswell as SNR time-series
            plot_snr_mass(results, total_time, event, strain, freqs, params, fdata_jax, psd_jax, delta_f)
            
            # contour, alignment and SNR time-series plots
            pycbc_plots(event, strain, conditioned, results, total_time)

            all_results.append(get_max_snr_array(results, event, strain, total_time))

        filename = f"all_results_for_T_{TEMPERATURE:.2f}_AR_{ANNEALING_RATE:.3f}_MI_{MAX_ITERS}_{LRL}_{LRU}_SEED{SEED}.txt"
        with open(f"{filename}", "w") as f:
             f.write(str(all_results))

if __name__ == "__main__":
    main()
