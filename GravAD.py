import jax
from jax import jit, grad, lax, vmap, random
import jax.numpy as jnp
from pycbc.catalog import Merger
from pycbc.psd import interpolate, inverse_spectrum_truncation
from pycbc.filter import resample_to_delta_t, highpass
from ripple.waveforms import IMRPhenomD
from ripple import ms_to_Mc_eta
import math
import time
import pickle
import re
import os
import glob

import helper
import plots
from to_csv import process_files
from constants import *



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
    """
    Simulates one step of simulated annealing. Calculates the gradient of the snr with respect to the masses,
    updates the masses using the learning rate and the gradients, then applies a random perturbation to the masses.

    Args:
    state (tuple): Current state of the system, includes the mass values, RNG key, temperature, annealing rate, and other parameters.

    Returns:
    tuple: The updated state of the system and the snr and mass values.
    """
    # Unpack the state variables
    current_i, mass1, mass2, rng_key, temperature, annealing_rate, data, psd, freqs, params, delta_f = state

    # Function to get SNR
    snr_func = make_function(data, psd, freqs, params, delta_f)

    # Define gradients of SNR function with respect to each mass
    dsnr1 = jit(grad(lambda m1, m2: snr_func(m1, m2), argnums=0))
    dsnr2 = jit(grad(lambda m1, m2: snr_func(m1, m2), argnums=1))

    # Get SNR using current state
    snr = jax.lax.convert_element_type(snr_func(mass1, mass2), jnp.float32)
    
    # Get gradient of SNR 
    gradient1 = jax.lax.convert_element_type(dsnr1(mass1, mass2), jnp.float32)
    gradient2 = jax.lax.convert_element_type(dsnr2(mass1, mass2), jnp.float32)

    # Get RNG 
    rng_key, subkey1 = random.split(rng_key)
    rng_key, subkey2 = random.split(rng_key)

    # Randomise the learning rate
    learning_rate1 = random.uniform(subkey1, minval=LRL, maxval=LRU)
    learning_rate2 = random.uniform(subkey2, minval=LRL, maxval=LRU)

    # Generate perturbations scaled by the temperature (simulated annealing)
    perturbation1 = random.normal(subkey1) * temperature
    perturbation2 = random.normal(subkey2) * temperature

    # Update mass1
    new_mass1 = mass1 + (learning_rate1 * gradient1) + perturbation1

    # Avoid mass1 falling below 11 solar masses and above 120 solar masses
    new_mass1 = jax.lax.cond(new_mass1 < 11., lambda _: 11. + jnp.abs(perturbation1), lambda _: new_mass1, None)
    new_mass1 = jax.lax.cond(new_mass1 > 120., lambda _: 120. - jnp.abs(perturbation1), lambda _: new_mass1, None)
    new_mass1 = jax.lax.cond(jnp.isnan(new_mass1), lambda _: jnp.nan, lambda _: new_mass1, None)
    
    # Update mass2
    new_mass2 = mass2 + (learning_rate2 * gradient2) + perturbation2

    # Avoid mass2 falling below 11 solar masses and above 120 solar masses
    new_mass2 = jax.lax.cond(new_mass2 < 11., lambda _: 11. + jnp.abs(perturbation2), lambda _: new_mass2, None)
    new_mass2 = jax.lax.cond(new_mass2 > 120., lambda _: 120. - jnp.abs(perturbation2), lambda _: new_mass2, None)
    new_mass2 = jax.lax.cond(jnp.isnan(new_mass2), lambda _: jnp.nan, lambda _: new_mass2, None)
    
    # Update the temperature for the annealing
    temperature *= annealing_rate

    # Create a new state
    new_state = (current_i+1, new_mass1, new_mass2, rng_key, temperature, annealing_rate, data, psd, freqs, params, delta_f)

    return new_state, (snr, mass1, mass2)

def get_optimal_mass(init_mass1, init_mass2, freqs, params, data, psd, delta_f):
    """
    Function to find the optimal mass values. It initializes the system state, and then runs the 
    one_step function for a set number of iterations.

    Args:
    init_mass1 (float): Initial value for mass1
    init_mass2 (float): Initial value for mass2
    freqs (jax.numpy.array): Frequency array from 0 to nyquist frequency with step size of delta_f
    params (list): List of parameters for waveform
    data (jax.numpy.array): Data frequency series array
    psd (jax.numpy.array): Power spectral density frequecy series array
    delta_f (float): Frequency step size

    Returns:
    tuple: History of SNR, mass1, and mass2 values.
    """
    # Generate rng key
    rng_key = random.PRNGKey(SEED)
    
    # Define the initial state of the system
    temperature = TEMPERATURE
    state = (0, init_mass1, init_mass2, rng_key, temperature, ANNEALING_RATE, data, psd, freqs, params, delta_f)
    
    snr_hist = []
    mass1_hist = []
    mass2_hist = []
    peak_snr = 0.0
    peak_iter = 0

    for i in range(MAX_ITERS):
        state, (snr, mass1, mass2) = one_step(state, i)
        snr_hist.append(snr)
        mass1_hist.append(mass1)
        mass2_hist.append(mass2)

        if snr > peak_snr:
            peak_snr = snr
            peak_iter = i
        if i - peak_iter > 10:
            state = state[:4] + (ADV_TEMPERATURE,) + state[5:]  # Create a new tuple with the updated temperature
        if i - peak_iter > 25:
            break
     
    return snr_hist, mass1_hist, mass2_hist

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

def jit_compile():

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
    max_snr = {}
    all_max_snr = max(results.values(), key=lambda x: x['snr'])
    max_snr['event_name'] = EVENT_NAME
    max_snr['strain'] = STRAIN
    max_snr['snr'] = jnp.asarray(all_max_snr['snr']).item()
    max_snr['mass1'] = jnp.asarray(all_max_snr['mass1s']).item()
    max_snr['mass2'] = jnp.asarray(all_max_snr['mass2s']).item()
    max_snr['final_mass'] = (all_max_snr['mass1s'] + all_max_snr['mass2s'])
    max_snr['iter'] = jnp.asarray(all_max_snr['iters']).item()

    # Get all keys, sort them, and get the largest one (last)
    final_key = sorted(list(results.keys()))[-1]

    # Get the final dictionary
    final_iter = results[final_key] 

    # Get 'iters' value from the final entry
    max_snr['total_iters'] = final_iter['iters']

    max_snr['time'] = total_time

    filename = f"{EVENT_NAME}_{STRAIN}_max_snr_T_{TEMPERATURE:.2f}_AR_{ANNEALING_RATE:.3f}_MI_{MAX_ITERS}_{LRL}_{LRU}_SEED{SEED}.pkl"

    if STRAIN == "simulated_signal":
        folder = "test_graphs/max_snr_simulated"
    else:
        folder = "test_graphs/max_snr_real"

    helper.save_pickle(folder, filename, max_snr)

    return max_snr

def real_signals():
    """
    This function processes real gravitational wave signals and calculates their SNR. It also generates optimal mass
    for the signals and saves the results for future analysis.
    """
    all_results = []

    for event in EVENTS:
        combined_snr = []
        for strain in STRAINS:

             # Start processing the signal for the specific event and strain
             print(f"Analysing Event: {event}, Strain: {strain}")

             # Preprocess the data to get the conditioned data, PSD, and frequency data
             fdata_jax, delta_f, psd_jax, conditioned = preprocess(event, strain)

             # Calculate the inverse PSD for the event
             inverse_psd_jax = 1 / psd_jax

             # Get the frequency series for the signal
             freqs = frequency_series(delta_f)

             # Generate a random key and initial masses
             rng_key = random.PRNGKey(SEED)
             init_mass1, init_mass2, rng_key = gen_init_mass(rng_key)

             # Make the function with the initial parameters and data
             make_function(fdata_jax, psd_jax, freqs, params, delta_f)
             
             # Start the timer to monitor the performance
             start_time = time.time()
             
             # Get the optimal mass for the signal
             snr,  mass1, mass2 = get_optimal_mass(init_mass1, init_mass2, freqs, params, fdata_jax, psd_jax, delta_f)
             
             # Calculate the total time taken for the process
             total_time = time.time() - start_time

             # Convert the results into array format
             snr = jnp.array(snr)
             mass1 = jnp.array(mass1)
             mass2 = jnp.array(mass2)
             iterations = jnp.array([i for i in range(len(snr))])
             results = {}

             # Store the results in a dictionary for each iteration
             for s, m1, m2, i in zip(snr, mass1, mass2, iterations):
                results[int(i)] = {'snr': float(s), 'mass1s': float(m1), 'mass2s': float(m2), "combined_mass": float(m1 + m2), 'iters': int(i)}
            
             # Save the results for future analysis and graphing
             filename = f"{event}_{strain}_results.pkl"
             folder = "test_graphs/results_real"
             helper.save_pickle(folder, filename, results)
             
             # Get the parameters that achieved the highest SNR
             max_snr = get_max_snr_array(results, event, strain, total_time)

             # Collect the SNRs for each detector of the same event
             combined_snr.append(max_snr['snr'])

             # Print out the total time taken and the number of iterations performed
             print(f"Time Taken: {total_time:.2f}, Templates:{iterations[-1]}")

             # Append the results to the all_results list
             all_results.append(get_max_snr_array(results, event, strain, total_time))

        # Calculate the combined SNR for the event and print it
        snr_for_event = jnp.sqrt(combined_snr[0]**2 + combined_snr[1]**2)
        print(f"SNR for {event}: {snr_for_event:.2f}\n")

        # Save all the results in a text file for later analysis
        filename = f"all_results_for_T_{TEMPERATURE:.2f}_AR_{ANNEALING_RATE:.3f}_MI_{MAX_ITERS}_{LRL}_{LRU}_SEED{SEED}.txt"
        folder = "test_graphs/all_results"
        helper.save_txt(folder, filename, all_results)

def simulated_signals():

    pattern = r'(\d+_\d+)'
    folder = "simulated_signals/"
    files = os.listdir(folder)
    files.sort()
    
    merger = Merger("GW150914")
    strain = merger.strain("H1") * 1e22
    strain = resample_to_delta_t(highpass(strain, 15.0), 1 / 2048)
    conditioned = strain.crop(2, 2)
    delta_f = conditioned.delta_f
    psd_jax = psd_func(conditioned)
    freqs = frequency_series(delta_f)
    #psd_jax = jnp.asarray([1] * len(freqs))
    rng_key = random.PRNGKey(SEED)
    init_mass1, init_mass2, rng_key = gen_init_mass(rng_key)

    for file in files:
        file_path = os.path.join(folder, file)
    
        with open(file_path, "rb") as f:
            fdata = pickle.load(f)
      
        print(file)
        fdata_jax = jnp.asarray(fdata)

        make_function(fdata_jax, psd_jax, freqs, params, delta_f)
        start_time = time.time()
        snr, mass1, mass2 = get_optimal_mass(init_mass1, init_mass2, freqs, params, fdata_jax, psd_jax, delta_f)
        total_time = time.time() - start_time

        snr = jnp.array(snr)
        mass1 = jnp.array(mass1)
        mass2 = jnp.array(mass2)
        iterations = jnp.array([i for i in range(len(snr))])
        results = {}

        # Store the results in a dictionary for each iteration
        for s, m1, m2, i in zip(snr, mass1, mass2, iterations):
            results[int(i)] = {'snr': float(s), 'mass1s': float(m1), 'mass2s': float(m2), "combined_mass": float(m1 + m2), 'iters': int(i)}

        match = re.search(pattern, file)
        event = match.group(1)

        # Save the results for future analysis and graphing
        filename = f"{event}_results.pkl"
        folder_save = "test_graphs/results_simulated"
        helper.save_pickle(folder_save, filename, results)

        max_snr = get_max_snr_array(results, event, "simulated_signal" , total_time)
        snrp = max_snr['snr']
        mass1 = max_snr['mass1']
        mass2 = max_snr['mass2']

        # Print out the total time taken and the number of iterations performed
        print(f"Time Taken: {total_time:.2f}, Templates:{iterations[-1]}, SNR: {snrp}, mass1: {mass1}, mass2: {mass2}")



def plotter(pattern, results_dir, max_snr_dir):
    """
    This function plots the signal-to-noise ratio (SNR) vs. mass for a set of results.
    It searches through a results folder and plot folders to match and plot relevant data.
    """
    # List all files in the results directory
    results_files = os.listdir(results_dir)
    
    # Sort the list of files for consistency
    results_files.sort()

    # Iterate through each file in the results directory
    for results_file in results_files:

        # Create the full path to the results file
        results_file_path = os.path.join(results_dir, results_file)

        # Open and unpickle the results file
        with open(results_file_path, "rb") as f:
            results = pickle.load(f)

        # Match the filename to the pattern to extract the event and strain
        match = re.match(pattern, results_file)
        if match:
            event = match.group(1)
            strain = match.group(2)
        else:
            print(f"Pattern not found in the file name: {results_file}")
            continue  # Skip to the next file if pattern not found

        print(f"Plotting: {event}_{strain}")
        
        # The pattern to match for max_snr files
        max_snr_file_pattern = f"{event}_{strain}*"
        
        # Find all files in the snr_dir that match the snr_file_pattern
        matching_max_snr_files = glob.glob(os.path.join(max_snr_dir, max_snr_file_pattern))

        # Loop over each matching file
        for matching_max_snr_file in matching_max_snr_files:
            
            # Open and unpickle the snr file
            with open(matching_max_snr_file, "rb") as f:
                max_snr = pickle.load(f)

            total_time = max_snr['time']

            # Plot the snr vs mass for the given event and strain
            print("Plotting mass vs snr")
            plots.plot_snr_vs_mass(event, strain, results, max_snr)

            print("Plotting snr vs iteration")
            plots.plot_snr_vs_iteration(event, strain, total_time, results)

            print("Plotting snr timeseries")
            plots.plot_snr_timeseries(event, strain, max_snr)

            print("Plotting alignment and contours\n")
            plots.pycbc_plots(event, strain, total_time, max_snr)


def main():
    
    pattern_real = r'(\w+)_(\w+)_.*'
    pattern_simu = r'(\d+)_(\d+)_.*'

    folder_results_real = "test_graphs/results_real"
    folder_results_sim  = "test_graphs/results_simulated"

    folder_max_snr_real = "test_graphs/max_snr_real"
    folder_max_snr_sim  = "test_graphs/max_snr_simulated"


    # Clear folders
    helper.clear_folder("test_graphs")

    # Compile the code --> allows for accurate timings
    jit_compile()

    # Analyse real signals
    if True:
        real_signals()
        plotter(pattern_real, folder_results_real, folder_max_snr_real)
    
    # Analyse simulated signals (WIP)
    if True:
        simulated_signals()
        plotter(pattern_simu, folder_results_sim, folder_max_snr_sim)

    # Get a compilation of results in csv
    process_files()

if __name__ == "__main__":
    main()
