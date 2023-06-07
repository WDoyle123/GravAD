import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.cm import ScalarMappable
from pycbc.waveform import get_td_waveform
from pycbc.filter import matched_filter
from pycbc.catalog import Merger
from pycbc.psd import interpolate, inverse_spectrum_truncation
from pycbc.filter import resample_to_delta_t, highpass
from pycbc.filter import sigma
import pylab

from GravAD import MAX_ITERS, SEED, TEMPERATURE, ANNEALING_RATE, LRL, LRU
from GravAD import gen_waveform, pre_matched_filter

def format_plot(ax, xlabel, ylabel, title, fontsize=24):
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize)
    ax.grid(True)

def create_colorbar(ax, mappable, label, fontsize=24):
    cbar = plt.colorbar(mappable, ax=ax)
    cbar.set_label(label, fontsize=fontsize)
    cbar.ax.tick_params(labelsize=fontsize)
    return cbar

def save_plot(filename):
    plt.savefig("test_graphs/" + filename, dpi=300, bbox_inches='tight')
    plt.close()

def plot_markers(ax, combined_mass, snr_values, mass1_values, mass2_values, iter_values, colors, max_snr):
    markers = {
        'Initial': (combined_mass[0], snr_values[0], mass1_values[0], mass2_values[0], snr_values[0], iter_values[0], colors[0]),
        'Peak': (max_snr['mass1'] + max_snr['mass2'], max_snr['snr'], max_snr['mass1'], max_snr['mass2'], max_snr['snr'], max_snr['iter'], colors[max_snr['iter']]),
        'Final': (combined_mass[-1], snr_values[-1], mass1_values[-1], mass2_values[-1], snr_values[-1], iter_values[-1], colors[-1])
    }
    for marker_name, (mass, snr, mass1, mass2, snr_value, iteration, color) in markers.items():
        ax.scatter(mass, snr, marker='x', color=color, s=400, linewidth=3, label=f"{marker_name}: mass1: {mass1:.2f}, mass2: {mass2:.2f}, SNR: {snr_value:.2f}, iter: {iteration}")
    ax.legend(fontsize=21)


def plot_snr_vs_mass(event_name, strain, total_time, results, max_snr):
    mass1_values, mass2_values, snr_values, iter_values, combined_mass = sort_results(results)
    fig, ax = plt.subplots(figsize=(18, 12))
    n = len(mass1_values)
    color_name = cm.viridis
    colors = color_name(jnp.linspace(-1, 1, n))
    norm = plt.Normalize(-1, n-1)
    mappable = ScalarMappable(norm=norm, cmap=color_name)
    last_key = list(results.keys())[-1]
    
    plot_markers(ax, combined_mass, snr_values, mass1_values, mass2_values, iter_values, colors, max_snr)

    for i in range(n - 1):
        plt.plot(combined_mass[i:i+2], snr_values[i:i+2], "o-", color=colors[i], alpha=0.15, markersize=5)

    format_plot(ax, 'Mass (solar mass)', 'SNR', f'SNR vs Mass {event_name}({strain}) (Time taken: {total_time:.2f} seconds), iterations: {max(iter_values)}')
    create_colorbar(ax, mappable, 'SNR Index')
    fig.tight_layout()
    save_plot(f'SNR_vs_Combined_Mass_for_{event_name}_{strain}_T_{TEMPERATURE:.2f}_AR_{ANNEALING_RATE:.3f}_MI_{MAX_ITERS}_{LRL}_{LRU}_SEED{SEED}.png')

def plot_snr_vs_iteration(event_name, strain, total_time, results):
    mass1_values, mass2_values, snr_values, iter_values, combined_mass = sort_results(results)
    fig, ax = plt.subplots(figsize=(18, 12))
    norm = plt.Normalize(vmin=min(combined_mass), vmax=max(combined_mass))
    ax.scatter(iter_values, snr_values, c=combined_mass, cmap=cm.viridis, norm=norm)
    format_plot(ax, 'Iteration', 'SNR', f'SNR vs Iteration {event_name}({strain}) (Time taken: {total_time:.2f} seconds), iterations: {max(iter_values)}')
    mappable = ScalarMappable(norm=norm, cmap=cm.viridis)
    create_colorbar(ax, mappable, 'Combined Mass (Solar Masses)')
    fig.tight_layout(pad=5.0)
    save_plot(f'SNR_vs_Iterations_for_{event_name}_{strain}_T_{TEMPERATURE:.2f}_AR_{ANNEALING_RATE:.3f}_MI_{MAX_ITERS}_{LRL}_{LRU}_SEED{SEED}.png')

def plot_snr_timeseries(event_name, strain, max_snr, freqs, params, fdata_jax, psd_jax, delta_f):
    fig, ax = plt.subplots(figsize=(18, 12))
    mass1 = max_snr['mass1']
    mass2 = max_snr['mass2']
    snrp = max_snr['snr']
    template = gen_waveform(mass1, mass2, freqs, params)
    snr = pre_matched_filter(template, fdata_jax, psd_jax, delta_f)
    plt.plot(snr)

    format_plot(ax, 'SNR Index', 'SNR', f'Optimal Template {event_name}({strain}) SNR:{snrp:.2f}', fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=18)
    fig.tight_layout()
    save_plot(f'Optimal_Template_for_{event_name}_{strain}_T_{TEMPERATURE:.2f}_AR_{ANNEALING_RATE:.3f}_MI_{MAX_ITERS}_{LRL}_{LRU}_SEED{SEED}.png')

def sort_results(results):
    mass1_values = []
    mass2_values = []
    snr_values = []
    combined_mass = []
    iter_values = []

    for result in results.values():
        mass1_values.append(jnp.array(result['mass1s'], dtype=jnp.float32).item())
        mass2_values.append(jnp.array(result['mass2s'], dtype=jnp.float32).item())
        combined_mass.append(jnp.array(result['mass1s'] + result['mass2s'], dtype=jnp.float32))
        snr_values.append(jnp.array(result['snr'], dtype=jnp.float32).item())
        iter_values.append(jnp.array(result['iters'], dtype=jnp.float32).item())

    return mass1_values, mass2_values, snr_values, iter_values, combined_mass

def pycbc_plots(EVENT_NAME, STRAIN, conditioned, total_time, max_snr):

    # From the PyCBC tutorial 3: https://colab.research.google.com/github/gwastro/pycbc-tutorials/blob/master/tutorial/3_WaveformMatchedFilter.ipynb
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
    save_plot(filename)

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
    save_plot(filename)
