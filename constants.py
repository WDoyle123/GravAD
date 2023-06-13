# Constants
SAMPLING_RATE = 2048
LOW_FREQ_CUTOFF = 20.
HIGH_FREQ_CUTOFF = 1000.
MAX_ITERS = 500
TEMPERATURE = 1 # Initial Temperature
ADV_TEMPERATURE = 2 # Advanced Temperature
ANNEALING_RATE = 0.99
LRU = 1.5 # Learning Rate Upper
LRL = 5.5 # Learning Rate Lower
SEED = 1
STRAINS = ['H1', 'L1']
EVENTS = [
        "GW150914", "GW151012", "GW151226", "GW170104", "GW170608", "GW170729",
        "GW170809", "GW170814", "GW170817", "GW170818", "GW170823"]

# Define waveform parameters
chi1 = 0.
chi2 = 0.
tc = 1.0
phic = 1. 
dist_mpc = 440.
inclination = 0.
params = [chi1, chi2, dist_mpc, tc, phic, inclination]

