import jax
from jax import jit, grad, random
import jax.numpy as jnp
from GravAD import make_function
from constants import * 

@jit
def sgd(state, _):
    """
    Context:

    Args:

    Returns:
    """
    # Unpack the state variables
    current_i, mass1, mass2, rng_key, data, psd, freqs, params, delta_f, signal_type = state

    # Function to get SNR
    snr_func = make_function(data, psd, freqs, params, delta_f, signal_type)

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

    # Update mass1
    new_mass1 = mass1 + (learning_rate1 * gradient1)

    # Avoid mass1 falling below 11 solar masses and above 120 solar masses
    new_mass1 = jax.lax.cond(new_mass1 < 11., lambda _: 11. + mass1, lambda _: new_mass1, None)
    new_mass1 = jax.lax.cond(new_mass1 > 120., lambda _: 120. - mass1, lambda _: new_mass1, None)
    new_mass1 = jax.lax.cond(jnp.isnan(new_mass1), lambda _: jnp.nan, lambda _: new_mass1, None)
    
    # Update mass2
    new_mass2 = mass2 + (learning_rate2 * gradient2)

    # Avoid mass2 falling below 11 solar masses and above 120 solar masses
    new_mass2 = jax.lax.cond(new_mass2 < 11., lambda _: 11. + mass2, lambda _: new_mass2, None)
    new_mass2 = jax.lax.cond(new_mass2 > 120., lambda _: 120. - mass2, lambda _: new_mass2, None)
    new_mass2 = jax.lax.cond(jnp.isnan(new_mass2), lambda _: jnp.nan, lambda _: new_mass2, None)

    # Create a new state
    new_state = (current_i+1, new_mass1, new_mass2, rng_key, data, psd, freqs, params, delta_f, signal_type)

    return new_state, (snr, mass1, mass2)

@jit
def sgd_sa(state, _):
    """
    Simulates one step of simulated annealing. Calculates the gradient of the snr with respect to the masses,
    updates the masses using the learning rate and the gradients, then applies a random perturbation to the masses.

    Args:
    state (tuple): Current state of the system, includes the mass values, RNG key, temperature, annealing rate, and other parameters.

    Returns:
    tuple: The updated state of the system and the snr and mass values.
    """
    # Unpack the state variables
    current_i, mass1, mass2, rng_key, temperature, annealing_rate, data, psd, freqs, params, delta_f, signal_type = state

    # Function to get SNR
    snr_func = make_function(data, psd, freqs, params, delta_f, signal_type)

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
    new_state = (current_i+1, new_mass1, new_mass2, rng_key, temperature, annealing_rate, data, psd, freqs, params, delta_f, signal_type)

    return new_state, (snr, mass1, mass2)

@jit
def sgd_adam(state, _):
    """
    Context:

    Args:

    Returns:
    """
    # Unpack the state variables
    current_i, mass1, mass2, rng_key, momentum1, momentum2, alpha, data, psd, freqs, params, delta_f, signal_type = state

    # Function to get SNR
    snr_func = make_function(data, psd, freqs, params, delta_f, signal_type)

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
    
    # Momentum
    momentum1 = alpha * momentum1 + (1 - alpha) * gradient1
    momentum2 = alpha * momentum1 + (1 - alpha) * gradient2

    # Update mass1
    new_mass1 = mass1 + (learning_rate1 * momentum1)

    # Avoid mass1 falling below 11 solar masses and above 120 solar masses
    new_mass1 = jax.lax.cond(new_mass1 < jnp.float32(11.), lambda _: jnp.float32(11.) + jnp.float32(jnp.abs(momentum1)), lambda _: jnp.float32(new_mass1), None)
    new_mass1 = jax.lax.cond(new_mass1 > jnp.float32(120.), lambda _: jnp.float32(120.) - jnp.float32(jnp.abs(momentum1)), lambda _: jnp.float32(new_mass1), None)
    new_mass1 = jax.lax.cond(jnp.isnan(new_mass1), lambda _: jnp.float32(jnp.nan), lambda x: x, jnp.float32(new_mass1))

    # Update mass2
    new_mass2 = mass2 + (learning_rate2 * momentum2)

    # Avoid mass2 falling below 11 solar masses and above 120 solar masses
    new_mass2 = jax.lax.cond(new_mass2 < jnp.float32(11.), lambda _: jnp.float32(11.) + jnp.float32(jnp.abs(momentum2)), lambda _: jnp.float32(new_mass2), None)
    new_mass2 = jax.lax.cond(new_mass2 > jnp.float32(120.), lambda _: jnp.float32(120.) - jnp.float32(jnp.abs(momentum2)), lambda _: jnp.float32(new_mass2), None)
    new_mass2 = jax.lax.cond(jnp.isnan(new_mass2), lambda _: jnp.float32(jnp.nan), lambda x: x, jnp.float32(new_mass2))   
    # Create a new state
    new_state = (current_i+1, new_mass1, new_mass2, rng_key, momentum1, momentum2, alpha, data, psd, freqs, params, delta_f, signal_type)

    return new_state, (snr, mass1, mass2)

@jit
def sgd_adam_sa(state, _):
    """
    Context:

    Args:

    Returns:
    """
    # Unpack the state variables
    current_i, mass1, mass2, rng_key, temperature, annealing_rate, momentum1, momentum2, alpha, data, psd, freqs, params, delta_f, signal_type = state

    # Function to get SNR
    snr_func = make_function(data, psd, freqs, params, delta_f, signal_type)

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

    # Momentum 
    momentum1 = alpha * momentum1 + (1 - alpha) * gradient1
    momentum2 = alpha * momentum2 + (1 - alpha) * gradient2

    # Update mass1
    new_mass1 = mass1 + (learning_rate1 * momentum1) + perturbation1

    # Avoid mass1 falling below 11 solar masses and above 120 solar masses
    new_mass1 = jax.lax.cond(new_mass1 < 11., lambda _: 11. + jnp.abs(perturbation1), lambda _: new_mass1, None)
    new_mass1 = jax.lax.cond(new_mass1 > 120., lambda _: 120. - jnp.abs(perturbation1), lambda _: new_mass1, None)
    new_mass1 = jax.lax.cond(jnp.isnan(new_mass1), lambda _: jnp.nan, lambda _: new_mass1, None)
    
    # Update mass2
    new_mass2 = mass2 + (learning_rate2 * momentum2) + perturbation2

    # Avoid mass2 falling below 11 solar masses and above 120 solar masses
    new_mass2 = jax.lax.cond(new_mass2 < 11., lambda _: 11. + jnp.abs(perturbation2), lambda _: new_mass2, None)
    new_mass2 = jax.lax.cond(new_mass2 > 120., lambda _: 120. - jnp.abs(perturbation2), lambda _: new_mass2, None)
    new_mass2 = jax.lax.cond(jnp.isnan(new_mass2), lambda _: jnp.nan, lambda _: new_mass2, None)
    
    # Update the temperature for the annealing
    temperature *= annealing_rate

    # Create a new state
    new_state = (current_i+1, new_mass1, new_mass2, rng_key, temperature, annealing_rate, momentum1, momentum2, alpha, data, psd, freqs, params, delta_f, signal_type)

    return new_state, (snr, mass1, mass2)

