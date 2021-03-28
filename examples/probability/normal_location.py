# Purpose: MCMC simulation under a normal-normal location model where
# the posterior is known analytically. 

from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


def collapse_chains(mcmc_chains: tf.Tensor, thinning_freq=1) -> np.ndarray:
    """Collapse chains and optionally thin."""
    chain_len = mcmc_chains.shape[0]
    select = np.arange(start=0, stop=chain_len, step=thinning_freq)
    thinned_chains = mcmc_chains.numpy()[select,:]
    return np.squeeze(np.concatenate(thinned_chains))


def plot_chains(mcmc_chains: tf.Tensor, ref_val: float) -> None:
  """Plot MCMC chains for a single parameter."""
  chain_len = mcmc_chains.shape[0]
  n_chain = mcmc_chains.shape[1]
  for i in range(n_chain):
    plt.plot(mcmc_chains[:, i], alpha=0.3)
  plt.hlines(ref_val, 0, chain_len, linestyle='dotted')
  plt.xlabel('Step')
  plt.ylabel('State')
  plt.show()
  return None


# ---------------------------------------------------------------------------
# Unknown mean, known variance. 
# ---------------------------------------------------------------------------

# The generative model is:
#   mu ~ N(loc0, scale0)
#   x_i ~ iid N(mu, 1).

def gen_data(n: int, mu=1.0) -> tf.Tensor:
  """Generate data.

  Args:
    n: Sample size.
    mu: True location.

  Returns:
    (n,) tensor of draws from the N(mu, 1) distribution.
  """
  rv_x = tfp.distributions.Normal(loc=mu, scale=1)
  return rv_x.sample((n,))


def joint_log_prob(mu: float,
                   x_obs: tf.Tensor,
                   loc0=0.0,
                   scale0=1.0e3) -> tf.Tensor:
  """Joint lob probability.

  Args:
    mu: Generative mean.
    x_obs: Observed data.
    loc0: Prior location of mean.
    scale0: Prior scale of mean.

  Returns:
    Joint log probability of (mu, x_obs) under the prior.
  """

  # Prior.
  prior = tfp.distributions.Normal(loc=loc0, scale=scale0)

  # Likelihood.
  lik = tfp.distributions.Normal(loc=mu, scale=1)

  # Joint log prob.
  return prior.log_prob(mu) + tf.reduce_sum(lik.log_prob(x_obs))


def analytical_posterior(x: tf.Tensor,
                         loc0=0.0,
                         scale0=1.0e3) -> Tuple[np.ndarray, np.ndarray]:
  """Calculate posterior mean and variance analytically.

    Args:
      x: Observed data.
      loc0: Prior mean.
      scale0: Prior scale.

    Returns:
      (Posterior mean, posterior variance)
    """

  x_np = x.numpy()
  n = x_np.shape[0]

  # Posterior mean.
  v0 = scale0**2
  mu_p = 1.0 / (1.0 / v0 + n) * (loc0 / v0 + x_np.sum())

  # Posterior variance.
  v_p = 1.0 / (1.0 / v0 + n)

  return (mu_p, v_p)


def normal_normal_experiment(mu: float, n: int, loc0: float, scale0: float,
                             init_state: tf.Tensor) -> Dict[str, Any]:
  """Normal-Normal MCMC experiment.

    Args:
      mu: True mean.
      n: Sample size.
      loc0: Prior location.
      scale0: Prior scale.
      init_state: Chain initial states. Including at least 3 to allow for
        calculation of the potential-scale reduction.

    Returns:
        Dictionary containing the generative mu, the observed data, and the
        posterior sample.
    """

  # Draw sample.
  x_obs = gen_data(n, mu=mu)

  def unnormalized_posterior(mu: float):
    return joint_log_prob(mu=mu, x_obs=x_obs, loc0=loc0, scale0=scale0)

  # Transition kernel.
  hmc_kernel = tfp.mcmc.HamiltonianMonteCarlo(
      target_log_prob_fn=unnormalized_posterior,
      step_size=0.25,
      num_leapfrog_steps=2)

  # Run MCMC chain.
  @tf.function
  def run_chain(initial_state, chain_length=1000, burnin=200):
    adaptive_kernel = tfp.mcmc.SimpleStepSizeAdaptation(
        hmc_kernel,
        num_adaptation_steps=int(0.8 * burnin),
        target_accept_prob=0.75)

    return tfp.mcmc.sample_chain(
        num_results=chain_length,
        num_burnin_steps=burnin,
        current_state=initial_state,
        kernel=adaptive_kernel,
        trace_fn=lambda current_state, kernel_results: kernel_results)

  # Generate chains, report acceptance and PSR.
  chains, kernel_results = run_chain(initial_state=init_state)
  psr = tfp.mcmc.potential_scale_reduction(chains)
  print('Acceptance rate:',
        kernel_results.inner_results.is_accepted.numpy().mean())
  print('Potential scale reduction:', psr.numpy())  # Should be near 1.

  # Plot MCMC chains.
  plot_chains(chains, ref_val=mu)

  # Analytical posterior mean and variance.
  (mu_p, v_p) = analytical_posterior(x_obs)
  print('\n')
  print(f'Analytical posterior mean: {mu_p:.3f}')
  print(f'Analytical posterior variance: {v_p:.3f}')
  print('\n')

  # Posterior sample.
  posterior = collapse_chains(chains)
  mu_hat_p = posterior.mean()
  v_hat_p = posterior.var()
  print(f'Estimated posterior mean: {mu_hat_p:.3f}')
  print(f'Estimated posterior variance: {v_hat_p:.3f}')
  print('\n')

  # Results.
  return {'mu': mu, 'data': x_obs, 'posterior': posterior}

# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------

# Initialize 3 chains, at [-1], [0], and [1] respectively.
init_state = tf.constant(value=[[-1], [0], [1]], dtype=tf.float32)

# Small sample size, uninformative prior.
_ = normal_normal_experiment(
    mu=1.0, n=10, loc0=0, scale0=1e3, init_state=init_state)

# Small sample size, informative prior.
_ = normal_normal_experiment(
    mu=1.0, n=10, loc0=0, scale0=2, init_state=init_state)

# Large sample size, uninformative prior.
_ = normal_normal_experiment(
    mu=1.0, n=100, loc0=0, scale0=1e3, init_state=init_state)

# Large sample size, informative prior.
_ = normal_normal_experiment(
    mu=1.0, n=100, loc0=0, scale0=2, init_state=init_state)