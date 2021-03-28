# Purpose: MCMC simulation under a normal model where both the location and
# scale are unknown. 

from typing import Any, Dict, List

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


def plot_chains(mcmc_chains: tf.Tensor, ref_val: List[float]) -> None:
    """Plot MCMC chains for a single parameter."""
    mu_chain = mcmc_chains[0]
    sigma_chain = mcmc_chains[1]
    chain_len = mu_chain.shape[0]
    n_chain = mu_chain.shape[1]
    
    fig, axs = plt.subplots(2)
    
    for i in range(n_chain):
        axs[0].plot(mu_chain[:, i], alpha=0.3)
        axs[1].plot(sigma_chain[:, i], alpha=0.3)
    
    axs[0].set_title('Mu')
    axs[1].set_title('Sigma')
    
    for ax in axs.flat:
        ax.set(xlabel='Step', ylabel='State')
    
    if ref_val:
        for i in range(len(ref_val)):
            axs[i].axhline(y=ref_val[i], 
                           xmin=0, xmax=chain_len, 
                           linestyle="--", color='gray')
    
    fig.subplots_adjust(hspace=0.75)
    plt.show()
    
    return None


# ---------------------------------------------------------------------------
# Unknown mean, unknown variance. 
# ---------------------------------------------------------------------------

# The generative model is:
#   mu ~ N(loc0, scale0)
#   sigma ~ InvGamma(shape, rate)
#   x_i ~ iid N(mu, sigma).

def gen_data(n: int, mu=1.0, sigma=1.0) -> tf.Tensor:
  """Generate data.

  Args:
    n: Sample size.
    mu: True location.

  Returns:
    (n,) tensor of draws from the N(mu, 1) distribution.
  """
  rv_x = tfp.distributions.Normal(loc=mu, scale=sigma)
  return rv_x.sample((n,))


def joint_log_prob(mu: float,
                   sigma: float,
                   x_obs: tf.Tensor,
                   loc0=0.0,
                   scale0=1.0e3,
                   shape0=1.0,
                   rate0=1.0) -> tf.Tensor:
  """Joint lob probability.

  Args:
    mu: Generative mean.
    x_obs: Observed data.
    loc0: Prior location for mu.
    scale0: Prior scale for mu.
    shape: Prior shape for sigma.
    rate: Prior rate for sigma.

  Returns:
    Joint log probability of (mu, sigma, x_obs) under the prior.
  """

  # Prior.
  prior_mu = tfp.distributions.Normal(loc=loc0, scale=scale0)
  prior_sigma = tfp.distributions.InverseGamma(
      concentration=shape0, scale=rate0)

  # Likelihood.
  lik = tfp.distributions.Normal(loc=mu, scale=sigma)

  # Joint log prob.
  return prior_mu.log_prob(mu) + prior_sigma.log_prob(sigma) + tf.reduce_sum(
      lik.log_prob(x_obs), axis=1, keepdims=True)



def mcmc_experiment(mu: float, sigma: float, n: int,
                    loc0: float, scale0: float,
                    shape0: float, rate0: float,
                    init_state: tf.Tensor,
                    chain_len: int) -> Dict[str, Any]:
  """Normal-Normal MCMC experiment.

    Args:
      mu: True mean.
      sigma: True scale.
      n: Sample size.
      loc0: Prior location for mu.
      scale0: Prior scale for mu.
      shape0: Prior shape for sigma.
      rate0: Prior rate for sigma.
      init_state: Chain initial states. Including at least 3 to allow for
        calculation of the potential-scale reduction.

    Returns:
        Dictionary containing the generative mu and sigma, 
        the observed data, and the posterior sample.
    """

  # Draw sample.
  x_obs = gen_data(n, mu=mu, sigma=sigma)

  def unnormalized_posterior(mu, sigma):
      
      return joint_log_prob(mu=mu, sigma=sigma, 
                            x_obs=x_obs, 
                            loc0=loc0, scale0=scale0,
                            shape0=shape0, rate0=rate0)

  # Transition kernel.
  hmc_kernel = tfp.mcmc.HamiltonianMonteCarlo(
      target_log_prob_fn=unnormalized_posterior,
      step_size=1.0,
      num_leapfrog_steps=2)

  # Run MCMC chain.
  @tf.function
  def run_chain(initial_state, chain_length=chain_len, burnin=1000):
    adaptive_kernel = tfp.mcmc.SimpleStepSizeAdaptation(
        hmc_kernel,
        num_adaptation_steps=int(0.8 * burnin),
        target_accept_prob=0.75)

    return tfp.mcmc.sample_chain(
        num_results=chain_length,
        num_burnin_steps=burnin,
        current_state=initial_state,
        kernel=adaptive_kernel,
        trace_fn=lambda _, kernel_results: kernel_results)

  # Generate chains.
  chains, kernel_results = run_chain(initial_state=init_state)
  
  # Calculate acceptance rate.
  acceptance = kernel_results.inner_results.is_accepted.numpy().mean()

  # Potential scale reduction.  
  psr = tfp.mcmc.potential_scale_reduction(chains)
  psr_mu = float(psr[0].numpy())
  psr_sigma = float(psr[1].numpy())
  
  # Reporting.
  print(f'Acceptance rate: {acceptance:.3f}',)
  print('Potential scale reduction:')
  print(f'mu: {psr_mu:.3f}')
  print(f'sigma: {psr_sigma:.3f}')

  # Plot MCMC chains.
  plot_chains(chains, ref_val=mu)

  # Posterior sample.
  mu_post = collapse_chains(chains[0])
  sigma_post = collapse_chains(chains[1])
  
  # Posterior moments.
  mean_mu = mu_post.mean()
  se_mu = mu_post.std()
  
  mean_sigma = sigma_post.mean()
  se_sigma = sigma_post.std()
  
  print(f'Estimated mean (SE) of mu: {mean_mu:.3f} ({se_mu:.3f})')
  print(f'Estimated mean (SE) of sigma: {mean_sigma:.3f} ({se_sigma:.3f})')

  # Results.
  return {'mu': mu, 'sigma': sigma, 'data': x_obs,
          'mu_post': mu_post, 'sigma_post': sigma_post}

# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------

# List of tensors, for mu and sigma respectively.
init_state = [
    tf.constant([[-1.0], [0.0], [1.0]]),
    tf.constant([[0.8], [1.0], [1.2]])    
]

# Smaller sample.
_ = mcmc_experiment(
    mu=0.0, sigma=1.0, n=100,
    loc0=0., scale0=1e-3,
    shape0=1, rate0=1,
    init_state=init_state,
    chain_len=5000)

_ = mcmc_experiment(
    mu=0.0, sigma=1.0, n=100,
    loc0=0., scale0=1,
    shape0=1, rate0=1,
    init_state=init_state,
    chain_len=5000)

# Larger sample.
_ = mcmc_experiment(
    mu=0.0, sigma=1.0, n=1000,
    loc0=0., scale0=1e-3,
    shape0=1, rate0=1,
    init_state=init_state,
    chain_len=5000)

_ = mcmc_experiment(
    mu=0.0, sigma=1.0, n=1000,
    loc0=0., scale0=1,
    shape0=1, rate0=1,
    init_state=init_state,
    chain_len=5000)