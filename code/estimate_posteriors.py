import numpy as np

import jax
import jax.numpy as jnp

import numpyro
import numpyro.infer.autoguide
import tqdm.auto as tqdm

import seaborn as sns


def transform_dataset(dataset):
  # we don't use the raw perturbation values, but use log (x + 1) transformed ones
  p = np.log(dataset.perturb.X["x"].values + 1)

  # we also scale the perturbation values between 0 and 1
  p = p / p.max()

  # these are the arguments that the inference functions need
  infer_args = {"p":p, "Y":dataset.get_Y(), "library":dataset.get_library()}

  return infer_args
  
  
def train_mcmc(model, infer_args, rng_key, num_warmup=1000, num_samples=2000):
  model.eval = False

  # Run NUTS
  kernel = numpyro.infer.NUTS(model.forward)
  mcmc = numpyro.infer.MCMC(kernel, num_warmup, num_samples)
  mcmc.run(rng_key, **infer_args)

  return mcmc


def sample_mcmc(model, mcmc):
  # get the samples from the posterior
  model.eval = True
  samples_mcmc = mcmc.get_samples()

  return samples_mcmc
  
  
def train_sample_mcmc(model, infer_args, rng_key, num_warmup=1000, num_samples=2000):
  mcmc = train_mcmc(model, infer_args, rng_key, num_warmup, num_samples)

  # some statistics of the posterior, the r_hat is a statistic of "convergence" for a particular random variable, this should be close to 1
  mcmc.print_summary()

  # get the samples from the posterior
  samples_mcmc = sample_mcmc(model, mcmc)
  
  return samples_mcmc
  
  
####################################### Notes for Vi ####################################### 
# we define our variational distribution or "guide"
# we will use the variational distribution to approximate the actual posterior as much as possible
# this is often a simple distribution from which it is easy to sample, in this case an autodiagonal normal
# note that this diagonalnormal distribution is also often called the "mean-field" approximation
# for each random variable, this distribution has two sets of parameters: a mean and a variance
# note that this distribution assumes no relationships between the posteriors of any variables! (i.e. covariance = 0)
# it may be possible that this assumption is not correct for some models, genes or variables
# if that is the case, you could try to use another variational distribution, e.g.:
# - to fully model dependencies between variables, you can use a AutoMultivariateNormal, that will model the full covariance matrix using a cholensky decomposition
# - to model some dependencies you can use a AutoLowRankMultivariateNormal, this will model a low-rank version of the full covariance matrix again using a cholensky decomposition
# in any case, these alternative guides are still approximations, i.e. we still only use a normal distribution!
# it may be that the actual posterior has heavy tails or multiple modes, things that we cannot model with just a normal distribution

# important to note: for variables that do not have a real support (e.g. unit interval, all positive numbers, ...) the samples are transformed after they are sampled from the normal 
# distribution (which as you know has real support) for example, the dispersion is only defined for positive numbers (we use a lognormal distribution), so after sampling x from the 
# normal distribution, the guide will take e^x as the value of the dispersion similarly, for the freq, this is only defined in [0, 1], so the guide will take the logit
# this is all done under the hood for you by the autoguide function, base on the support of the prior distribution
# guide = numpyro.infer.autoguide.AutoDiagonalNormal(model.forward, init_loc_fn=numpyro.infer.init_to_feasible)
####################################### Notes for Vi ####################################### 


def train_vi(model, infer_args, rng_key, step_size=0.01, n_iterations = 2000):

  # set eval off to do inference 
  model.eval = False

  # set distribution
  guide = numpyro.infer.autoguide.AutoDiagonalNormal(model.forward, init_loc_fn=numpyro.infer.init_to_median)

  # we use ADAM for optimization
  optim = numpyro.optim.Adam(step_size)

  # our loss function is the ELBO, which is an approximation (=lower bound) to the actual KL divergence between the guide and the "real" posterior
  # you can find several derivations of this on the internet:
  # - https://bjlkeng.github.io/posts/variational-bayes-and-the-mean-field-approximation/
  # - https://www.ritchievink.com/blog/2019/09/16/variational-inference-from-scratch/
  loss = numpyro.infer.Trace_ELBO()

  # we will do stochastic variational inference: stochastic because we sample from the prior and posterior (=guide) at each iteration
  svi = numpyro.infer.svi.SVI(model.forward, guide, optim, loss)

  # define the initial state
  current_state = svi.init(rng_key, **infer_args)

  # keep track of the losses
  losses = []
  progress = tqdm.tqdm(range(n_iterations))
  for i in progress:
      # take one step
      current_state, current_loss = svi.update(current_state, **infer_args)
      losses.append(current_loss.item())
      progress.set_description(str(round(current_loss.item(), 2)))

  return guide, svi, current_state, losses


def sample_vi(model, guide, svi, current_state, num_samples=2000):

# let's get samples from this posterior, in the same format as the samples from the mcmc posterior
  model.eval = True

  parameters = svi.get_params(current_state) # make sure that you understand what these parameters of the posterior mean (no pun intended)

  samples_vi_raw = []

  for i in range(num_samples):
      rng_key = jax.random.PRNGKey(i)
      samples_vi_raw.append(guide.sample_posterior(rng_key, parameters))
      
  samples_vi = {}
  for site_id in samples_vi_raw[0].keys():
      samples_vi[site_id] = jnp.stack([samples_vi_raw[i][site_id] for i in range(len(samples_vi_raw))])

  return samples_vi
  
  
def train_sample_vi(model, infer_args, rng_key, step_size=0.01, n_iterations = 1000, num_samples=2000):
#Train and get samples from VI
  guide, svi, current_state, losses = train_vi(model, infer_args, rng_key, step_size, n_iterations)

  sns.lineplot(x = range(len(losses)), y = losses)

  samples_vi = sample_vi(model, guide, svi, current_state, num_samples)
  
  return samples_vi