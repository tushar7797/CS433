# This script contains functions to:
# 1. Generate-save-load-transform the dataset
# 2. Approximate and sample the posterior with MCMC-NUTS and SVI
# 3. Compute Pearson correlations between model parameters

import jax
import jax.numpy as jnp

import latenta
import latento

import numpy as np
import numpyro
import numpyro.infer.autoguide

import pickle

import seaborn as sns
from scipy import stats

import tqdm.auto as tqdm


"""Generates a synthetic dataset that contains information about the gene expression of each cell 
given a configuration JSON parameter. The configuration describes the number of cells and the 
characteristics of the gene expression. """
def create_dataset(config):
    # creates the toy
    toy = latenta.toy.wrap.cases.get_case(config)
    
    # sample the count data
    latenta.toy.sample.sample(toy)
    
    # create a dataset object out of this
    dataset = latenta.toy.wrap.wrap_model(toy)
    
    # create a "latento" dataset (=numpyro) instead of a "latenta" (=pyro) one
    dataset = latento.dataset.PerturbedDataset.from_latenta(dataset)
    
    return dataset


""" This function is used to save data structures as pickle objects. 
We use this to save generated datasets and trained models. """
def save_pickle(object_, object_dir):
    with open(object_dir, 'wb') as output:
        pickle.dump(object_, output, pickle.HIGHEST_PROTOCOL)


""" This function is used to load pickle objects to the appropriate data structures. 
We use this to load generated datasets and trained models. """
def load_pickle(object_dir):
    with open(object_dir, 'rb') as input:
        object_pkl = pickle.load(input)
    return object_pkl


"""This function packs together in a JSON format the necessary variables for the different
types of models. Cell perturbation values are log transformed and scaled to max [0 1]."""
def transform_dataset(dataset):
  # we don't use the raw perturbation values, but use log (x + 1) transformed ones
  p = np.log(dataset.perturb.X["x"].values + 1)

  # we also scale the perturbation values between 0 and 1
  p = p / p.max()

  # these are the arguments that the inference functions need
  infer_args = {"p":p, "Y":dataset.get_Y(), "library":dataset.get_library()}

  return infer_args
  
  
"""Function used to estimate the exact posterior with Marchov Chain Monte Calro and NUTS sampler."""
def train_mcmc(model, infer_args, rng_key, num_warmup=1000, num_samples=2000):
  model.eval = False

  # Run NUTS
  kernel = numpyro.infer.NUTS(model.forward)
  mcmc = numpyro.infer.MCMC(kernel, num_warmup, num_samples)
  mcmc.run(rng_key, **infer_args)

  return mcmc


"""Function used to generate samples from the posterior approximated by MCMC-NUTS."""
def sample_mcmc(model, mcmc):
  # get the samples from the posterior
  model.eval = True
  samples_mcmc = mcmc.get_samples()

  return samples_mcmc
  
  
"""Function used to approximate and derive samples from a posterior distribution by using MCMC-NUTS sampler."""
def train_sample_mcmc(model, infer_args, rng_key, num_warmup=1000, num_samples=2000):
 
  # aproxiamte posterior
  mcmc = train_mcmc(model, infer_args, rng_key, num_warmup, num_samples)

  # some statistics of the posterior, the r_hat is a statistic of "convergence" for a particular random variable, this should be close to 1
  mcmc.print_summary()

  # get the samples from the posterior
  samples_mcmc = sample_mcmc(model, mcmc)
  
  return samples_mcmc
  
  
####################################### Notes for Black Box Variational Inference  ####################################### 
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
####################################### Notes for Black Box Variational Inference  ####################################### 


"""Posterior estimation with stochastic variational inference. We initialize the Diagonal Normal 
distribution with median values, we use ADAM for optimization and ELBO as a loss function."""
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


"""Function used to generate samples from the posterior approximated by VI. 
Number of samples should be equal to MCMC."""
def sample_vi(model, guide, svi, current_state, num_samples=2000):

  model.eval = True

  parameters = svi.get_params(current_state)

  samples_vi_raw = []

  # generate the same number of posterior samples as MCMC
  for i in range(num_samples):
      rng_key = jax.random.PRNGKey(i)
      samples_vi_raw.append(guide.sample_posterior(rng_key, parameters))
      
  samples_vi = {}
  for site_id in samples_vi_raw[0].keys():
      samples_vi[site_id] = jnp.stack([samples_vi_raw[i][site_id] for i in range(len(samples_vi_raw))])

  return samples_vi
  
  
"""Function used to approximate and derive samples from a posterior distribution 
by using stochastic variational inference."""
def train_sample_vi(model, infer_args, rng_key, step_size=0.01, n_iterations = 1000, num_samples=2000):
  # aproxiamte posterior
  guide, svi, current_state, losses = train_vi(model, infer_args, rng_key, step_size, n_iterations)
 
  # plot ELBO losses
  sns.lineplot(x = range(len(losses)), y = losses)
  
  # get the samples from the posterior
  samples_vi = sample_vi(model, guide, svi, current_state, num_samples)
  
  return samples_vi
  
  
""" Used to compute pearson correlations between parameter distributions for a given model and gene."""
def mcmc_pearson_correlation(mcmc_models_dists, dataset, cor_threshold=0.5, p_threshold=0.05, save_path=''):

  str_log_list=[]
  count = 0

  # for every model [nothing, linear, switch]
  for model in mcmc_models_dists:
      
    # for all possible pairs of parameters [dispersion, freq, beta, swtich]
    for index_1, dist_1 in enumerate(model[1]):
      for index_2, dist_2 in enumerate(model[1]):
        if index_2>index_1:
          # for all genes
          for index_gene, gene_ix in enumerate(dataset.var['gene_ix']):
            count+=1
            samples_mcmc_1 = model[0]['transcriptome/'+dist_1][:, gene_ix]
            samples_mcmc_2 = model[0]['transcriptome/'+dist_2][:, gene_ix]
            
            # compute Pearson correlation between samples of the parameters
            pearson_cor = stats.pearsonr(samples_mcmc_1, samples_mcmc_2)

            # accept correlation if thresholds are satisfied
            if (np.abs(pearson_cor[0])>=cor_threshold) and (pearson_cor[1]<=p_threshold):
              gene_ix_name = dataset.var.loc[dataset.var['gene_ix'] == gene_ix].index[0]
              str_log = f'{model[2]}, {dist_1} vs. {dist_2}, {gene_ix_name}: cor={pearson_cor[0]}, p={pearson_cor[1]}'
              str_log_list.append(str_log)
              
  print(f'Calculating Pearson correlation by comparing a set of distributions for a given model and gene:')
  print(f'cor_threshold={cor_threshold} and p_threshold={p_threshold}')
  print(f'Pearson correlations found: {len(str_log_list)} out of {count} total combinations')

  # save correlations to the specified folder
  if save_path != '':
    with open(save_path, 'w') as f:
        print(f"Log file saved in dir: {save_path}")
        f.write(f'Calculating Pearson correlation by comparing a set of distributions for a given model and gene:\n')
        f.write(f'cor_threshold={cor_threshold} and p_threshold={p_threshold}\n')
        f.write(f'Pearson correlations found: {len(str_log_list)} out of {count} total combinations\n')
        for cor in str_log_list:
            f.write("%s\n" % cor)
        
  return str_log_list, count