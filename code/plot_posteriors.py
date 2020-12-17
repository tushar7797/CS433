# This script contains functions to generate all the figures of the ./figures/ directory:

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import seaborn as sns
import scpyro


"""This function returns samples for a specific model parameter 
(i.e. freq) of a given gene (MCMC samples used)."""
def get_dist_mcmc(name, gene_ix, samples_mcmc):

  dist_name = 'transcriptome/'+name
  samples_mcmc_ = samples_mcmc[dist_name][:, gene_ix]
  
  return samples_mcmc_


"""This function returns samples for a specific model parameter 
(i.e. freq) of a given gene (MCMC and VI samples used)."""
def get_dist(name, gene_ix, samples_mcmc, samples_vi):

  dist_name = 'transcriptome/'+name
  samples_mcmc_ = samples_mcmc[dist_name][:, gene_ix]
  samples_vi_ = samples_vi[dist_name][:, gene_ix]
  
  return samples_mcmc_, samples_vi_


"""This function creates a figure that visualizes the differences, for all genes, 
between MCMCM and VI for a given parameter distribution (i.e. single parameter figures 
of ./figures/set1/ directory). If save_path='' the figure is not saved."""
def print_MCMC_VI_single_variable(dist_type, model_name, samples_mcmc, samples_vi, dataset, save_path='', nrows=6, ncols=5):

  fig, ax = plt.subplots(nrows, ncols, figsize=(16, 14))
  fig.suptitle(model_name+': MCMC vs. VI '+ dist_type+ ' distribution for different types of genes', fontsize=14)
  fig.tight_layout(rect=[0, 0.03, 1, 0.95])

  # for each gene
  for index, gene_ix in enumerate(dataset.var['gene_ix']):

    axis_i = int(np.floor(index / (nrows-1)))
    axis_j = index % ncols
    # print(index, axis_i, axis_j)
    
    # get MCMC and VI samples for the given parameter
    samples_mcmc_, samples_vi_ = get_dist(dist_type, gene_ix, samples_mcmc, samples_vi)
    
    # make the plot between MCMC and VI
    sns.kdeplot(samples_mcmc_, label = "MCMC", ax=ax[axis_i, axis_j]).set(ylabel=None)
    sns.kdeplot(samples_vi_, label = "VI", ax=ax[axis_i, axis_j]).set(ylabel=None)
    
    ax[axis_i, axis_j].set_title(dataset.var.loc[dataset.var['gene_ix'] == gene_ix].index[0], fontsize=10)
    if axis_i == (nrows-1):
      ax[axis_i, axis_j].set_xlabel('Log Perturbation', fontsize=10)
    if axis_j == 0:
      ax[axis_i, axis_j].set_ylabel('Density', fontsize=10)

  plt.legend(loc='upper right', labelspacing=0.1, borderpad=0.1)
  plt.show()

  if save_path != '':
    fig.savefig(f"{save_path}/{model_name}/{model_name}_{dist_type.replace('/', '-')}.jpg", format= 'jpg', dpi=200)
    print(f"Figure saved in dir: {save_path}/{model_name}/{model_name}_{dist_type.replace('/', '-')}.jpg")
    
    
"""This function creates a figure that visualizes the differences, for all genes, 
between MCMCM and VI given two parameter distributions (i.e. two parameter figures 
of ./figures/set1/ directory). If save_path='' the figure is not saved."""
def print_MCMC_VI_two_variables(dist_type1, dist_type2, model_name, samples_mcmc, samples_vi, dataset, save_path='', nrows=6, ncols=5):

  fig, ax = plt.subplots(nrows, ncols, figsize=(16, 14))
  fig.suptitle(model_name+': MCMC vs. VI '+ dist_type1+ ' and '+ dist_type2 +' distributions for different types of genes', fontsize=14)
  fig.tight_layout(rect=[0, 0.03, 1, 0.95])
  
  # for each gene
  for index, gene_ix in enumerate(dataset.var['gene_ix']):

    axis_i = int(np.floor(index / (nrows-1)))
    axis_j = index % ncols
    
    # get MCMC and VI samples for both parameters
    samples_mcmc_1, samples_vi_1 = get_dist(dist_type1, gene_ix, samples_mcmc, samples_vi)
    samples_mcmc_2, samples_vi_2 = get_dist(dist_type2, gene_ix, samples_mcmc, samples_vi)

    # make the plot between MCMC and VI
    sns.kdeplot(samples_mcmc_1, samples_mcmc_2, label = "MCMC", ax=ax[axis_i, axis_j]).set(ylabel=None)
    sns.kdeplot(samples_vi_1, samples_vi_2, label = "VI", ax=ax[axis_i, axis_j]).set(ylabel=None)
    
    ax[axis_i, axis_j].set_title(dataset.var.loc[dataset.var['gene_ix'] == gene_ix].index[0], fontsize=10)
    if axis_i == (nrows-1):
      ax[axis_i, axis_j].set_xlabel(dist_type1, fontsize=10)
    if axis_j == 0:
      ax[axis_i, axis_j].set_ylabel(dist_type2, fontsize=10)

  plt.legend(loc='upper right', labelspacing=0.1, borderpad=0.1)
  plt.show()
  
  if save_path != '':
    fig.savefig(f"{save_path}/{model_name}/{model_name}_{dist_type1.replace('/', '-')}_{dist_type2.replace('/', '-')}.jpg", format= 'jpg', dpi=200)
    print(f"Figure saved in dir: {save_path}/{model_name}/{model_name}_{dist_type1.replace('/', '-')}_{dist_type2.replace('/', '-')}.jpg")
    
    
"""This function creates a figure that visualizes the differences of a specific model parameter between
all different models, for all genes (i.e. all figures of ./figures/set2/ directory). If save_path='' 
the figure is not saved."""
def print_compare_models_MCMC(dist_type, samples_mcmc_list, dataset, save_path='', nrows=6, ncols=5):

  fig, ax = plt.subplots(nrows, ncols, figsize=(16, 14))
  fig.suptitle('MCMC '+ dist_type+ ' distribution comparison for different models', fontsize=14)
  fig.tight_layout(rect=[0, 0.03, 1, 0.95])

  # for all genes
  for index, gene_ix in enumerate(dataset.var['gene_ix']):

    axis_i = int(np.floor(index / (nrows-1)))
    axis_j = index % ncols
    # print(index, axis_i, axis_j)
    
    # for all models make the plot of a single parameter
    for samples_mcmc in samples_mcmc_list:
      samples_mcmc_ = get_dist_mcmc(dist_type, gene_ix, samples_mcmc[0])
      sns.kdeplot(samples_mcmc_, label = samples_mcmc[1], ax=ax[axis_i, axis_j]).set(ylabel=None)
    
    ax[axis_i, axis_j].set_title(dataset.var.loc[dataset.var['gene_ix'] == gene_ix].index[0], fontsize=10)
    if axis_i == (nrows-1):
      ax[axis_i, axis_j].set_xlabel('Log Perturbation', fontsize=10)
    if axis_j == 0:
      ax[axis_i, axis_j].set_ylabel('Density', fontsize=10)

  plt.legend(loc='upper right', labelspacing=0.1, borderpad=0.1)
  plt.show()

  if save_path != '':
    fig.savefig(f"{save_path}/mcmc_all_models_{dist_type.replace('/', '-')}.jpg", format= 'jpg', dpi=200)
    print(f"Figure saved in dir: {save_path}/mcmc_all_models_{dist_type.replace('/', '-')}.jpg")
    
    
"""This function creates a figure that visualizes the differences between the true rho and the rho 
estimated by MCMC and VI for a given model and for all genes (i.e. rho estimation figures of 
./figures/set1/ directory). If save_path=''the figure is not saved."""
def rho_estimation(model_name, model, samples_mcmc, samples_vi, dataset, rng_key_, infer_args, save_path):
  
  model.eval = True

  # to check out the model, we will see how gene expression depends on the perturbation
  # we do not care about the library so we just fix it to 100
  design = {
      "p":jnp.linspace(0, 1, 50),
      "library":jnp.array([100] * 50)
  }

  # predict samples of MCMC and VI for a given converged model
  predictive_mcmc = numpyro.infer.Predictive(model.forward, samples_mcmc, return_sites = ["rho"])
  predictions_mcmc = predictive_mcmc(rng_key_, **design)

  predictive_vi = numpyro.infer.Predictive(model.forward, samples_vi, return_sites = ["rho"])
  predictions_vi = predictive_vi(rng_key_, **design)

  # which genes do we want to plot
  gene_ixs = dataset.var['gene_ix'].tolist()

  # plot the models for the genes
  x = design["p"]
  Ys = {"vi":predictions_vi["rho"], "mcmc":predictions_mcmc["rho"]}
  
  #"#FF4136" = red, "#0074D9"= blue
  model_colors = {"vi":"#FF4136", "mcmc":"#0074D9"}
  model_labels = {"vi":"VI", "mcmc":"MCMC"}

  # empirical values
  x_empirical = infer_args["p"]
  Y_empirical = dataset.rho.values

  # gold standard values (we have these because this is toy data)
  x_gs = infer_args["p"]
  Y_gs = dataset["rho"].values

  rho = dataset

  n_col=5
  fig, axes = scpyro.plotting.axes_wrap(len(gene_ixs), n_col=n_col)
  fig.suptitle(model_name+': MCMC vs. VI rho estimation for different types of genes', fontsize=14)

  legend_artists = {}
  for gene_ix, ax in zip(gene_ixs, axes):
      # plot models
      for model_id, Y in Ys.items():
          y = Y[:, :, gene_ix]
          
          color = model_colors[model_id]
          labels = model_labels[model_id]
          q_left = [0.99, 0.8, 0.65] # which upper quantiles to plot
          q = q_left + [0.5] + [round(1 - q, 3) for q in q_left[::-1]] # which quantiles to plot
          qy = np.quantile(y, q, 0)
          qmed = qy[len(q_left)]

          for i in range(len(q_left)):
              artist = ax.fill_between(x, qy[i], qy[-i-1], color = color, lw = 0, alpha = 0.2)
              pass
          if gene_ix == len(gene_ixs)-1:
            artist = ax.plot(x, qmed, color = color, lw = 3, label = labels)
          else:
            artist = ax.plot(x, qmed, color = color, lw = 3)
          legend_artists[model_id] = artist[0]
          
      # plot empirical
      y_empirical = Y_empirical[:, gene_ix]
      ax.scatter(x_empirical, y_empirical, s = 3, color = "#333333")
      
      # plot gold standard
      y_gs = Y_gs[:, gene_ix]
      if gene_ix == len(gene_ixs)-1:
        sns.lineplot(x = x_gs, y = y_gs, ax = ax, color = "green", lw = 4, label = "GS rho") # gold standard rho
      else:
        sns.lineplot(x = x_gs, y = y_gs, ax = ax, color = "green", lw = 4)

      ax.set_title(dataset.symbol(feature_ix = gene_ix), fontsize=10)

      if gene_ix >= len(gene_ixs)-n_col:
        ax.set_xlabel('Perturbation', fontsize=12)
      if gene_ix % n_col == 0:
        ax.set_ylabel('rho', fontsize=12)

  fig.legend(legend_artists, loc='upper right')
  
  if save_path != '':
    fig.savefig(f"{save_path}/{model_name}/{model_name}_rho_estimation.jpg", format= 'jpg', dpi=200)
    print(f"Figure saved in dir: {save_path}/{model_name}/{model_name}_rho_estimation.jpg")