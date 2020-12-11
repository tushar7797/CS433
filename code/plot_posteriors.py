import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def get_dist_mcmc(name, gene_ix, samples_mcmc):

  dist_name = 'transcriptome/'+name
  samples_mcmc_ = samples_mcmc[dist_name][:, gene_ix]
  
  return samples_mcmc_


def get_dist(name, gene_ix, samples_mcmc, samples_vi):

  dist_name = 'transcriptome/'+name
  samples_mcmc_ = samples_mcmc[dist_name][:, gene_ix]
  samples_vi_ = samples_vi[dist_name][:, gene_ix]
  
  return samples_mcmc_, samples_vi_


def print_MCMC_VI_single_variable(dist_type, model_name, samples_mcmc, samples_vi, dataset, save_path='', nrows=6, ncols=5):

  fig, ax = plt.subplots(nrows, ncols, figsize=(16, 14))
  fig.suptitle(model_name+': MCMC vs. VI '+ dist_type+ ' distribution for different types of genes', fontsize=14)
  fig.tight_layout(rect=[0, 0.03, 1, 0.95])

  for index, gene_ix in enumerate(dataset.var['gene_ix']):

    axis_i = int(np.floor(index / (nrows-1)))
    axis_j = index % ncols
    # print(index, axis_i, axis_j)
    samples_mcmc_, samples_vi_ = get_dist(dist_type, gene_ix, samples_mcmc, samples_vi)
    
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
    fig.savefig(f"{save_path}/{model_name}/{model_name}_{dist_type.replace('/', '-')}.jpg", format= 'jpg', dpi=600)
    print(f"Figure saved in dir: {save_path}/{model_name}/{model_name}_{dist_type.replace('/', '-')}.jpg")


def print_MCMC_VI_two_variables(dist_type1, dist_type2, model_name, samples_mcmc, samples_vi, dataset, save_path='', nrows=6, ncols=5):

  fig, ax = plt.subplots(nrows, ncols, figsize=(16, 14))
  fig.suptitle(model_name+': MCMC vs. VI '+ dist_type1+ ' and '+ dist_type2 +' distributions for different types of genes', fontsize=14)
  fig.tight_layout(rect=[0, 0.03, 1, 0.95])

  for index, gene_ix in enumerate(dataset.var['gene_ix']):

    axis_i = int(np.floor(index / (nrows-1)))
    axis_j = index % ncols

    samples_mcmc_1, samples_vi_1 = get_dist(dist_type1, gene_ix, samples_mcmc, samples_vi)
    samples_mcmc_2, samples_vi_2 = get_dist(dist_type2, gene_ix, samples_mcmc, samples_vi)

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
    fig.savefig(f"{save_path}/{model_name}/{model_name}_{dist_type1.replace('/', '-')}_{dist_type2.replace('/', '-')}.jpg", format= 'jpg', dpi=600)
    print(f"Figure saved in dir: {save_path}/{model_name}/{model_name}_{dist_type1.replace('/', '-')}_{dist_type2.replace('/', '-')}.jpg")
    
    
def print_compare_models_MCMC(dist_type, samples_mcmc_list, dataset, save_path='', nrows=6, ncols=5):

  fig, ax = plt.subplots(nrows, ncols, figsize=(16, 14))
  fig.suptitle('MCMC '+ dist_type+ ' distribution comparison for different models', fontsize=14)
  fig.tight_layout(rect=[0, 0.03, 1, 0.95])

  for index, gene_ix in enumerate(dataset.var['gene_ix']):

    axis_i = int(np.floor(index / (nrows-1)))
    axis_j = index % ncols
    # print(index, axis_i, axis_j)
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
    fig.savefig(f"{save_path}/mcmc_all_models_{dist_type.replace('/', '-')}.jpg", format= 'jpg', dpi=600)
    print(f"Figure saved in dir: {save_path}/mcmc_all_models_{dist_type.replace('/', '-')}.jpg")