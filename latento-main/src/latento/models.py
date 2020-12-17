import numpyro
import jax.numpy as jnp
import jax
import numpy as np

from . import distributions

# define a *nothing* model, i.e. a model where the gene expression does not change
class ModelNothing():
    def __init__(self, dataset):
        # define prior distributions, most of these are derived empirically from the data
        self.dispersion_dist = dataset.get_empirical_dist("dispersion") # this is a log-normal distribution
        self.freq_dist = dataset.get_empirical_dist("freq") # this is a beta distribution
        
        self.n_genes = dataset.var.shape[0]
        
        # whether we are in evaluation mode or not
        # to speed up inference, we only store some variables (e.g. the rho) in evaluation mode
        self.eval = False
        
    # p = perturbation for each cell, Y = gene expression for each cell and gene, library = number of counts for each cell
    def forward(self, p, library, Y = None):
        # all gene-specific parameters
        with numpyro.plate("genes", self.n_genes):
            # the dispersion of the negative binomial distribution
            dispersion = numpyro.sample('transcriptome/dispersion', self.dispersion_dist)
            
            # the frequency = within a given cell, how many % of observed gene counts come from a particular gene?
            # this is the baseline frequency of a gene, i.e. without any perturbation applied to it
            freq = numpyro.sample('transcriptome/freq', self.freq_dist)
            
        n_cells = len(library)
        with numpyro.plate("cells", n_cells):
            # we expand the freq and dispersion across the cell dimension so that it matches the dimensions of the gene expression (=Y)
            freq = freq[None, :].repeat(n_cells, 0)
            dispersion = dispersion[None, :].repeat(n_cells, 0)
            
            # the gene does nothing, so the rho is equal to the freq
            rho = freq
            if self.eval: # in evaluation mode, we store the rho so that we can retrieve it later e.g. in de Predictive class (see later)
                numpyro.deterministic("rho", rho)
            
            # sample the observed gene expression (=transcriptome)
            # the most commonly used distribution for count data is the poisson distribution, which has only one parameter: it's mean or location parameter
            # however, gene expression data is overdispersed, i.e. the counts have a higher variance that is expected by a poisson distribution
            # we therefore use a negative binomial distribution, which allows to account for this overdispersion
            # negative binomial distribution is also called a gamma-poisson distribution
            # the normal parameterization of a negative binomial (e.g. the ones you find on wikipedia) don't have a "mean" (=loc) and "dispersion" (=scale) parameter however
            # however, we like to model the "average expression" in a cell (=rho) and the "variability in expression" (=dispersion)
            # we therefore use a different parameterization for the negative binomial that has this "mean" and "dispersion"
            # this different parameterization is quite straightforward, you can check latenta/distributions.py for how do it
            numpyro.sample("transcriptome", distributions.NegativeBinomial(rho * library[:, None], dispersion).to_event(1), obs = Y)


# define a *linear* model, where the gene expression depends on a perturbation in a linear way
class ModelLinear():
    def __init__(self, dataset):
        self.dispersion_dist = dataset.get_empirical_dist("dispersion")
        self.freq_dist = dataset.get_empirical_dist("freq")
        
        # beta = the slope of the linear function, different for each gene
        # we again use an empirical prior for this
        self.beta_dist = dataset.get_empirical_dist("deviation")
        
        self.n_genes = dataset.var.shape[0]
        
        self.eval = False
        
    def forward(self, p, library, Y = None):
        # this is the same as for the nothing model, but now we also sample the beta
        with numpyro.plate("genes", self.n_genes):
            dispersion = numpyro.sample('transcriptome/dispersion', self.dispersion_dist)
            freq = numpyro.sample('transcriptome/freq', self.freq_dist)
            
            beta = numpyro.sample('transcriptome/perturb/beta', self.beta_dist)
            
        n_cells = len(library)
        with numpyro.plate("cells", len(library)):
            freq = freq[None, :].repeat(n_cells, 0) # expand to cells by genes
            dispersion = dispersion[None, :].repeat(n_cells, 0) # expand to cells by genes
            
            # the deviation is how much the gene expression deviates from baseline (baseline is where deviation = 1)
            # e.g. deviation = 2 means that the gene expression is two times higher than normal
            deviation = 1 + beta * p[:, None] # linear function
            
            # a gene expression lower that 0 is meaningless (and in fact is undefined in the negative binomial distribution)
            # we therefore anneal the gene expression to 0 once it comes close (starting from the cutoff value)
            # this annealing uses an exponential function
            cutoff = 0.1
            a = cutoff / np.e
            b = 1/cutoff
            deviation = jnp.where(deviation < cutoff, cutoff, deviation)
            
            # rho = freq times the annealed deviation
            rho = freq * deviation
            if self.eval:
                numpyro.deterministic("rho", rho)
            
            # we again sample the observed gene expression as before
            numpyro.sample("transcriptome", distributions.NegativeBinomial(rho * library[:, None], dispersion).to_event(1), obs = Y)
                        
            
# define a *switch* model, where the gene expression jumps up or down (=beta) at a particular perturbation value (=switch)
class ModelSwitch():
    def __init__(self, dataset):
        self.dispersion_dist = dataset.get_empirical_dist("dispersion")
        self.freq_dist = dataset.get_empirical_dist("freq")
        self.beta_dist = dataset.get_empirical_dist("deviation")
        self.switch_dist = dataset.get_empirical_dist("switch")

        self.n_genes = dataset.var.shape[0]
        
        self.eval = False
        
    def forward(self, p, library, Y = None, skew = 50):

        with numpyro.plate("genes", self.n_genes):
            dispersion = numpyro.sample('transcriptome/dispersion', self.dispersion_dist)
            freq = numpyro.sample('transcriptome/freq', self.freq_dist)
            beta = numpyro.sample('transcriptome/perturb/beta', self.beta_dist)
            switch = numpyro.sample('transcriptome/switch', self.switch_dist)

        n_cells = len(library)
        with numpyro.plate("cells", len(library)):
            freq = freq[None, :].repeat(n_cells, 0) # expand to cells by genes
            dispersion = dispersion[None, :].repeat(n_cells, 0) # expand to cells by genes
            
            # approximate the switch model using a "relaxation"
            sigmoid = 1 / (1 + jnp.exp(-skew * (p[:, None] - switch)))
            deviation = 1 + beta * sigmoid
            
            # a gene expression lower that 0 is meaningless (and in fact is undefined in the negative binomial distribution)
            # we therefore anneal the gene expression to 0 once it comes close (starting from the cutoff value)
            # this annealing uses an exponential function
            cutoff = 0.1
            deviation = jnp.where(deviation < cutoff, cutoff, deviation)
            
            # rho = freq times the annealed deviation
            rho = freq * deviation
            if self.eval:
                numpyro.deterministic("rho", rho)
            
            # we again sample the observed gene expression as before
            numpyro.sample("transcriptome", distributions.NegativeBinomial(rho * library[:, None], dispersion).to_event(1), obs = Y)