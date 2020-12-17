import scipy
import scanpy as sc
import numpy as np
import jax.numpy as jnp
import jax
import pandas as pd
import xarray as xr
import numpyro

class SinglecellDataset():
    obs_id = "cell"
    var_id = "gene"
    
    def __init__(self, adata):
        self.adata = adata

        if "counts" not in self.adata.layers:
            self.adata.layers["counts"] = adata.X.copy()

        self.adata.obs = self.adata.obs.set_index(f"{self.obs_id}", drop = False)
        self.adata.obs[f"{self.obs_id}_ix"] = range(self.adata.obs.shape[0])

        self.adata.var = self.adata.var.set_index(self.var_id, drop = False)
        self.adata.var[f"{self.var_id}_ix"] = range(self.adata.var.shape[0])

        self.modality_ids = []

        self.library = self.counts.sum(1).values

    def __getitem__(self, key):
        return pd.DataFrame(self.adata.layers[key], index = self.counts.index, columns = self.counts.columns)

    @property
    def modalities(self):
        return {modality_id:self.__getattribute__(modality_id) for modality_id in self.modality_ids}

    @property
    def uns(self):
        return self.adata.uns
        
    @property
    def obs(self):
        return self.adata.obs

    @property
    def var(self):
        return self.adata.var

    @property
    def obsm(self):
        return self.adata.obsm

    @property
    def n_features(self):
        return self.adata.n_vars

    @property
    def layers(self):
        return self.adata.layers

    @property
    def counts(self):
        if "counts" in self.adata.layers:
            X = self.adata.layers["counts"]
        else:
            X = self.adata.X
        if isinstance(X, scipy.sparse.csr.csr_matrix):
            return pd.DataFrame(X.toarray(), index = self.obs.index, columns = self.var.index)
        else:
            return pd.DataFrame(X, index = self.obs.index, columns = self.var.index)

    @property
    def rho(self):
        return self.counts / self.library[:, None]
    
    def preprocess(self):
        sc.pp.normalize_total(self.adata)
        sc.pp.log1p(self.adata)
        
        sc.tl.pca(self.adata)
        sc.pp.neighbors(self.adata)

    def feature_id(self, symbol = None, feature_ix = None):
        if symbol is not None:
            if isinstance(symbol, list): assert np.all([symbol_ in self.var["symbol"].tolist() for symbol_ in symbol])
            mapped = self.var.set_index("symbol").loc[symbol][self.var_id]
        elif feature_ix is not None:
            if isinstance(symbol, list): assert np.all([feature_ix_ in self.var[f"{self.var_id}_ix"].tolist() for feature_ix_ in feature_ix])
            mapped = self.var.set_index(f"{self.var_id}_ix").loc[feature_ix]["feature_id"]
        if isinstance(symbol, list) or isinstance(feature_ix, list):
            mapped = mapped.tolist()
        return mapped
    
    def symbol(self, feature_id = None, feature_ix = None):
        if feature_id is not None:
            if isinstance(feature_id, list): assert np.all([feature_id_ in self.var[self.var_id].tolist() for feature_id_ in feature_id])
            mapped = self.var.loc[feature_id]["symbol"]
        elif feature_ix is not None:
            mapped = self.var.set_index(f"{self.var_id}_ix").loc[feature_ix]["symbol"]

        if isinstance(feature_id, list) or isinstance(feature_ix, list):
            mapped = mapped.tolist()
        return mapped

    def feature_ix(self, symbol = None, feature_id = None):
        if symbol is not None:
            mapped = self.var.set_index("symbol").loc[symbol][f"{self.var_id}_ix"]
        elif feature_id is not None:
            mapped = self.var.loc[feature_id][f"{self.var_id}_ix"]
        if isinstance(symbol, list) or isinstance(feature_id, list):
            mapped = mapped.tolist()
        return mapped

    def select_cells(self, cell_ix):
        return self.__class__(self.adata[cell_ix, :])

    def select_features(self, feature_ix):
        return self.__class__(self.adata[:, feature_ix])

    def get_library(self):
        return jnp.array(self.library)

    def get_rho_empirical(self, feature_ids = None):
        if feature_ids is None:
            feature_ids = self.var.index.tolist()
        
        rho_empirical = jnp.array(
            (self.counts[feature_ids].values  / self.library[:,None])
        )

        rm = rho_empirical.mean(0) + (1/self.counts.shape[1]) / 50
        rv = rho_empirical.var(0) + (1/self.counts.shape[1]) / 100

        alpha = (rm**2 * (1 - rm) - (rv * rm)) / (rv)
        beta = (alpha * (1 - rm)) / rm

        assert jnp.all(rm > 0.), "Some features have only zero counts"

        return numpyro.distributions.Beta(alpha, beta)

    def get_empirical_dist(self, variable, feature_ids = None):
        if feature_ids is None:
            feature_ids = self.var.index.tolist()
        if variable == "deviation":
            return self.get_rho_diff_empirical(feature_ids = feature_ids)
        elif variable == "freq":
            x =  self.get_rho_empirical(feature_ids = feature_ids)
            return x
        elif variable == "dispersion":
            return numpyro.distributions.LogNormal(np.log(0.5), 0.5).expand((self.var.shape[0],))
        elif variable == "switch":
            return numpyro.distributions.Uniform(low=0.0, high=1.0).expand((self.var.shape[0],))
        else:
            raise NotImplementedError

    def get_empirical(self, variable, feature_ids = None):
        if feature_ids is None:
            feature_ids = self.var.index.tolist()
        if variable == "deviation":
            return xr.DataArray(((self.rho / self.rho.mean(0)) - (self.rho / self.rho.mean(0)).mean(0))[feature_ids])
        elif variable == "freq":
            return xr.DataArray(self.rho[feature_ids])
        elif variable == "dispersion":
            return pd.Series(np.ones(self.rho.shape), index = self.rho.columns)[feature_ids]
        else:
            raise NotImplementedError

    def get_rho_diff_empirical(self, feature_ids = None, device = None):
        if feature_ids is None:
            feature_ids = self.var.index.tolist()

        # the difference between two normal distributions is mu1 - mu2 and var1 + var2
        # we'll allow 2.5 times the empirical variability
        # I guess this should be a parameter in some way

        var = jnp.array((self.rho[feature_ids] / self.rho[feature_ids].mean(0)).var(0).values) * 5
        var = jax.ops.index_update(var, jnp.isnan(var), 1/(self.rho.shape[0] * 10))

        return numpyro.distributions.Normal(0., np.log(var))

    def get_Y(self):
        return jnp.array(self.counts.values)

    def get_rho(self):
        return jnp.array(self.rho.values)

    def get_weights(self):
        if "weights" in self.obs.columns:
            return jnp.array(self.obs["weights"])
        return jnp.ones(self.obs.shape[0])

    def get_batch(self, device = None):
        self.n_batch = len(self.obs["batch"].cat.categories)

        if not pd.api.types.is_categorical_dtype(self.obs["batch"]):
            self.obs["batch"] = self.obs["batch"].astype("category")
        self.obs["batch"].cat.categories = self.obs["batch"].cat.categories.astype("str")
        
        return jax.nn.one_hot(
            jnp.array(self.obs["batch"].cat.codes.values.copy()),
            num_classes = len(self.obs["batch"].cat.codes)
        )

    def numel(self):
        return np.prod(self.counts.shape)

class PerturbedDataset(SinglecellDataset):
    obs_id = "cell"
    
    def __init__(self, adata, perturb, **kwargs):
        super().__init__(adata, **kwargs)

        self.perturb = perturb
        self.modality_ids = ["perturb"]

    def select_cells(self, cell_ix):
        return self.__class__(self.adata[cell_ix, :], **{modality_id:modality.select_cells(cell_ix) for modality_id, modality in self.modalities.items()})

    def select_features(self, feature_ix):
        return self.__class__(self.adata[:, feature_ix], **self.modalities)

    def get_p(self, perturb_id = None):
        if perturb_id is None:
            perturb_id = self.perturb.var.index[0]
        return jnp.array(self.perturb.X[perturb_id])

    @classmethod
    def from_latenta(cls, dataset):
        adata = dataset.adata
        perturb = Modality(layers = {"counts":dataset.perturb.X}, var = dataset.perturb.var)

        return cls(adata, perturb)


class Modality():
    def __init__(self, default_layer = "counts", layers = None, var = None):
        if layers is None:
            layers = {}
        self.layers = layers

        if var is None:
            var = pd.DataFrame(index = pd.Series(name = "gene", dtype = pd.StringDtype()))
        self.var = var
        self.default_layer = default_layer

    @property
    def X(self):
        return self[self.default_layer]

    def __getitem__(self, key):
        return self.layers[key]

    def __setitem__(self, key, value):
        self.layers[key] = value

    def select_cells(self, cell_ix):
        return self.__class__(
            self.default_layer,
            {layer_id:layer.loc[cell_ix] for layer_id, layer in self.layers.items()},
            self.var
        )