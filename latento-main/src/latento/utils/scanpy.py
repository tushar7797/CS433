import pandas as pd

def rank_genes_groups_df(adata, key='rank_genes_groups'):
    """
    Create a data frame with columns from .uns['rank_genes_groups'] (eg. names, logfoldchanges, pvals). 

    Shouldn't we use https://scanpy.readthedocs.io/en/stable/api/scanpy.get.rank_genes_groups_df.html?
    """

    dd = []
    groupby = adata.uns[key]['params']['groupby']
    for group in adata.obs[groupby].cat.categories:
        cols = []
        # inner loop to make data frame by concatenating the columns per group
        for col in adata.uns[key].keys():
            if col != 'params':
                cols.append(pd.DataFrame(adata.uns[key][col][group], columns=[col]))
        
        df = pd.concat(cols,axis=1)
        df['group'] = group
        dd.append(df)

    # concatenate the individual group data frames into one long data frame
    rgg = pd.concat(dd)
    rgg['group'] = rgg['group'].astype('category')
    return rgg.set_index('group')

def add_obs(adata, df, column_suffix = None):
    """
    Add a dataframe to the obs, optionally with a suffix for the columns
    We'll use ref paths once its available (https://github.com/theislab/anndata/pull/342)

    This is mainly useful when working with multimodal data
    """

    df_obs = df.copy()
    if column_suffix is not None:
        df_obs.columns = [col + "_" + column_suffix for col in df_obs.columns]

    df_obs.index = adata.obs.index
    
    adata.obs = pd.concat([
        adata.obs[[col for col in adata.obs.columns if col not in df_obs.columns]],
        df_obs
    ], 1)

    return None