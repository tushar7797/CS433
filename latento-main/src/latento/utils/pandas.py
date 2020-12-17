import pandas as pd

def crossing(*dfs):
    for df in dfs:
        df["___key"] = 0
    if len(dfs) == 0:
        return pd.DataFrame()
    dfs = [df for df in dfs if df.shape[0] > 0] # remove empty dfs
    base = dfs[0]
    for df in dfs[1:]:
        base = pd.merge(base, df, on = "___key")
    return base.drop(columns = ["___key"])