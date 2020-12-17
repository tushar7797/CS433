import numpy as np
import json
import pandas as pd
import textwrap

def is_jsonable(x):
    try:
        json.dumps(x)
        return True
    except (TypeError, OverflowError):
        return False

# altair saving cannot handle numpy int64, so we convert them to float (which it can handly apparntly, no idea why)
# also, altair tries to save non-serializable objects, so we remove them here
def prepare_altair(df):
    df["order_ix"] = list(range(df.shape[0]))
    for column_id, dtype in df.dtypes.items():
        if dtype == np.int64:
            df[column_id] = df[column_id].astype(float)

    df = df[[column for column in df.columns if is_jsonable(df[column].iloc[0])]]
    return df


def supress_legend(scale):
    scale = scale.copy()
    scale.legend = None
    return scale

def label_expr(mapping:pd.Series):
    return "{" + ",".join([f"{id}:{textwrap.wrap(label, width=30).__str__()}" for id, label in mapping.items()]) + "}" + "[datum.value]"

def style_table_bar(percentage, color):
    return f"background: linear-gradient(to right, {color} {percentage:.0%}, #ffffff00 {percentage:.0%});color:white;font-weight:bold;"