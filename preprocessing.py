import re

import pandas as pd
import numpy as np

def remove_liver_metadata(data: pd.DataFrame):
    # Remove metadata rows
    mask_metadata = data.apply(lambda row: re.match(r"^A_\d", str(row.name)) != None, axis=1)
    data = data.loc[mask_metadata]
    data = data.astype(float)
    return data

def remove_hyppthalamus_metadata(data: pd.DataFrame):
    # Remove metadata rows
    mask_metadata = data.apply(lambda row: re.match(r"^\d", str(row.name)) != None, axis=1)
    data = data.loc[mask_metadata]
    data = data.astype(float)
    return data

def preprocess_expression_data(data: pd.DataFrame, min_max_thr = None, min_var_thr = None):

    # Remove rows with low maximal expression
    if not min_max_thr:
        min_max_thr = data.max(axis = 1).quantile(.025)

    mask = data.apply(lambda row: row.max() > min_max_thr, axis=1)
    data = data.loc[mask]

    # Remove rows with low variance
    if not min_var_thr:
        min_var_thr = data.var(axis = 1).quantile(.1)

    mask = data.apply(lambda row: row.var() > min_var_thr, axis=1)
    data = data.loc[mask]

    return data

