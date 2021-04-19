
import pandas as pd 
import numpy as np

import scipy.optimize as opt
from multiprocessing import Pool
import numpy.polynomial.polynomial as poly

from tqdm.auto import tqdm
import time
import itertools


def normalize_by_sigmoid(df, sig_params):
    return df / sig_params.loc[0]


def normalize_on_background(df):
    return (df / df.iloc[0, :]) - 1


def compute_cts(df, thresh):
    """
    Function to compute Ct for amplification curves.
        df (pd.DataFrame) - dataframe where each column is an amplification curve
        thresh (float) - threshold for computing Ct
    """
    n_rows, n_columns = df.shape

    # Extract insides of df, so as to work with numpy only
    # Note: This is because we will make this function numpy compatible in the future
    cols = df.columns
    x = df.index
    df = df.values

    cts = np.zeros(n_columns)

    for i in range(n_columns):
        y = df[:, i]
        idx, = np.where(y > thresh)
        if len(idx) > 0:
            idx = idx[0]
            p1 = y[idx-1]
            p2 = y[idx]
            t1 = x[idx-1]
            t2 = x[idx]
            cts[i] = t1 + (thresh - p1)*(t2 - t1)/(p2 - p1)
        else:
            cts[i] = -99

    return pd.DataFrame({'Ct': cts}, index=cols)


def remove_background(df, order=0, n_ct_fit=5, n_ct_skip=0):
    """
    Function to remove background from amplification curve using polynomial fit.
        df (pd.DataFrame) - dataframe where each column is an amplification curve
        order (int) - order of polynomial fit
        n_ct_fit (int) - number of samples to use for fit. This is the number of CT you will use
        n_ct_skip (int) - number of initial samples to skip. In this funct is n of CTs to skip before fitting
            (i.e. if n=5 and n_skip=2, use 3rd to 8th data point for fitting)
    """

    # If order < 0, do nothing... Maybe raise ValueError in the future?
    if order < 0:
        return df

    n_rows, n_columns = df.shape

    # x --> indices to use for fitting
    x = np.arange(n_ct_skip, n_ct_skip+n_ct_fit)

    # x_new --> indices for extrapolating the background fit
    x_new = np.arange(n_rows)

    # If order == 0, simply remove mean
    if order == 0:
        return df- df.iloc[n_ct_skip:n_ct_skip+n_ct_fit, :].mean()
    # Otherwise, fit a polynomial
    else:
        df_new = df.copy()
        for col in tqdm(range(n_columns)):
            coefs = poly.polyfit(x, df.iloc[x, col], order)
            df_new.iloc[:, col] = poly.polyval(x_new, coefs)

        return df - df_new


def is_positive_iloc(df, ampl_thresh, ct_thresh):
    return (df.iloc[ct_thresh:, :] > ampl_thresh).all(axis='rows')


def is_positive_loc(df, ampl_thresh, ct_thresh):
    return (df.loc[ct_thresh:, :] > ampl_thresh).all(axis='rows')
