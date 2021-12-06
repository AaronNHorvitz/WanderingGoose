import numpy as np
import pandas as pd
import statsmodels.api as sm


def smooth_lowess(
    y_series: pd.Series,
    lowess_window_length: int = 7,
    smoothing_iterations: int = 2,
    fill_missing: bool = True,
    allow_negative_values: bool = False,
) -> np.array:
    """
    This is a customized LOWESS (locally weighted scatterplot smoothing) implementation using Statstmodels, that attempts to
    "smooth" a discrete data set to a continuous one.  In the case of the LOWESS, each smoothed value is given by a weighted
    linear least squares regression over the span.
    
    Params
    ------
    y_series:  Pandas series of discrete points as the inputs variable y
    lowess_window_length:  This is the window length passed smoothed through the data set. It cannot be less than 3. 
    smoothing_iterations: number of times to iterate the smoother
    fill_missing: If True then it interpolates NaN values using a the 'linear' interpolate method. 
    allow_negative_values: If False, then all negative values are replaced with a zero.
    
    Returns
    -------
    yhat: a Pandas series of the smoothed values.
    
    """

    y_series = y_series.fillna(0)  # replace all NaN values with 0
    x_series = list(np.arange(0, len(y_series), 1))
    print("lowess_window_length:  ", lowess_window_length)

    window = lowess_window_length / len(x_series)

    if allow_negative_values == False:
        y_series[y_series < 0] = 0  # replace all values less than zero with 0

    # if the dataset has NaN values, they will be replaced with linear interpolation
    if fill_missing == True:
        y_series = y_series.replace(to_replace=0, value=None).interpolate(
            method="linear"
        )
        y_series = y_series.fillna(0)

    # log transform the series
    y_series = np.log1p(y_series)

    lowess = sm.nonparametric.lowess
    smooth = lowess(y_series, x_series, frac=window, it=smoothing_iterations)
    index, yhat = np.transpose(smooth)

    # reverse transform the series by taking the inverse log 1p
    yhat = np.expm1(yhat)

    return yhat
