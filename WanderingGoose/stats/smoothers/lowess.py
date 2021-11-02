import numpy as np

def smooth_lowess(y_series, lowess_window_length=100, smoothing_iterations=2):
    """
    This is a customized LOWESS (locally weighted scatterplot smoothing) implementation that "smooths" a discrete data set.
    In the LOWESS, each smoothed value is given by a weighted linear least squares regression over the span.

    This implementation uses an upstream process to account for missing and zero values and data series that can't have negative values.
    It also allows for a finite window length above 3, rather than a percentage of the total dataset.

    It uses the statstmodels implementation:
    statsmodels.nonparametric.smoothers_lowess.lowess
    source: https://www.statsmodels.org/stable/generated/statsmodels.nonparametric.smoothers_lowess.lowess.html

    Params:

    y_series:  Pandas series of discrete points as the inputs variable y
    lowess_window_length:  This is the window length passed smoothed through the data set. It cannot be less than 3.
    smoothing_iterations: number of times to iterate the smoother

    Returns:
    yhat: a Pandas series of the smoothed values.
    """
    # TODO: Need to update to allow zeros

    import statsmodels.api as sm

    y_series = y_series.fillna(0)  # replace all NaN values with 0
    x_series = list(np.arange(0, len(y_series), 1))

    window = lowess_window_length / len(x_series)

    lowess = sm.nonparametric.lowess
    smooth = lowess(y_series, x_series, frac=window, it=smoothing_iterations)
    index, yhat = np.transpose(smooth)

    return yhat
