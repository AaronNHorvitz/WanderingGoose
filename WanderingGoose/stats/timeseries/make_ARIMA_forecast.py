import pandas as pd
import numpy as np

import statsmodels.api as sm

def generate_ARIMA_forecast(
    y:pd.Series, 
    params:tuple, 
    forecast_length:int, 
    lookback_window:int=100, 
    allow_negative_values:bool=True
    )->np.array:

    """
    Generates a forecast of a given forecast_length based on the best parameters (p,d,q) to fit an ARIMA model with, the length
    of previous rows to train the ARIMA model on
    
    Params
    ------
    y (pd.Series) - The original data set to build the ARIMA model with. 
    params (tuple) - (p,d,q) parameters used to fit the ARIMA model.
    forecast_length (int) - length of time to forecast to
    lookback_window (int) - a slice back from the most recent row, going back the given number of rows to train the model on
    allow_negative_values (bool) - If negative values are allowed then True.

    Returns
    -------
    Forecast - Numnpy array of the length of the ``forecast_length``. 
    """

    # Slice initial values back to the lookback window
    y_train = y.iloc[-lookback_window:,]
    y_train = np.array(y_train)

    # Log transform to prevent negative values and first difference the values
    if not allow_negative_values:
        y_train = np.log1p(y_train)

    last_val = y_train[-1]
    y_train = np.diff(y_train)

    # Rebuild the ideal model based on the params and forecast out to the forecast length. 
    model_fit = sm.tsa.arima.ARIMA(endog=y_train, order=params, enforce_invertibility=True).fit()
    forecast = model_fit.forecast(steps = forecast_length)

    # Retransform values
    forecast = np.insert(forecast,0, last_val)  # adding in last the last recorded actual value as an anchor to the 
                                                # the cum sum and log transform process to the forecast. This should
                                                # make the forecast 1 value too long, since the first forecast value
                                                # will be an actual value. 
    forecast = np.cumsum(forecast)

    # Reverse Log transform if preventing negative values
    if not allow_negative_values:
        forecast = np.expm1(forecast)

    forecast = np.delete(forecast, 0)           # delete the first forecast value so the forecast conforms to the proper length. 

    return forecast