import numpy as np

import statsmodels.api as sm


def test_ARIMA(y: np.array, params: tuple, penalty_criteria: str = "BIC") -> float:
    """
    Measures penalty term used to fit an Autoregressive Integrated Moving Average (ARIMA) model
    
    Params
    ------
    y (np.array): Array of input variables
    params (tuple): 3 values p, d, and q. (p, d, q)
    penalty_criteria (str): Penalty penalty_criteria used to evaluate the model. 
        -'BIC' - Bayes Information Criterion
        -'AIC' - Akaike Information Criterion
        -'AICc' - Akaike Information Criterion with small sample correction
        -'HQIC' - Hannan-Quinn Information Criterion
        -'SSE' - Sum of squared errors
        -'MSE' - Mean squared error    
    Returns
    -------
    Returns the penalty value as a float values or infinite value (np.inf) if the ARIMA model fails due to an error.  
    A printed message stating that the proper penalty_criteria was not selected to measure the model.
    
    """
    try:
        model_fit = sm.tsa.arima.ARIMA(
            endog=y, order=params, enforce_invertibility=True
        ).fit()
    except:
        print(f"The params: {params} failed to fit the ARIMA. Returning np.inf value.")
        return np.inf

    if penalty_criteria == "BIC":  # Bayes Information Criterion (BIC)
        return model_fit.bic
    elif penalty_criteria == "AIC":  # Akaike Information Criterion (AIC)
        return model_fit.aic
    elif (
        penalty_criteria == "AICc"
    ):  # Akaike Information Criterion with small sample correction (AICc)
        return model_fit.aicc
    elif penalty_criteria == "HQIC":  # Hannan-Quinn Information Criterion (HQIC)
        return model_fit.hqic
    elif penalty_criteria == "SSE":  # Sum of squared errors (SSE)
        return model_fit.sse
    elif penalty_criteria == "MSE":  # Mean squared error (MSE)
        return model_fit.mse
    else:
        raise Exception(
            "Proper penalty_criteria was not selected to measure the model.\n\
        It needs to be either 'BIC', 'AIC', 'AICc', 'HQIC', 'SSE', or 'MSE'."
        )
