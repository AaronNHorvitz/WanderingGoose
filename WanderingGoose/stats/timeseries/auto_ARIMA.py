from itertools import product

import numpy as np
import pandas as pd

from WanderingGoose.stats.tests.stationarity_randomwalk.augmented_dickey_fuller_test import (
    augmented_dickey_fuller_test,
)
from WanderingGoose.stats.tests.model_estimators.ARIMA_testing import test_ARIMA


def auto_ARIMA(
    y: pd.Series,
    max_p_and_q: int = 5,
    lookback_window: int = 100,
    penalty_criteria: str = "AIC",
    allow_negative_values=True,
) -> tuple:

    """
    This is a variation of the Hyndman-Khandakar algorithm, with an additional pentalty anchoring the p and q values
    no higher than 1. This helps avoid overfitting the ARIMA model. In addition, an ADF is used to test for stationarity
    instead of the KPSS unit test as recommended by Hyndman-Khandakar to find the parameter 'd' for the differencing order.
    
    Params
    ------
    y (pd.Series) - The original data set to build the ARIMA model with. 
    max_p_and_q (int) - Maximum value for p and q values
    lookback_window(int) - Furthest back point to train the model
    penalty_criteria (str): Penalty penalty_criteria used to evaluate the model. 
        -'BIC' - Bayes Information Criterion
        -'AIC' - Akaike Information Criterion
        -'AICc' - Akaike Information Criterion with small sample correction
        -'HQIC' - Hannan-Quinn Information Criterion
        -'SSE' - Sum of squared errors
        -'MSE' - Mean squared error 
    allow_negative_values (bool) - True if negative values allowed in the model. False if not. 

    Returns
    -------
    Tuple with the selected p,d,q values that produced the lowest measure based on the penalty criteria.
    
    """
    # Slice initial values back to the lookback window
    y = y.iloc[
        -lookback_window:,
    ]

    # The number of differences 0 ≤ d ≤ 2 is determined using repeated KPSS or ADF tests.
    for d in range(1, 3):

        # Log transform to prevent negative values and first difference the values
        if not allow_negative_values:
            y = np.log1p(y)

        y_diff = y.diff().dropna()

        adf_test_results = augmented_dickey_fuller_test(
            y_diff, significance_level="5%", print_test_results=False
        )
        if adf_test_results == True:
            print(f"d = {d}")
            break
    
    # Generate `p` and `q` grid
    pq_range = np.arange(np.minimum(max_p_and_q, 12)) # capped at `max_p_and_q` or 12, whichever is smaller
    pq_values = list(product(pq_range, pq_range))

    # Test different `p` and `q` values
    test_results = []
    for p, q in pq_values:
        result = round(
            test_ARIMA(y, params=(p, d, q), penalty_criteria=penalty_criteria), 7
        )
        test_results.append(result)
        params = (p, d, q)
    index = test_results.index(min(test_results))
    p, q = pq_values[index] # Get min `p`, `q` values
    params = (p, d, q)

    return params
