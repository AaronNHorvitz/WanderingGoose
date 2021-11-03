import numpy as np

from WanderingGoose.stats.tests.augmented_dickey_fuller_test import augmented_dickey_fuller_test

def auto_ARIMA(
    y:pd.Series,  
    max_p_and_q:int=5, 
    lookback_window:int=100, 
    penalty_criteria:str='AIC',
    allow_negative_values=True

)->tuple:

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
    y = y.iloc[-lookback_window:,]

    # The number of differences 0 ≤ d ≤ 2 is determined using repeated KPSS or ADF tests.
    for d in range(1,3):

        # Log transform to prevent negative values and first difference the values
        if not allow_negative_values: 
            y = np.log1p(y)

        y_diff = y.diff().dropna()

        adf_test_results = (augmented_dickey_fuller_test(y_diff, significance_level = '5%', print_test_results = False))
        if adf_test_results == True: 
            print(f'd = {d}')
            break

    p_vals = [0,1,0,1,2,0,2,1,3,0,3,1,4,0,4,1,5,0,5,1,6,0,6,1,7,0,7,1,8,0,8,1,9,0,9,1,10, 0, 10, 11, 0, 11, 1]
    q_vals = [0,0,1,1,0,2,1,2,0,3,1,3,0,4,1,4,0,5,1,5,0,6,1,6,0,7,1,7,0,8,1,8,0,9,1,9,0 ,10, 1 , 0 ,11, 1, 11]

    p_vals = p_vals[0:q_vals.index(max_p_and_q)+1:] # select based on q_vals because the max val comes latest in the list
    q_vals = q_vals[0:q_vals.index(max_p_and_q)+1:]

    # Account for max_p_and_q
    test_results = []
    for p,q in zip(p_vals, q_vals):
            result = round(test_ARIMA(y, params=(p,d,q), penalty_criteria=penalty_criteria),7)
            test_results.append(result)
            params = (p,d,q)
    index = test_results.index(min(test_results))
    p = p_vals[index]
    q = q_vals[index]
    params = (p,d,q)
    
    return params