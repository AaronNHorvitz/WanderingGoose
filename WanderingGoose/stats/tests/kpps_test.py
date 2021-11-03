import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import kpss

def kpps_test(y:pd.Series, significance:str = '5%', test_around_trend = 'c')->bool:
    
    """
    Kwiatkowski-Phillips-Schmidt-Shin (KPSS), is a type of Unit root test that 
    tests for the stationarity of a given series around a given trend. This function 
    gives the option to test for stationarity either around the trend or not. 
    
    Parameters
    ----------
    y (pd.Series): Series to be tested
    significance (str): Significance level for the KPPS test. Acceptable values are
                        '10%','5%','2.5%', and '1%' Raises an error otherwise. 
    test_around_trend 
    
    """
    if significance not in ['10%','5%','2.5%','1%']:
        raise Exception ("The proper significance (var: significance) level was not selected to measure the KPPS test.\n\
        It needs to be either '10%','5%','2.5%', or '1%'")
    if test_around_trend not in ['c','ct']:
        raise Exception ("The proper regression to test around a trend (var: test_around_trend) was not selected.\n\
        It needs to be either 'ct' to test stationarity around a trend or 'c' to test stationarity around a constant.")
        
    regression = 'ct' if test_around_trend == True else 'c'
    numerical_sig_val = {'10%': 0.10, '5%': 0.05, '2.5%': 0.025, '1%': 0.01}
    statistic, p_value, n_lags, critical_values = kpss(x=y, regression = 'c', nlags = 'auto')
    p_value, selected_sig = critical_values[significance], numerical_sig_val[significance]

    return True if p_value < selected_sig else False