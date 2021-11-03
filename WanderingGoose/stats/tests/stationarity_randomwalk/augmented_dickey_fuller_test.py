import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller


def augmented_dickey_fuller_test(
    y: np.array, significance_level: str = "5%", print_test_results=True
) -> bool:

    """
    Augmented Dickey-Fuller unit root test for stationarity at the 1%, 5%, and 10%. 
    Uses StatsModles `adfuller` function 
    Documentation: https://www.statsmodels.org/dev/generated/statsmodels.tsa.stattools.adfuller.html
    
    Params
    ------
    - y (array_like, 1d): An ordered series of any scalar or sequence that can be interpreted as an ndarray.  
    - significance_level (str): (Default '5%') Significance level of the test. Values can be '1%', '5%', and '10%'
    - print_test_results (bool): prints test results if True.
    
    Returns
    ------
    Boolean value. True if the ordered series passes the stationarity test or False if it fails. 
    """

    test_results = adfuller(y)  # tuple of test results
    test_statistic = test_results[0]  # test statistic
    pval = test_results[
        1
    ]  # MacKinnon’s approximate p-value based on MacKinnon (1994, 2010)
    critical_value = test_results[4][
        significance_level
    ]  # critical values for the test statistic at the 1%, 5%, and 10% levels.

    # Check Test Results Pass/ Fail
    numerical_sig_val = {
        "1%": 0.01,
        "5%": 0.05,
        "10%": 0.1,
    }

    # Confidence Test (True/False)
    if pval < numerical_sig_val[significance_level]:
        confidence_test = True
    else:
        confidence_test = False

    # Hypothesis Test (True/False)
    if test_statistic < critical_value:
        hypothesis_test = True
    else:
        hypothesis_test = False

    # Full AdFuller Test (True/False)
    if (confidence_test == True) and (hypothesis_test == True):
        adfuller_test_result = True
    else:
        adfuller_test_result = False

    if print_test_results:

        if adfuller_test_result == True:
            test_result = "PASSED"
        else:
            test_result = "FAILED"

        print("\n")
        print(
            "------------------------------------------------------------------------"
        )
        print(f"Augmented Dickey-Fuller unit root test for stationarity: {test_result}")
        print(
            "------------------------------------------------------------------------"
        )
        if hypothesis_test:
            print(
                f"Passed-  test statistic:   {round(test_statistic,5)}  <  {round(critical_value,5)}   :critical value"
            )
        else:
            print(
                f"Failed-  test statistic:   {round(test_statistic,5)}  >=  {round(critical_value,5)}   :critical value"
            )

        if confidence_test:
            print(
                f"Passed-  p-value:           {round(pval,5):0.5}  <  {numerical_sig_val[significance_level]}       :signifcanc"
            )
        else:
            print(
                f"Failed-  p-value:           {round(pval,5)}  >=  {numerical_sig_val[significance_level]}       :significance"
            )
        print("\n")

    return adfuller_test_result
