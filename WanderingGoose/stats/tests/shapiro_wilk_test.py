import numpy as np

from scipy.stats import shapiro

def shapiro_wilk_test(data, alpha:float = 0.5, print_result:bool=False):
    """
    This function conducts a Shapiro-Wilk Test for normality on a data series.

    The Shapiro-Wilk test, proposed in 1965, calculates a W statistic that tests whether a 
    random sample, x1,x2,…,xn comes from (specifically) a normal distribution . Small values 
    of W are evidence of departure from normality.

    Source: US Department of Commerce, National Institute of Standards and Technology (NIST) 
    https://www.itl.nist.gov/div898/handbook/prc/section2/prc213.htm

    Params
    ------
        
        data (np.array): array of sample data
        alpha (float): test threshold
        print_result (bool): selected if printed results are desired
        
    Returns: string with the test results
    """
    # Normality test
    w_statistic, p_value = shapiro(data)

    # Interpret
    if p_value > alpha:

        result = ' > alpha = {:0.5f}'.format(alpha)
        conclusion = "Gaussian (Don't reject H{})".format('\u2092') 

    else:
        result = ' < alpha = {:0.5f}'.format(alpha)
        conclusion = "Not Gaussian (Reject H{})".format('\u2092') 

    result_string = 'W statistic = {:0.5f}\np = {:0.5f}{}\n{}'.format(w_statistic, p_value, result, conclusion)
    
    if print_result:
        print(result_string)
    
    return result_string
