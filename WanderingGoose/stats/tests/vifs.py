import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


def get_vifs(X, add_intercept):

    X = sm.add_constant(X) if add_intercept == True else X

    if (X.columns[0] == "const") and (len(X.columns) == 2):
        vifs = [np.nan, np.nan]

    elif len(X.columns) == 1:
        vifs = [np.nan]

    elif X.columns[0] == "const":
        data = X.drop(columns=["const"]).copy()
        vifs = [variance_inflation_factor(data.values, i) for i in range(data.shape[1])]
        vifs = [np.nan] + vifs
    else:
        data = X.copy()
        vifs = [variance_inflation_factor(data.values, i) for i in range(data.shape[1])]

    vifs = pd.Series(vifs, index=X.columns.to_list(), name="VIF")

    return vifs
