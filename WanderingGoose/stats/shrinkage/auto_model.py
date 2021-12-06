from itertools import chain, combinations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from IPython.display import display
from tqdm.notebook import tqdm_notebook as tqdm
from sklearn.metrics import mean_squared_error

from WanderingGoose.stats.tests.multicollinearity.vifs import get_vifs
from WanderingGoose.stats.regression.least_squares import wls_regression


def all_possible_models(
    y,
    X,
    w=False,
    num_show=10,
    stopping_rule="bic",
    dislay_results=True,
    pval_max=0.5,
    vif_max=10,
    add_intercept=True,
    screen_pvalues=True,
    screen_vifs=True,
):
    # TODO: Add docstring
    min_cols = 1
    max_cols = len(X.columns)
    # if w is false, then adjust the all models so it works like a an OLS, not a WLS

    if w == False:

        X["weight"] = 1
        w = X["weight"]
        X = X.drop(columns="weight")

    def find_num_models(total_cols, max_cols, min_cols: int = 1):

        """
        Takes the maximum number of columns and the minimum number of columns and returns
        an integer value which represents the number of possible models that can be evaluated.

        Parameters:
        total_cols: A non-negative integer
        max_cols: A non-negative integer
        min_cols: A non-negative integer

        Returns:
        num_models: A non-negative integer

        """
        from math import comb

        upper, lower = 0, 0

        for col_num in range(1, total_cols + 1):

            upper += comb(total_cols, col_num)

            if col_num == min_cols - 1:
                lower = upper

        num_models = (
            upper - lower
        )  # Add one to account for the same number of max, min, and total columns

        return num_models

    def num_combinatons_powerset(n):

        # Relation to binomial theorem
        # Abs (2^S)

        # https://en.wikipedia.org/wiki/Power_set

        return int(2 ** n)

    def powerset(iterable):

        # iterable engine made by AJ Hill

        s = list(iterable)
        return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))

    min_cols = min_cols - 1
    if min_cols < 0:
        min_cols = 0

    cols_list = X.columns.to_list()
    total_cols = len(X.columns.to_list())

    if max_cols > total_cols:
        max_cols = total_cols

    total_model_fits = find_num_models(total_cols, max_cols, min_cols)
    total_iterations = num_combinatons_powerset(total_cols) - 1

    print("\nAll Possible Models\n")
    print("Total columns/features:          {}".format(total_cols))
    print("Max coefficients to use:         {}".format(max_cols))
    print("Min coefficients to use:         {}".format(min_cols))
    print("Total number of models to fit:   {}".format(total_model_fits))
    print("Number of combinations:          {}".format(total_iterations))
    print("Number of top models to display: {}".format(num_show))
    print(
        "\n****Fitting {} model(s) out of {} possible combinations?****\n".format(
            total_model_fits, total_iterations
        )
    )

    pbar = tqdm(total=total_iterations - 1)  # display progress...

    sets = list(powerset(cols_list))

    all_models = {}
    model_num = 0

    for cols in sets[1:][::-1]:
        pbar.update(1)

        if (
            len(cols) < min_cols or len(cols) > max_cols
        ):  # screen out columns from max or min
            continue

        X_predictors = X[list(cols)]

        if add_intercept == True:
            X_predictors = sm.add_constant(X_predictors)

        # fitted model
        model_wls = sm.WLS(y, X_predictors, weights=w)  # fit model
        fitted_wls = model_wls.fit()

        # screen for pvalues and vifs

        pvalues = list((fitted_wls.pvalues).values)
        vifs = list(get_vifs(X_predictors, add_intercept).values)

        # screen for pvalues
        if (
            screen_pvalues == True
            and max([0 if np.isnan(val) else val for val in pvalues]) >= pval_max
        ):
            continue

        # screen for vifs
        if (
            screen_vifs == True
            and max([0 if np.isnan(val) else val for val in vifs]) > vif_max
        ):
            continue

        # metrics
        model_num += 1
        num_terms = len(cols)
        coeffs = list((fitted_wls.pvalues).index)

        llf = fitted_wls.llf
        nobs = fitted_wls.nobs
        df_modelwc = len(pvalues)
        aic = fitted_wls.aic
        aicc = sm.tools.eval_measures.aicc(llf, nobs, df_modelwc)
        bic = fitted_wls.bic
        rsquared_adj = fitted_wls.rsquared_adj
        rmse = (
            (mean_squared_error(y, fitted_wls.predict(sm.add_constant(X_predictors))))
            * 0.5
            if add_intercept == True
            else (mean_squared_error(y, fitted_wls.predict(X_predictors))) * 0.5
        )

        all_models[model_num] = [
            coeffs,
            pvalues,
            vifs,
            num_terms,
            aic,
            aicc,
            bic,
            rsquared_adj,
            rmse,
        ]

        # display progress

    pbar.close()
    df = pd.DataFrame(all_models).T

    df.columns = [
        "coeffs",
        "pvalues",
        "vif",
        "num_terms",
        "aic",
        "aicc",
        "bic",
        "rsquared_adj",
        "rmse",
    ]
    df = df.sort_values(by=stopping_rule)

    if dislay_results:
        df_display = df[
            ["coeffs", "num_terms", "aic", "aicc", "bic", "rsquared_adj", "rmse"]
        ]
        display(df_display.head(num_show))

        columns = [feature for feature in df.iloc[0][0] if feature != "const"]

        print("\nBest Model Fit Based on {}".format(stopping_rule.upper()))
        print("\n\nColumns Used:\n")

        for col in columns:
            print(col)
        print("\n\n")
        results, fitted_wls_model = wls_regression(
            y,
            X[columns],
            w,
            add_intercept=True,
            display_summary_of_fit=True,
            display_analysis_of_variance=True,
            display_parameter_estimates=True,
            display_diagnostic_plots=True,
            display_leverage_plots=False,
            display_marginal_model_plots=False,
            return_data=True,
        )

    return df, fitted_wls_model
