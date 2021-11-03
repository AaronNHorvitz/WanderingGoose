import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from IPython.display import display

from WanderingGoose.stats.regression.least_squares import wls_regression
from WanderingGoose.stats.tests.model_estimators import get_wls_metrics

def get_fitted_wls_model(y, X, w, add_intercept=True):

    X = sm.add_constant(X) if add_intercept == True else X
    fitted_wls_model = sm.WLS(y, X, weights=w).fit()

    return fitted_wls_model

def forward_remove_next_feature(X, y, w, add_intercept=True):

    fitted_wls_model = get_fitted_wls_model(X, y, w, add_intercept=True)
    df = pd.DataFrame(fitted_wls_model.pvalues)
    df = pd.DataFrame(fitted_wls_model.pvalues).sort_values(by=0, ascending=False)
    df = df.loc[df.index != "const"]
    remove_feature = df.index[0]
    sig_prob = df[0][0]
    return sig_prob, remove_feature


def backwards_linear_selection(y, X, w=None, stopping_rule="BIC"):

    if w == None:
        w = 1
        X["weight"] = 1
        X = X.drop(
            columns="weight"
        )  # extra check to remove the weight columns if an assigned one exists

    original_X = X.copy()

    features_list = X.columns.to_list()
    remove_from_features_list = X.columns.to_list()

    step_history_list = []
    remaining_columns_list = []
    sig_prob_list = []
    aic_list = []
    bic_list = []
    aicc_list = []
    rsquared_adj_list = []
    remove_list = ["All"]
    action_list = ["Entered"]

    for step in range(0, len(X.columns)):

        fitted_wls_model = get_fitted_wls_model(X, y, w, add_intercept=True)
        aic, bic, aicc, rsquared_adj = get_wls_metrics(fitted_wls_model)
        sig_prob, remove_feature = forward_remove_next_feature(
            y, X, w, add_intercept=True
        )

        step_history_list.append(step + 1)
        remove_list.append(remove_feature)
        action_list.append("Removed")
        remaining_columns_list.append(X.columns.to_list())
        sig_prob_list.append(round(sig_prob, 6))
        aic_list.append(aic)
        aicc_list.append(aicc)
        bic_list.append(bic)
        rsquared_adj_list.append(rsquared_adj)

        remove_from_features_list.remove(remove_feature)

        X = X[remove_from_features_list]

    action_list = action_list[0:-1]
    remove_list = remove_list[0:-1]

    step_history_dict = {
        "Step": step_history_list,
        "Parameter": remove_list,
        "Action": action_list,
        "Sig Prob": sig_prob_list,
        "AIC": aic_list,
        "AICc": aicc_list,
        "BIC": bic_list,
        "RSquare Adj": rsquared_adj_list,
        "remaining_cols": remaining_columns_list,
    }

    step_history = pd.DataFrame(step_history_dict)

    min_amount = step_history[stopping_rule].min()
    best_cols = step_history.loc[
        step_history[stopping_rule] == step_history[stopping_rule].min()
    ]["remaining_cols"].values[0]

    best_cols = step_history.loc[
        step_history[stopping_rule] == step_history[stopping_rule].min()
    ].copy()
    best_cols["Step"] = step + 2
    best_cols["Parameter"] = "Best"
    best_cols["Action"] = "Specific"

    # Plot the stepwise progress....
    fig, ax1 = plt.subplots(figsize=(13.5, 4))
    ax1.set_xlabel("Steps", color="black", fontsize=15)
    ax1.set_ylabel("AIC-BIC", color="black", fontsize=15)
    ax1.plot(step_history["Step"], step_history["AIC"], label="AIC")
    ax1.plot(step_history["Step"], step_history["BIC"], label="BIC")
    ax1.tick_params(axis="y", labelcolor="black")
    plt.legend(loc=2, fontsize=14)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel(
        "R-Squared Adj.", color="black", fontsize=15
    )  # we already handled the x-label with ax1
    ax2.plot(
        step_history["Step"],
        step_history["RSquare Adj"],
        label="RSquare Adj",
        color="dodgerblue",
        linestyle="--",
    )
    ax2.tick_params(axis="y", labelcolor="black")

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.title("\nStepwise Fit Backward\n", fontsize=20)
    plt.legend(loc=1, fontsize=14)
    plt.show()

    # Display the stepwise results dataframe
    step_history = step_history.append(best_cols, ignore_index=True)
    display(step_history)

    # Get the selected columns based on the stopping rule:
    col_list = best_cols.values[0][-1]

    results, fitted_ols_model = wls_regression(
        y=y,
        X=original_X[col_list],
        w=w,
        add_intercept=True,
        display_summary_of_fit=True,
        display_analysis_of_variance=True,
        display_parameter_estimates=True,
        display_diagnostic_plots=False,
        display_leverage_plots=False,
        display_marginal_model_plots=False,
        return_data=True,
    )

    return results, fitted_ols_model, step_history
