import pandas as pd
from IPython.display import display
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm

from WanderingGoose.stats.tests.multicollinearity.vifs import get_vifs
from WanderingGoose.viz.wls_diagnostics import make_wls_diagnostic_plots
from WanderingGoose.viz.diagnostic.marginal_model import marginal_model_plots
from WanderingGoose.viz.diagnostic.leverage import make_leverage_plots


def beautify_analysis_of_variance(analysis_of_variance):
    # TODO: this is another report / output; refactor later
    def reformat_p_values(pvalue):

        if float(pvalue) < 0.0001:
            new_pvalue = "<.0001*"

        elif (float(pvalue) >= 0.0001) and (float(pvalue) < 0.05):
            pvalue = round(pvalue, 4)
            new_pvalue = "{}*".format(pvalue)

        else:
            pvalue = round(pvalue, 4)
            new_pvalue = str(pvalue)

        return new_pvalue

    def color_p_values(val):

        if "*" in val:
            val = float(val.replace("<", "").replace("*", ""))
            if val < 0.01:
                color = "darkorange"
            elif (val >= 0.01) and (val < 0.05):
                color = "red"
            else:
                color = "black"
            return "color: %s" % color

        else:
            return "color: black"

    def bold_f_val(val):
        if val == "Prob > F":
            return "font-weight: bold"
        else:
            return "font-weight: normal"

    pvalue = analysis_of_variance.iloc[2, 4]
    new_pvalue = reformat_p_values(pvalue)
    analysis_of_variance.iloc[2, 4] = new_pvalue
    analysis_of_variance["Sum of Squares"] = analysis_of_variance[
        "Sum of Squares"
    ].astype(str)
    analysis_of_variance["Mean Square"] = analysis_of_variance["Mean Square"].astype(
        str
    )
    analysis_of_variance["F Ratio"] = analysis_of_variance["F Ratio"].astype(str)
    s = analysis_of_variance.style.set_properties(**{"text-align": "left"})
    s = s.applymap(color_p_values, subset=["F Ratio"])
    s = s.applymap(bold_f_val, subset=["F Ratio"])
    s = s.set_caption("Analysis of Variance").set_table_styles(
        [
            {
                "selector": "caption",
                "props": [
                    ("color", "darkorange"),
                    ("font-size", "18px"),
                    ("font-weight:", "bold"),
                ],
            }
        ]
    )

    return s


def get_linear_model_anova(fitted_ols_model):
    # TODO: Should this be moved to tests?
    # TODO: is `fitted_ols_model` misleading?

    model_mse = round(fitted_ols_model.mse_model, 3)
    resid_mse = round(fitted_ols_model.mse_resid, 3)
    total_mse = round(fitted_ols_model.mse_total, 3)

    model_df = int(fitted_ols_model.df_model)
    resid_df = int(fitted_ols_model.df_resid)
    total_df = int(model_df + resid_df)

    model_ss = round(model_mse * model_df, 3)
    error_ss = round(resid_mse * resid_df, 3)
    total_ss = round(total_mse * total_df, 3)

    f_ratio = round(fitted_ols_model.fvalue, 4)
    prob_f = round(fitted_ols_model.f_pvalue, 4)

    source = ["Model", "Error", "C.Total"]
    df = [model_df, resid_df, total_df]
    ss = [model_ss, error_ss, total_ss]
    ms = [model_ss, error_ss, ""]
    fr = [f_ratio, "Prob > F", prob_f]

    analysis_of_variance_dict = {
        "Source": source,
        "Deg of Freedom": df,
        "Sum of Squares": ss,
        "Mean Square": ms,
        "F Ratio": fr,
    }

    analysis_of_variance = pd.DataFrame(analysis_of_variance_dict)

    return analysis_of_variance, analysis_of_variance_dict


def beautify_summary_of_fit(summary_of_fit):
    # TODO: this needs to be refactored into a general report / display view for any regression method

    summary_of_fit[""] = round(summary_of_fit[""], 6)
    summary_of_fit = summary_of_fit.astype(str)

    summary_of_fit[""].iloc[-1] = summary_of_fit[""].iloc[-1].replace(".0", "")

    s = summary_of_fit.style.set_properties(**{"text-align": "left"})
    s = s.set_caption("Summary of Fit").set_table_styles(
        [
            {
                "selector": "caption",
                "props": [
                    ("color", "darkorange"),
                    ("font-size", "18px"),
                    ("font-weight:", "bold"),
                ],
            }
        ]
    )

    return s


def get_wls_model(X, y, w, add_intercept=True):

    X = sm.add_constant(X) if add_intercept == True else X
    wls_model = sm.WLS(y, X, weights=w)
    fitted_wls_model = sm.WLS(y, X, weights=w).fit()

    return wls_model, fitted_wls_model


def get_wls_results(y, wls_model, fitted_wls_model):

    # fitted_wls_model = wls_model.fit()

    res = sm.OLS(wls_model.wendog, wls_model.wexog).fit()
    influence = res.get_influence()

    results = pd.DataFrame()
    results["y"] = y  # target variable
    results["yhat"] = fitted_wls_model.predict()  # predictions
    results["residuals"] = results["y"] - results["yhat"]  # residuals
    results[
        "studentized_residuals"
    ] = influence.resid_studentized_external  # studentized residuals
    results["cooks_distance"] = influence.cooks_distance[0]  # cook's distance

    return results


def get_summary_of_fit(X, y, fitted_ols_model, add_intercept):

    rsquared = fitted_ols_model.rsquared
    rsquared_adj = fitted_ols_model.rsquared_adj
    root_mean_square_error = (
        (mean_squared_error(y, fitted_ols_model.predict(sm.add_constant(X)))) * 0.5
        if add_intercept == True
        else (mean_squared_error(y, fitted_ols_model.predict(X))) * 0.5
    )
    mean_of_response = round(y.mean(), 5)
    num_observations = fitted_ols_model.nobs

    bic = fitted_ols_model.bic
    aic = fitted_ols_model.aic

    summary_of_fit_dict = {
        "RSquare": rsquared,
        "RSquare Adj": rsquared_adj,
        "Root Mean Square Error": root_mean_square_error,
        "AIC": aic,
        "BIC": bic,
        "Mean of Response": mean_of_response,
        "Observations (or Sum Wgts)": num_observations,
    }

    summary_of_fit = pd.DataFrame(
        {" ": list(summary_of_fit_dict.keys()), "": list(summary_of_fit_dict.values())}
    )

    return summary_of_fit, summary_of_fit_dict


def get_parameter_estimates(X, fitted_ols_model, add_intercept):

    terms = fitted_ols_model.params.index.to_list()
    estimates = fitted_ols_model.params.to_list()
    standard_errors = fitted_ols_model.bse.to_list()
    t_ratios = fitted_ols_model.tvalues.to_list()
    p_values = fitted_ols_model.pvalues.to_list()
    lower_95 = fitted_ols_model.conf_int(alpha=0.5, cols=None)[0].to_list()
    upper_95 = fitted_ols_model.conf_int(alpha=0.5, cols=None)[1].to_list()
    vifs = get_vifs(X, add_intercept).to_list()

    parameter_estimates_dict = {
        "Term": terms,
        "Estimate": estimates,
        "Std Error": standard_errors,
        "t Ratio": t_ratios,
        "Prob > |t|": p_values,
        "Lower 95%": lower_95,
        "Upper 95%": upper_95,
        "VIF": vifs,
    }

    parameter_estimates = pd.DataFrame(parameter_estimates_dict)

    return parameter_estimates, parameter_estimates_dict


def beautify_parameter_estimates(parameter_estimates):
    def reformat_p_values(pvalue):

        if float(pvalue) < 0.0001:
            new_pvalue = "<.0001*"

        elif (float(pvalue) >= 0.0001) and (float(pvalue) < 0.05):
            pvalue = round(pvalue, 4)
            new_pvalue = "{}*".format(pvalue)

        else:
            pvalue = round(pvalue, 4)
            new_pvalue = str(pvalue)

        return new_pvalue

    def color_p_values(val):

        if "*" in val:
            val = float(val.replace("<", "").replace("*", ""))
            if val < 0.01:
                color = "darkorange"
            elif (val >= 0.01) and (val < 0.05):
                color = "red"
            else:
                color = "black"
            return "color: %s" % color

        else:
            return "color: black"

    def reformat_vifs(vif):

        vif = round(vif, 4)

        if str(vif) == "nan":
            new_vif = "."

        elif vif <= 10:
            new_vif = "{}*".format(vif)

        else:
            new_vif = str(vif)

        return new_vif

    def color_vifs(val):

        if "*" in val:
            val = float(val.replace("*", ""))
            if val <= 5.0:
                color = "darkorange"
            elif (val > 5.0) and (val <= 10.0):
                color = "red"
            else:
                color = "black"
            return "color: %s" % color

        else:
            return "color: black"

    parameters_beautified = parameter_estimates.copy()

    parameters_beautified["Estimate"] = round(parameters_beautified["Estimate"], 5)
    parameters_beautified["Std Error"] = round(parameters_beautified["Std Error"], 5)
    parameters_beautified["t Ratio"] = round(parameters_beautified["t Ratio"], 2)
    parameters_beautified["Prob > |t|"] = parameters_beautified["Prob > |t|"].apply(
        reformat_p_values
    )
    parameters_beautified["Lower 95%"] = round(parameters_beautified["Lower 95%"], 5)
    parameters_beautified["Upper 95%"] = round(parameters_beautified["Upper 95%"], 5)
    parameters_beautified["VIF"] = parameters_beautified["VIF"].apply(reformat_vifs)
    parameters_beautified = parameters_beautified.astype(str)
    parameters_beautified["Term"] = [
        "Intercept" if term == "const" else term
        for term in parameters_beautified["Term"].to_list()
    ]
    s = parameters_beautified.style.set_properties(**{"text-align": "left"})
    s = s.applymap(color_p_values, subset=["Prob > |t|"])
    s = s.applymap(color_vifs, subset=["VIF"])
    s = s.set_caption("Parameter Estimates").set_table_styles(
        [
            {
                "selector": "caption",
                "props": [
                    ("color", "darkorange"),
                    ("font-size", "18px"),
                    ("font-weight:", "bold"),
                ],
            }
        ]
    )
    return s


def wls_regression(
    y,
    X,
    w,
    add_intercept=True,
    display_summary_of_fit=True,
    display_analysis_of_variance=True,
    display_parameter_estimates=True,
    display_diagnostic_plots=True,
    display_leverage_plots=False,
    display_marginal_model_plots=False,
    return_data=False,
):
    y = y.copy()
    w = w.copy()
    X = X.copy()

    wls_model, fitted_wls_model = get_wls_model(X, y, w, add_intercept=add_intercept)
    results = get_wls_results(y, wls_model, fitted_wls_model)

    yhat = results["yhat"]
    residuals = results["residuals"]
    studentized_residuals = results["studentized_residuals"]
    cooks_distance = results["cooks_distance"]

    if display_summary_of_fit:
        summary_of_fit = get_summary_of_fit(
            X, y, fitted_wls_model, add_intercept=add_intercept
        )[0]
        display(beautify_summary_of_fit(summary_of_fit))

    if display_analysis_of_variance:
        analysis_of_variance = get_linear_model_anova(fitted_wls_model)[0]
        display(beautify_analysis_of_variance(analysis_of_variance))

    if display_parameter_estimates:
        parameter_estimates = get_parameter_estimates(
            X, fitted_wls_model, add_intercept=add_intercept
        )[0]
        parameters_beautified = beautify_parameter_estimates(parameter_estimates)
        display(parameters_beautified)

    if display_diagnostic_plots:
        results = get_wls_results(y, wls_model, fitted_wls_model)
        make_wls_diagnostic_plots(results)

    if display_marginal_model_plots:
        # marginal_model_plots(results,X, show_actuals = True, show_predicteds = True, smoothness = 0.5)
        marginal_model_plots(y, yhat, X)

    if display_leverage_plots:
        make_leverage_plots(fitted_wls_model)

    if return_data:
        return results, fitted_wls_model


def ols_regression(
    y,
    X,
    add_intercept=True,
    display_summary_of_fit=True,
    display_analysis_of_variance=True,
    display_parameter_estimates=True,
    display_diagnostic_plots=True,
    display_leverage_plots=False,
    display_marginal_model_plots=False,
    return_data=False,
):
    y = y.copy()
    X = X.copy()

    fitted_ols_model = ols_model_fit(X, y, add_intercept=add_intercept)
    results = get_results(y, fitted_ols_model)
    yhat = results["yhat"]
    residuals = results["residuals"]
    studentized_residuals = results["studentized_residuals"]
    cooks_distance = results["cooks_distance"]

    if display_summary_of_fit:
        summary_of_fit = get_summary_of_fit(
            X, y, fitted_ols_model, add_intercept=add_intercept
        )[0]
        display(beautify_summary_of_fit(summary_of_fit))

    if display_analysis_of_variance:
        analysis_of_variance = get_anova(fitted_ols_model)[0]
        display(beautify_analysis_of_variance(analysis_of_variance))

    if display_parameter_estimates:
        parameter_estimates = get_parameter_estimates(
            X, fitted_ols_model, add_intercept=add_intercept
        )[0]
        display(beautify_parameter_estimates(parameter_estimates))

    if display_diagnostic_plots:
        make_diagnostic_plots(y, fitted_ols_model)

    if display_marginal_model_plots:
        # marginal_model_plots(results,X, show_actuals = True, show_predicteds = True, smoothness = 0.5)
        marginal_model_plots(y, yhat, X)
    if display_leverage_plots:
        make_leverage_plots(fitted_ols_model)

    if return_data:
        return results, fitted_ols_model


def ols_model_fit(X, y, add_intercept=True):

    X = sm.add_constant(X) if add_intercept == True else X
    fitted_ols_model = sm.OLS(y, X).fit()

    return fitted_ols_model


def get_results(y, fitted_ols_model):

    influence = fitted_ols_model.get_influence()
    results = pd.DataFrame()
    results["y"] = y  # target variable
    results["yhat"] = fitted_ols_model.predict()  # predictions
    results["residuals"] = results["y"] - results["yhat"]  # residuals
    results[
        "studentized_residuals"
    ] = influence.resid_studentized_external  # studentized residuals
    results["cooks_distance"] = influence.cooks_distance[0]  # cook's distance


def get_anova(fitted_ols_model):

    model_mse = round(fitted_ols_model.mse_model, 3)
    resid_mse = round(fitted_ols_model.mse_resid, 3)
    total_mse = round(fitted_ols_model.mse_total, 3)

    model_df = int(fitted_ols_model.df_model)
    resid_df = int(fitted_ols_model.df_resid)
    total_df = int(model_df + resid_df)

    model_ss = round(model_mse * model_df, 3)
    error_ss = round(resid_mse * resid_df, 3)
    total_ss = round(total_mse * total_df, 3)

    f_ratio = round(fitted_ols_model.fvalue, 4)
    prob_f = round(fitted_ols_model.f_pvalue, 4)

    source = ["Model", "Error", "C.Total"]
    df = [model_df, resid_df, total_df]
    ss = [model_ss, error_ss, total_ss]
    ms = [model_ss, error_ss, ""]
    fr = [f_ratio, "Prob > F", prob_f]

    analysis_of_variance_dict = {
        "Source": source,
        "Deg of Freedom": df,
        "Sum of Squares": ss,
        "Mean Square": ms,
        "F Ratio": fr,
    }


def make_diagnostic_plots(y, fitted_ols_model):

    results = get_results(y, fitted_ols_model)

    title = pd.DataFrame()
    s = title.style.set_properties(**{"text-align": "left"})
    s = s.set_caption("Diagnostic Plots").set_table_styles(
        [
            {
                "selector": "caption",
                "props": [
                    ("color", "darkorange"),
                    ("font-size", "18px"),
                    ("font-weight:", "bold"),
                ],
            }
        ]
    )
    display(s)

    with plt.style.context("ggplot"):

        fig = plt.figure(figsize=(8, 7), dpi=200)
        # fig.suptitle('Diagnostic Plots', color = 'orange', fontsize=16, horizontalalignment = 'center')

        ax1 = plt.subplot2grid((4, 4), (0, 0), rowspan=2, colspan=2)
        ax2 = plt.subplot2grid((4, 4), (0, 2), rowspan=2, colspan=2)
        ax3 = plt.subplot2grid((4, 4), (2, 0), rowspan=2, colspan=3)
        ax4 = plt.subplot2grid((4, 4), (2, 3), rowspan=2, colspan=1)

        fmt = "{x:,.0f}"
        tick = mpl.ticker.StrMethodFormatter(fmt)
        residuals_limit = max(
            min(results.studentized_residuals), max(results.studentized_residuals)
        )

        ax1.set_title("\nActual by Predicted Plot\n\n", fontsize=10, color="darkorange")
        ax1.scatter(
            results.yhat,
            results.y,
            s=20,
            color="black",
            facecolors="burlywood",
            edgecolors="black",
            marker="o",
        )
        ax1.plot(results.yhat, results.yhat, color="r", linewidth=0.80)
        ax1.ticklabel_format(useOffset=False, style="plain")
        ax1.yaxis.set_major_formatter(tick)
        ax1.xaxis.set_major_formatter(tick)
        ax1.set_xlabel("Predicted", fontsize=10)
        ax1.set_ylabel("Actual", fontsize=10)
        ax1.tick_params(axis="x", rotation=35)

        ax2.set_title(
            "\nStudentized Residuals vs.\n Cook's Distance\n",
            fontsize=10,
            color="darkorange",
        )
        ax2.scatter(
            results.cooks_distance,
            results.studentized_residuals,
            s=10,
            color="black",
            facecolors="lightblue",
            edgecolors="black",
            marker="o",
        )
        ax2.axvline(
            x=0.5, color="salmon", linestyle="--", label="Cook's D = 0.5", linewidth=1.5
        )
        ax2.axvline(
            x=1.0, color="salmon", linestyle="-", label="Cook's D = 1.0", linewidth=1.5
        )
        ax2.set_xlabel("Cook's Distance", fontsize=10)
        ax2.set_ylabel("\nStudentized Residuals", fontsize=10)
        ax2.set_ylim(-residuals_limit * 1.10, residuals_limit * 1.10)
        ax2.set_xlim(-0.25, 3)
        ax2.tick_params(axis="x", rotation=35)
        ax2.legend(loc=5)

        ax3.set_title(
            "\nStudentized Residuals vs. \nPredicted\n", fontsize=10, color="darkorange"
        )
        ax3.scatter(
            results.yhat,
            results.studentized_residuals,
            s=10,
            color="black",
            facecolors="lightblue",
            edgecolors="black",
            marker="o",
        )
        ax3.axhline(y=0, color="salmon", linestyle="--", label="Zero Line")
        ax3.xaxis.set_major_formatter(tick)
        ax3.set_xlabel("Predicted", fontsize=10)
        ax3.set_ylabel("\nStudentized Residuals", fontsize=10)
        ax3.set_ylim(-residuals_limit * 1.10, residuals_limit * 1.10)
        ax3.tick_params(axis="x", rotation=35)
        ax3.legend(loc=2)

        ax4.set_title("Distribution\n\n", fontsize=10, color="darkorange")
        ax4.hist(
            results.studentized_residuals,
            histtype="stepfilled",
            color="burlywood",
            alpha=0.5,
            edgecolor="black",
            linewidth=1,
            bins=20,
            orientation="horizontal",
        )
        ax4.axhline(y=0, color="salmon", linestyle="--")
        ax4.xaxis.set_major_formatter(tick)
        ax4.set_ylim(-residuals_limit * 1.10, residuals_limit * 1.10)
        ax4.tick_params(axis="x")

        plt.tight_layout()
        plt.show()
