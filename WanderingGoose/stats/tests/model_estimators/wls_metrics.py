import statsmodels.api as sm

def get_wls_metrics(fitted_wls_model):

    pvalues = fitted_wls_model.pvalues
    llf = fitted_wls_model.llf
    nobs = fitted_wls_model.nobs
    df_modelwc = len(pvalues)
    aic = fitted_wls_model.aic
    bic = fitted_wls_model.bic
    aicc = sm.tools.eval_measures.aicc(llf, nobs, df_modelwc)
    rsquared_adj = fitted_wls_model.rsquared_adj

    return aic, bic, aicc, rsquared_adj