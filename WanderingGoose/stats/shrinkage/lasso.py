def lasso_selector(X, y, test_size=0.40, random_state=42):

    # perform the lasso using SelectFromModel with the esimator equal to LassoCV

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.40, random_state=42
    )

    lasso_selector = SelectFromModel(estimator=LassoCV()).fit(X_train, y_train)

    lasso_features = pd.DataFrame()
    lasso_features["feature"] = X_train.columns
    lasso_features["selected"] = lasso_selector.get_support()
    lasso_features["coeff"] = lasso_selector.estimator_.coef_

    lasso_features = lasso_features.loc[lasso_features["selected"] == True]

    # put features in a list and display results
    lasso_features_list = lasso_features.feature.to_list()

    print("\n---TRAIN SET-----\n\n")
    ols_regression(
        y=y_train,
        X=X_train[lasso_features_list],
        add_intercept=True,
        display_summary_of_fit=True,
        display_analysis_of_variance=True,
        display_parameter_estimates=True,
        display_diagnostic_plots=True,
        display_leverage_plots=False,
        display_marginal_model_plots=True,
        return_data=False,
    )

    # check accuracy on the test set
    lr = LinearRegression().fit(X_train, y_train)
    yhat = lr.predict(X_test)
    results = pd.DataFrame({"y": y_test, "yhat": yhat})
    results["residuals"] = results.y - results.yhat

    from statsmodels.graphics.gofplots import ProbPlot

    with plt.style.context("ggplot"):

        fig = plt.figure(figsize=(15, 10), dpi=200)

        ax1 = plt.subplot2grid((40, 10), (0, 0), rowspan=14, colspan=8)
        ax2 = plt.subplot2grid((40, 10), (20, 0), rowspan=14, colspan=8)

        fmt = "{x:,.0f}"
        tick = ticker.StrMethodFormatter(fmt)
        residuals_limit = max(min(results), max(results))

        ax1.set_title(
            "\nActual (Test) by Predicted (Trained Model)\n\n",
            fontsize=14,
            color="darkorange",
        )
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
        ax1.set_xlabel("Predicted By Trained Model", fontsize=14)
        ax1.set_ylabel("Test Set Values", fontsize=14)
        ax1.tick_params(axis="x", rotation=35)

        ax2.set_title("\nResiduals vs. Predicted\n", fontsize=14, color="darkorange")
        ax2.scatter(
            results.yhat,
            results.residuals,
            s=20,
            color="black",
            facecolors="burlywood",
            edgecolors="black",
            marker="o",
        )
        ax2.axhline(y=0, color="salmon", linestyle="--", label="Zero Line")
        ax2.xaxis.set_major_formatter(tick)
        ax2.set_xlabel("Predicted By Trained Model", fontsize=14)
        ax2.set_ylabel("\nResiduals From Test Values", fontsize=14)
        # ax2.set_ylim(-residuals_limit*1.10, residuals_limit*1.10)
        ax2.tick_params(axis="x", rotation=35)
        ax2.legend(loc=2)

        plt.tight_layout()
        plt.show()

        return lasso_features_list
