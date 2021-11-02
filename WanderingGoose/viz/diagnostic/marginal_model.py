import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from WanderingGoose.stats.smoothers import smooth_lowess


def marginal_model_plots(y, yhat, X, smoothness=0.5):

    # screen out binary("dummy") columns
    new_cols_list = []

    for col in X.columns:

        if X[col].nunique() > 2:
            new_cols_list.append(col)

    X = X[new_cols_list]

    def find_number_features(X):

        try:
            return X.shape[1]
        except:
            return 1

    def find_number_rows(num_features):

        num_rows = int(num_features / 4)
        if num_features % 4 > 0:
            num_rows += 1
        return num_rows

    def find_number_columns(num_features):

        if num_features < 4:
            return num_features

        else:
            return 4

    num_features = find_number_features(X)
    num_rows = find_number_rows(num_features)
    num_columns = find_number_columns(num_features)

    fig, ax = plt.subplots(
        num_rows,
        num_columns,
        sharex=False,
        figsize=(num_columns * 10, num_rows * 10),
        dpi=200,
    )
    fig.suptitle("Marginal Model Plots\n", color="orange", fontsize=25)

    row = 0
    col = 0
    coeff_num = 1
    y_smooth = smooth_lowess(y)
    yhat_smooth = smooth_lowess(yhat)

    # start off by turning off all the axis

    #     for c in range(0,num_columns):
    #         for r in range(0,num_rows):
    #             ax[r,c].axis('off')

    for column_name in X.columns:

        # turn the axis back on for those boxes that have plots
        # ax[row,col].axis('on')

        values = X[column_name]
        smoothed_values = smooth_lowess(values)
        x_max = X[column_name].max()
        x_min = X[column_name].min()
        y_max = y.max()
        y_min = y.min()

        data = (
            pd.DataFrame({"x": values, "y": y, "yhat": yhat})
            .sort_values("x")
            .reset_index(drop=True)
        )
        data["y_smooth"] = smooth_lowess(
            data["y"], lowess_window_length=100, smoothing_iterations=5
        )
        data["yhat_smooth"] = smooth_lowess(
            data["yhat"], lowess_window_length=100, smoothing_iterations=5
        )
        data["x_smooth"] = smooth_lowess(
            data["x"], lowess_window_length=100, smoothing_iterations=5
        )

        values = data["x"]
        smoothed_values = data["x_smooth"]
        x_max = X[column_name].max()
        x_min = X[column_name].min()
        y_max = y.max()
        y_min = y.min()

        #         if num_rows == 1:
        #             print('THIS SHIT:' num_columns, num_rows)

        #             if num_columns == 1:
        #                 fig, ax = plt.subplots(figsize=(num_columns*10,num_rows*10, squeeze=False), dpi = 200)

        #             ax[col].set_title(column_name, fontsize = 40)
        #             ax[col].scatter(data['x'], data['y'], s = 1, color = 'black', alpha = 0.9)
        #             ax[col].scatter(data['x'], data['yhat'], s = 1, color = 'blue', alpha = 0.9)

        #             ax[col].plot(data['x_smooth'], data['y_smooth'], color = 'black', linestyle = '-', linewidth = 5.5, alpha = 0.55, label = 'Actual')
        #             ax[col].plot(data['x_smooth'], data['yhat_smooth'], color = 'blue', linestyle = '--', linewidth = 5.5, alpha = 0.95, label = 'Predicted')

        #             ax[col].set_ylabel('y',  rotation=0, fontsize = 20, labelpad=20)
        #             ax[col].set_xlabel('Coeff {}'.format(coeff_num), fontsize = 20, labelpad=20)

        #             ax[col].set_xlim(x_min,x_max)
        #             ax[col].set_ylim(y_min,y_max)
        #             ax[col].legend(fontsize = 24, loc = 1)

        #        else:
        if num_rows == 1 and num_columns == 1:

            ax = np.array([[ax]])

        ax[row, col].set_title(column_name, fontsize=40)
        ax[row, col].scatter(data["x"], data["y"], s=1, color="black", alpha=0.9)
        ax[row, col].scatter(data["x"], data["yhat"], s=1, color="blue", alpha=0.9)

        ax[row, col].plot(
            data["x_smooth"],
            data["y_smooth"],
            color="black",
            linestyle="-",
            linewidth=5.5,
            alpha=0.55,
            label="Actual",
        )
        ax[row, col].plot(
            data["x_smooth"],
            data["yhat_smooth"],
            color="blue",
            linestyle="--",
            linewidth=5.5,
            alpha=0.95,
            label="Predicted",
        )

        ax[row, col].set_ylabel("y", rotation=0, fontsize=20, labelpad=20)
        ax[row, col].set_xlabel("Coeff {}".format(coeff_num), fontsize=20, labelpad=20)

        ax[row, col].set_xlim(x_min, x_max)
        ax[row, col].set_ylim(y_min, y_max)
        ax[row, col].legend(fontsize=24, loc=1)

        coeff_num += 1
        col += 1
        if col >= 4:
            col = 0
            row += 1
