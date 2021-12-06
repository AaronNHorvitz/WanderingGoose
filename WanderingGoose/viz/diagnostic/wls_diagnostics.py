import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display


def make_wls_diagnostic_plots(results: pd.DataFrame):
    """
    Make Weighted Least Squares diagnostic plots

    Params
    ------
    results : DataFrame
        Results object created by calling `get_wls_results()`
    """
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
            s=20,
            color="black",
            facecolors="burlywood",
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
        ax2.tick_params(axis="x", rotation=35)
        ax2.legend(loc=5)

        ax3.set_title(
            "\nStudentized Residuals vs. \nPredicted\n", fontsize=10, color="darkorange"
        )
        ax3.scatter(
            results.yhat,
            results.studentized_residuals,
            s=20,
            color="black",
            facecolors="burlywood",
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
