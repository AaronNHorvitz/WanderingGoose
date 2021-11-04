import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def poly_fit_plot(
    x,
    y,
    deg = 1, 
    figsize = (7,7),
    title: str = None,
    x_label: str = None,
    y_label: str = None, 
    show_correlation: bool = True,
    show_fit: bool = True, 
    show_confidence_limits = True,
    show_prediction_limits = True,
    ):
    
    # Fit a linear curve and estimate its y-values and their error.
    coeffs = np.polyfit(x, y, deg=deg) 
    y_hat = np.polyval(coeffs, x)
    y_err = y - y_hat

    # Find Confidence Limits
    conf = x.std() * np.sqrt(1 / len(x) + (x - x.mean()) ** 2 / np.sum((x - x.mean()) ** 2))
    upper_conf = y_hat + conf
    lower_conf = y_hat - conf

    # Find Prediction Limits
    n = len(x)
    dof = len(x)+len(y) - 2
    t = stats.t.ppf(1-0.025, df=dof)
    s_err = np.sum(np.power(y_err, 2))
    pred = t * np.sqrt((s_err/(n-2))*(1+1.0/n + (np.power((x-x.mean()),2) / ((np.sum(np.power(x,2))) - n*(np.power(x,2))))))
    upper_pi = y_hat+pred
    lower_pi = y_hat-pred

    # Build plot
    fig, ax = plt.subplots(figsize=figsize)

    # Show Fit
    if show_fit:
        ax.plot(x, y_hat, "-", color = 'darkblue', linewidth = 0.5, label = 'Fit')

    # Confidence Limits
    if show_confidence_limits:
        ax.fill_between(x, lower_conf, upper_conf, alpha=0.2, label = '95% Confidence Limits', color = 'steelblue')

    # Show Prediction Limits
    if show_prediction_limits:
        ax.plot(x, upper_pi, linestyle = "--", dashes=(5, 5), color = 'cornflowerblue', alpha = 1, linewidth = 1, label = '95% Prediction Limits')
        ax.plot(x, lower_pi, linestyle = "--", dashes=(5, 5), color = 'cornflowerblue', alpha = 1, linewidth = 1)

    # Some styles only make sense if the number of points to plot
    # are below certain thresholds...
    marker = "o" if len(x) < 50 else "."
    ax.scatter(x, y, marker=marker, color="white", ec="k", alpha=0.5, label = 'Data')

    # Calc Pearson's r
    if show_correlation:
        corr_coeff = np.corrcoef(x, y)[0, 1]
        ax.text(
            1,
            0,
            f"Pearson's r: {corr_coeff:,.5f}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
        )

    # Add title
    if title:
        ax.set_title(f'{title}\n')

    else: 
        ax.set_title('Poly Fit Plot\n')

    # Add X and Y labels
    if y_label:
        ax.set_ylabel(f'{y_label}\n')

    else:
        ax.set_yabel('Y - Values\n')

    if x_label:
        ax.set_xlabel(f'{x_label}\n')

    else:
        ax.set_xlabel('\nX - Values')    

    # Add legend
    ax.legend(prop={'size': 10.5})

    return ax