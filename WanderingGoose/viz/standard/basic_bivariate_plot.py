import matplotlib.pyplot as plt
import numpy as np


def basic_bivariate_plot(
    x, y, title: str = None, figsize=(5, 5), show_correlation: bool = True,
) -> plt.Axes:
    """
    Creates a basic bivariate plot with confidence limits.

    Params
    ------
    x, y : array-like or scalar
        The horizontal / vertical coordinates of the data points.
        *x* values are optional and default to ``range(len(y))``.

        Commonly, these parameters are 1D arrays.

        They can also be scalars, or two-dimensional (in that case, the
        columns represent separate data sets).

        These arguments cannot be passed as keywords.

    title : str
        The title for the chart if desired

    figsize : tuple of (width, height)

    show_correlation : bool
        Flag used to control display of the [Pearson's r][2] (correlation
        coefficient or product-moment) for the chart.

    Returns
    -------
    A matplotlib ``Axes`` object

    NOTES:
    Originally based on code from [Matplotlib docs][1].

    [1]: https://matplotlib.org/stable/gallery/lines_bars_and_markers/fill_between_demo.html#sphx-glr-gallery-lines-bars-and-markers-fill-between-demo-py
    [2]" https://en.wikipedia.org/wiki/Pearson_correlation_coefficient
    """

    # Fit a linear curve an estimate its y-values and their error.
    a, b = np.polyfit(x, y, deg=1)
    y_est = a * x + b
    y_err = x.std() * np.sqrt(
        1 / len(x) + (x - x.mean()) ** 2 / np.sum((x - x.mean()) ** 2)
    )

    # Build plot
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(x, y_est, "-")
    ax.fill_between(x, y_est - y_err, y_est + y_err, alpha=0.2)

    # Some styles only make sense if the number of points to plot
    # are below certain thresholds...
    marker = "o" if len(x) < 50 else "."
    ax.scatter(x, y, marker=marker, color="tab:brown", ec="k", alpha=0.5)

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
        ax.set_title(title)

    return ax
