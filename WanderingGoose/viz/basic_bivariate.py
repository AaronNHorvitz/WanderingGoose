import numpy as np
import matplotlib.pyplot as plt


def basic_bivariate_plot(x, y, title: str = None, figsize=(5, 5)) -> plt.Axes:
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

    Returns
    -------
    A matplotlib ``Axes`` object

    NOTES:
    Originally based on code from [Matplotlib docs][1].


    [1]: https://matplotlib.org/stable/gallery/lines_bars_and_markers/fill_between_demo.html#sphx-glr-gallery-lines-bars-and-markers-fill-between-demo-py
    """

    # Fit a linear curve an estimate its y-values and their error.
    a, b = np.polyfit(x, y, deg=1)
    y_est = a * x + b
    y_err = x.std() * np.sqrt(
        1 / len(x) + (x - x.mean()) ** 2 / np.sum((x - x.mean()) ** 2)
    )

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(x, y_est, "-")
    ax.fill_between(x, y_est - y_err, y_est + y_err, alpha=0.2)
    ax.plot(x, y, "o", color="tab:brown")

    # Add title
    if title:
        ax.set_title(title)

    return ax
