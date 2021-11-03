import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from IPython.display import display


def make_leverage_plots(fitted_ols_model):

    title = pd.DataFrame()
    s = title.style.set_properties(**{"text-align": "left"})
    s = s.set_caption("Leverage Plots").set_table_styles(
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

    num_params = len(fitted_ols_model.params)
    rows = (
        (num_params // 3 + 1)
        if ((num_params / 3 - num_params // 3) > 0)
        else (num_params // 3)
    )
    figure_length = rows * 14
    fig = plt.figure(figsize=(14, figure_length / 3))
    fig = sm.graphics.plot_partregress_grid(fitted_ols_model, grid=(4, 3), fig=fig)
    fig.tight_layout(pad=5)
