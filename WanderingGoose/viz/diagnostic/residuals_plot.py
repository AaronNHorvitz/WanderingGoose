import numpy as np
import matplotlib.pyplot as plt

def residuals_plot(y, y_hat, figsize = (7,7), title: str = None,):
    
    # Calculate Error
    y_err = y - y_hat

    # Build plot
    fig, ax = plt.subplots(figsize=figsize)
    ax.axhline(y=0, linestyle = "-", color = 'darkblue', linewidth = 0.5, label = 'Fit')

    # Some styles only make sense if the number of points to plot
    # are below certain thresholds...
    marker = "o" if len(y) < 50 else "."
    x = np.arange(0,len(y),1)
    ax.scatter(x, y_err, marker=marker, color="white", ec="k", alpha=0.5, label = 'Residuals')

    # Add title
    if title == None:
        ax.set_title('Residuals Plot \n')
    else:
        ax.set_title(f'Residuals Plot\n{title}\n')
    
    # X and Y Labels
    ax.set_xlabel('\nX')
    ax.set_ylabel('Residuals\n')

    # Add legend
    ax.legend(prop={'size': 10.5})
    
    return ax

