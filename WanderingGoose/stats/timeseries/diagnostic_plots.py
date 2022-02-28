import pandas as pd
import numpy as np

import datetime as dt

def side_by_side_datetime_boxplots(dates, values, figsize, date_type):
    
    """
    Creates side-by-side boxplots.
    """

    if date_type=='days':
        day_names = df['date'].dt.day_name()
    data = pd.DataFrame({'day_name':day_names, 'values':values})
    data = data.pivot(columns='day_name', values = 'values')
    col_names = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    data = data[col_names]
    ax = data.boxplot(
        figsize=(7,5),
        grid=False,
        return_type='axes'
    )
    
    
    return ax