import pandas as pd
import numpy as np

import datetime as dt

from statsmodels.tsa.stattools import acf, pacf

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib as mpl


def time_series_plot(
    df,
    figsize = (7,5),
    time_series_name = None,  
    solo_plot = True, 
    ax = None,
):
    
    # Grab the current axis if none provided. 
    if solo_plot is True and ax is None:
        ax = None
    elif solo_plot is False and ax is None:
        ax=plt.gca()

    # Grab the name of the time series
    if time_series_name == None:
        time_series_name = df['y'].name

    # Build a solo plot
    if solo_plot:
        fig, ax = plt.subplots(figsize=figsize)

    # Determine the horizontal label for the time series and ticker 
    if np.issubdtype(df['ds'], np.datetime64):
        time_series_id_label = 'Date'

    elif np.issubdtype(df['ds'], np.number):
        time_series_id_label = 'Series Number'
    else: 
        time_series_id_label = 'Row Number'

    marker = "o" if len(df['ds']) < 50 else "."

    ax.scatter(
        df['ds'], 
        df['y'],
        marker=marker, 
        s = 70, 
        color="grey", ec="k", alpha=1, 
        label = 'Eigenvalue'
    )

    ax.plot(
        df['ds'],
        df['y'],
        color='steelblue', alpha=1,
        linewidth=1
    )

    ax.set_xlabel(f'\n(x)  {time_series_id_label}', fontsize=12)
    ax.set_ylabel(f'(y)  {time_series_name}\n', fontsize=12)
    ax.set_title(f'\nTime Series {time_series_name}\n', fontsize=12)

    if solo_plot is True:
        plt.show()
        plt.close()
        return 

    return ax

def weekly_seasonal_boxplots(df, figsize=(7,5), solo_plot=True, ax=None):
    
    """
    Creates side-by-side boxplots showing weekly seasonality.
    """
    
    # Get Vals
    day_name_vals = df['ds'].dt.day_name()
    values = df['y']
    
    # Create DataFrame to Plot
    data = pd.DataFrame({'day_name':day_name_vals, 'values':values})
    data = data.pivot(columns='day_name', values = 'values')
    rename_dict = {
        'Monday':'Mon','Tuesday':'Tue','Wednesday':'Wed','Thursday':'Thu','Friday':'Fri','Saturday':'Sat', 'Sunday':'Sun'
    }
    col_names = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
    data = data.rename(columns=rename_dict)[col_names]
    
    # Build a solo plot
    if solo_plot:
        fig, ax = plt.subplots(figsize=figsize)
        
    ax = data.boxplot(
        figsize=(7,5),
        grid=False,
        return_type='axes'
    )
    ax.set_title('\nWeekly\n', fontsize=15)
    
    return ax

def monthly_seasonal_boxplots(df, figsize=(7,5), solo_plot=True, ax=None):
    
    """
    Creates side-by-side boxplots showing monthly seasonality.
    """

    # Get Vals
    month_name_vals = df['ds'].dt.month_name()
    values = df['y']
        
    # Create DataFrame to Plot
    data = pd.DataFrame({'month_name':month_name_vals, 'values':values})
    data = data.pivot(columns='month_name', values = 'values')

    rename_dict = {
        'January':'Jan','February':'Feb','March':'Mar','April':'Apr','May':'May','June':'Jun',
        'July':'Jul', 'August':'Aug','September':'Sep','October':'Oct','November':'Nov','December':'Dec'
    }
    col_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    data = data.rename(columns=rename_dict)[col_names]

    # Build a solo plot
    if solo_plot:
        fig, ax = plt.subplots(figsize=figsize)

    ax = data.boxplot(
        figsize=(7,5),
        grid=False,
        return_type='axes'
    )
    ax.set_title('\nMonthly\n', fontsize=15)
    
    return ax

def yearly_seasonal_boxplots(df, figsize=(18,5), solo_plot=True, ax=None):
    
    """
    Creates side-by-side boxplots showing yearly seasonality.
    """
    
     # Get Vals   
    year_vals = df['ds'].dt.year
    values = df['y']

    # Create DataFrame to Plot
    data = pd.DataFrame({'year':year_vals, 'values':values})
    data = data.pivot(columns='year', values = 'values')
    data = data[data.columns.sort_values()] # make sure the dates are in order
    

    # Build a solo plot
    if solo_plot:
        fig, ax = plt.subplots(figsize=figsize)
        
    ax = data.boxplot(
        grid=False,
        return_type='axes'
    )
    ax.set_title('\nYearly\n', fontsize=15)
    
    return ax


def seasonal_boxplots(df, figsize=(12,9)):
    
    fig = plt.figure(constrained_layout=True, figsize=figsize) 
    specs = gridspec.GridSpec(ncols=2, nrows=2, figure=fig) 

    ax1 = fig.add_subplot(specs[0, 0])
    ax1 = weekly_seasonal_boxplots(df, figsize=(figsize[0]/2,5), solo_plot=False, ax=ax1)
    
    ax2 = fig.add_subplot(specs[0, 1]) 
    ax2 = monthly_seasonal_boxplots(df, figsize=(figsize[0]/2,5), solo_plot=False, ax=ax2)

    ax3 = fig.add_subplot(specs[1, :]) 
    ax3 = yearly_seasonal_boxplots(df, figsize=(figsize[0],5), solo_plot=False, ax=ax3)
    
    plt.show()

def time_series_diagnostic_values(df):

    num_lags= 25 if len(df) >=25 else len(df)

    acf_values, acf_interval, ljung_box_vals, p_vals = acf(
        x=df['y'],
        nlags=num_lags,
        qstat=True,
        alpha=0.05, 
        fft=False
    )

    pacf_values, pacf_interval = pacf(
        x=df['y'],
        nlags=num_lags,
        method='ywadjusted',
        alpha=0.05
    )

    autocorrelations = pd.DataFrame(
        {'Lag':np.arange(num_lags+1),
         'AutoCorr':[val for val in acf_values],
         'ACF Inervals':[val for val in acf_interval],
         'Ljung_Box Q':[val for val in np.insert(ljung_box_vals, 0, None)],
         'p-Value':[val for val in np.insert(p_vals, 0, None)],
         'Partial':[val for val in pacf_values],
         'PACF Intervals':[val for val in pacf_interval]
        }
    )

    return autocorrelations


