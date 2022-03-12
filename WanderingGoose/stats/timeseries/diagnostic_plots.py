import pandas as pd
import numpy as np

import datetime as dt

from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.seasonal import STL
from statsmodels.nonparametric.smoothers_lowess import lowess

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


def seasonal_dompostition_plots(
    df,
    figsize = (15,15),
    time_series_name=None,
    plot_trend=False,
    solo_plot=False
):


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
  
    # Set marker size
    marker = "o" if len(df['ds']) < 50 else "."
    data = pd.DataFrame(
        lowess(endog=df['y'],exog=np.arange(len(df['y'])),frac=0.15)[:, 1], columns=['y_lowess'])
    df = pd.merge(df,data, left_index=True, right_index=True)
    df['residuals'] = df['y']-df['y_lowess']
    
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharey=False, figsize = figsize)
    
    #Time Series Plot with Trend    
    if plot_trend:
        ax1.plot(
            df['ds'],df['y_lowess'],color='red',linestyle='--',linewidth=2.5, label='Trendline'
        )

    ax1.scatter(
        df['ds'], df['y'], marker=marker, s = 70, color="grey", ec="k", alpha=1, label='Actual Values'
    )
    ax1.plot(
        df['ds'], df['y'], color='steelblue', alpha=1, linewidth=1
    )

    #ax1.set_xlabel(f'\n(x)  {time_series_id_label}', fontsize=12)
    ax1.set_ylabel(f'(y)  {time_series_name}\n', fontsize=12)
    ax1.set_title(f'\nTime Series {time_series_name}\n', fontsize=14)
    ax1.legend(fontsize=14)
    
    #Trend Plot
    ax2.plot(
        df['ds'],df['y_lowess'],color='dodgerblue',linestyle='--',linewidth=2.5, label='Trendline'
    )
    #ax2.set_xlabel(f'\n(x)  {time_series_id_label}', fontsize=12)
    ax2.set_ylabel(f'(y)  {time_series_name}\n', fontsize=12)
    #ax2.set_title(f'\nTrend Line Plot\n', fontsize=12)
    ax2.legend(fontsize=14)

    #DeTrend Plot
    ax3.plot(
        df['ds'], df['residuals'], color='steelblue', alpha=1, linewidth=1, label='Detrended Seasonal Plot'
    )
    #ax3.set_xlabel(f'\n(x)  {time_series_id_label}', fontsize=12)
    ax3.set_ylabel(f'(y)  {time_series_name}\n', fontsize=12)
    #ax3.set_title(f'\nDetrended Data {time_series_name}\n', fontsize=12)
    ax3.legend(fontsize=14)
    ax3.axhline(y=0, color='darkgrey',linestyle='--', linewidth=1)

    #Residuals Plot
    ax4.scatter(
        df['ds'], df['residuals'], marker=marker, s = 70, color="grey", ec="k", alpha=1, label='Residual Values'
    )
    ax4.set_xlabel(f'\n(x)  {time_series_id_label}', fontsize=12)
    ax4.set_ylabel(f'(y)  {time_series_name}\n', fontsize=12)
    #ax4.set_title(f'\nDetrended Data {time_series_name}\n', fontsize=12)
    ax4.legend(fontsize=14)
    ax4.axhline(y=0, color='darkgrey', linestyle='--',linewidth=1)

    if solo_plot is True:
        plt.show()
        plt.close()
        return 

def time_series_basic_diagnostics(df, figsize = (14,10)):
    
    seasonal_dompostition_plots(df, figsize = (13,13))
      
    # Obtain vals
    n_lags = 25 if len(df.y) >= 25 else len(df.y) 
    lags = np.arange(0,n_lags+1, 1)

    #ACF Vals
    acf_vals, acf_intervals, ljung_box_vals, p_vals = acf(df['y'], nlags=n_lags, qstat=True, alpha=0.05, fft=False)
    acf_lower = acf_intervals[:,0]-acf_vals
    acf_upper = acf_intervals[:,1]-acf_vals

    #PACF Vals
    pacf_vals, pacf_intervals = pacf(df.y,nlags=n_lags,alpha=0.05)
    pacf_lower = pacf_intervals[:,0]-pacf_vals
    pacf_upper = pacf_intervals[:,1]-pacf_vals

    #Ljung-Box Q and P-vals
    ljung_box_vals = np.insert(ljung_box_vals, 0, np.nan, axis=0)
    p_vals = np.insert(p_vals, 0, np.nan, axis=0)

    # Create side by side figures
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, sharey=True, figsize = figsize, 
                                   gridspec_kw={'width_ratios': [1.5, 3, 2.5, 1.5, 3]})
    fig.subplots_adjust(wspace=0) # remove vertical space between axes

    # Build Text For ACF Plot
    for lag_val, acf_val in zip(lags, acf_vals):
        acf_text = str(acf_val.item())[0:6]
        lag_text = str(lag_val.item())[0:2]
        lag_text = ' {}'.format(lag_text) if len(lag_text) == 1 else lag_text
        acf_text = '1.0000' if acf_text == '1.0' else acf_text

        ax1.text(0.1, lag_val+0.2, lag_text, fontsize=11)
        ax1.text(0.4, lag_val+0.2, acf_text, fontsize=11)

    ax1.text(0.1, -1.0, 'Lag')
    ax1.text(0.4, -1.0, "AutoCorr")
    ax1.tick_params(labelbottom=False,labeltop=False)
    ax1.set_yticks([])
    ax1.set_xticks([])
    ax1.set_axis_off()

    # Build ACF Plot
    ax2.barh(lags, acf_vals, align='center', color='darkgrey',edgecolor='black')
    ax2.plot(acf_lower, lags, color='dodgerblue')
    ax2.plot(acf_upper, lags, color='dodgerblue')
    ax2.set_title('\nAuto Correlation (ACF)\n')

    ax2.set_xlim(-1.1,1.1)
    ax2.set_ylim(-0.5, n_lags+0.5)

    ax2.tick_params(labelbottom=True,labeltop=True)
    ax2.set_yticks([])
    ax2.invert_yaxis()

    # Build Text For Ljung-Box Q
    for lag_val, q_val, p_val in zip(lags, ljung_box_vals, p_vals):

        if p_val < 0.001:
            p_text = '<.0001*'
            p_color = 'darkorange'
        elif p_val < 0.05:
            p_text = '<.0500*'
            p_color = 'maroon'
        else:
            p_text = str(p_val.item())[0:6]
            p_color = 'black'

        q_text = str(q_val.item())[0:7]
        q_text = '.' if q_text == 'nan' else q_text
        p_text = '.' if p_text == 'nan' else p_text

        ax3.text(0.20, lag_val+0.2, q_text, fontsize=11)
        ax3.text(0.65, lag_val+0.2, p_text, fontsize=11, color=p_color)

    ax3.text(0.15, -1.0, 'Ljung-Box Q')
    ax3.text(0.65, -1.0, "p-Value")
    ax3.tick_params(labelbottom=False,labeltop=False)
    ax3.set_yticks([])
    ax3.set_xticks([])
    ax3.set_axis_off()

    # Build Text For PACF Plot
    for lag_val, pacf_val in zip(lags, pacf_vals):
        pacf_text = str(pacf_val.item())[0:6]
        lag_text = str(lag_val.item())[0:2]
        lag_text = ' {}'.format(lag_text) if len(lag_text) == 1 else lag_text
        pacf_text = '1.0000' if pacf_text == '1.0' else pacf_text

        ax4.text(0.1, lag_val+0.2, lag_text, fontsize=11)
        ax4.text(0.4, lag_val+0.2, pacf_text, fontsize=11)

    ax4.text(0.1, -1.0, 'Lag')
    ax4.text(0.4, -1.0, "Partial")
    ax4.tick_params(labelbottom=False,labeltop=False)
    ax4.set_yticks([])
    ax4.set_xticks([])
    ax4.set_axis_off()

    # Build PACF Plot
    ax5.barh(lags, pacf_vals, align='center', color='darkgrey',edgecolor='black')
    ax5.plot(pacf_lower, lags, color='dodgerblue')
    ax5.plot(pacf_upper, lags, color='dodgerblue')
    ax5.set_title('\nPartial Auto Correlation (PACF)\n')

    ax5.set_xlim(-1.1,1.1)
    ax5.set_ylim(-0.5, n_lags+0.5)

    ax5.tick_params(labelbottom=True,labeltop=True)
    ax5.set_yticks([])
    ax5.invert_yaxis()

    plt.show()
    plt.close()




