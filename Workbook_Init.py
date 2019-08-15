#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
pd.set_option('display.max_columns', 500)

import numpy as np

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.ticker as ticker

import seaborn as sns
sns.set_style('whitegrid')

from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from pandas.plotting import autocorrelation_plot

from sklearn import metrics

from sklearn.linear_model import LinearRegression

import warnings

import datetime
from datetime import timedelta 

# In[2]:


# Source - Bryan's notebook

# MAPE: Mean Absolute Percentage Error 
## another useful metric --not implemented in sklearn.metrics
## See: https://en.wikipedia.org/wiki/Mean_absolute_percentage_error

def MAPE(y_true, y_pred): 
#     y_true, y_pred = check_arrays(y_true, y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def report_metrics(y_true, y_pred):
    print("Explained Variance:\n\t", metrics.explained_variance_score(y_true, y_pred))
    print("MAE:\n\t", metrics.mean_absolute_error(y_true, y_pred))
    # print("RMSE:\n\t", np.sqrt(metrics.mean_squared_error(y_true, y_pred)))
    # print("MAPE:\n\t", MAPE(y_true, y_pred))
    # print("r^2:\n\t", metrics.r2_score(y_true, y_pred))


# In[9]:


def zipcode_melt(df):
    """Takes in dataframe of Zipcodes and Average Home Value in Columns.
    Returns melted dataframe with Index of year-month datetimes and values of ZipCode and Average Home Value"""
    df_temp = df.drop(columns=['RegionID', 'City', 'State', 'Metro', 'CountyName', 'SizeRank'])
    df_temp = df_temp.melt(id_vars='RegionName', value_name='price', var_name='year-month')
    df_temp['year-month'] = pd.to_datetime(df_temp['year-month'], format='%Y-%m')
    df_temp.set_index('year-month', inplace=True)
    return df_temp


# In[5]:


#create a function that will help us to quickly
def test_stationarity(timeseries, window):

   #Determing rolling statistics
    rolmean = timeseries.rolling(window=window).mean()
    rolstd = timeseries.rolling(window=window).std()

   #Plot rolling statistics:
    fig = plt.figure(figsize=(12, 8))
    orig = plt.plot(timeseries.iloc[window:], color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()

    #Perform Dickey-Fuller test:
    print ('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)


# In[6]:


def subplots_acf_pacf(series):
    fig = plt.figure(figsize=(12,8))
    ax1 = fig.add_subplot(211)
    fig = plot_acf(series, lags=40, ax=ax1)
    ax2 = fig.add_subplot(212)
    fig = plot_pacf(series, lags=40, ax=ax2)
    plt.show()


# In[15]:


def find_your_Ps_and_Qs(df, field_name, d, p_list=range(6), q_list=range(6), train_pct=0.8):
    df_use = df.copy()

    result_list = []
    all_zips = sorted(list(set(df_use.RegionName.values)))
    n_zips = len(all_zips)

    df_temp = df_use[field_name]
    df_temp = df_temp.groupby("year-month").mean()

    # split for pseudo forecasting

    train_num = int(np.round(len(df_temp) * train_pct, 0))

    train = df_temp[:train_num]
    test = df_temp[train_num:]

    output = []

    for q in q_list:
        for p in p_list:
            try:
                model = ARIMA(train, order=(p, d, q))
                results_AR = model.fit(disp=-1)

                y_pred = pd.Series(results_AR.forecast(steps=len(test))[0], index=test.index)

                temp_dict = {
                    "p": p,
                    "d": d,
                    "q": q,
                    "aic": results_AR.aic,
                    "bic" : results_AR.bic,
                    "variance": metrics.explained_variance_score(test.squeeze(), y_pred.squeeze()),
                    "mae": metrics.mean_absolute_error(test.squeeze(), y_pred.squeeze()),
                    "rmse": np.sqrt(metrics.mean_squared_error(test.squeeze(), y_pred.squeeze())),
                    "mape": MAPE(test.squeeze(), y_pred.squeeze()),
                    "r2": metrics.r2_score(test.squeeze(), y_pred.squeeze())
                }

                output.append(temp_dict)
            except:
                pass

    output_df = pd.DataFrame(output).sort_values('r2', ascending=False)
    col_order = ['p', 'd', 'q', 'r2', 'aic', 'bic', 'variance', 'mae', 'rmse', 'mape']
    output_df = output_df[col_order]
    return output_df


# In[16]:


def ARIMA_allZips(df, field_name, pdq_list, verbose=True):

    result_list = []

    df_use = df.copy()

    all_zips = sorted(list(set(df_use.RegionName.values)))
    n_zips = len(all_zips)


    for i in range(n_zips):

        zip_idx = all_zips[i]

        df_temp = df_use[df_use.RegionName==zip_idx].dropna().copy()[field_name]

        run_success = False
        trys = 0
        while run_success == False and trys<=len(pdq_list):
            try:
                p, d, q = pdq_list[trys]
                model = ARIMA(df_temp, order=(p, d, q), freq='MS')
                results_AR = model.fit(disp=-1)
                int(results_AR.aic)
                run_success = True 
            except:
                trys += 1
                run_success = False


        if run_success:
            temp_dict = {}
            temp_dict['zip'] = zip_idx
            temp_dict['aic'] = results_AR.aic
            temp_dict['bic'] = results_AR.bic
            temp_dict['curr_price'] = df_temp.values[-1]
            forecast_10 = results_AR.forecast(steps=120)
            ci95_10 = forecast_10[-1][-1]
            forecast_1 = results_AR.forecast(steps=12)
            ci95_1 = forecast_1[-1][-1]
            temp_dict['10yr_Forecast'] = forecast_10[0][-1]
            temp_dict['10yr_95CI_Low'] = ci95_10[0]
            temp_dict['10yr_95CI_High'] = ci95_10[1]
            temp_dict['1yr_Forecast'] = forecast_1[0][-1]
            temp_dict['1yr_95CI_Low'] = ci95_1[0]
            temp_dict['1yr_95CI_High'] = ci95_1[1]
            temp_dict['p'] = p
            temp_dict['d'] = d
            temp_dict['q'] = q

            result_list.append(temp_dict)
            if verbose: print('{}: ({}, {}, {})'.format(i, p, d, q))
        else:
            if verbose: print('{}: failed'.format(i))


    output_df = pd.DataFrame(result_list)
    #results_df['Return_Low'] = results_df['10yr_95CI_Low'] / results_df['curr_price'] - 1
    #results_df['Return_Mid'] = results_df['10yr_Forecast'] / results_df['curr_price'] - 1
    #results_df['Return_High'] = results_df['10yr_95CI_High'] / results_df['curr_price'] - 1
    output_df.set_index('zip', inplace=True)
    col_order = ['p', 'd', 'q', 'aic', 'bic', 'curr_price', 
                      '1yr_95CI_Low', '1yr_Forecast', '1yr_95CI_High',
                      '10yr_95CI_Low', '10yr_Forecast', '10yr_95CI_High']
    output_df = output_df[col_order]
    return output_df




# In[7]:


def agg_model(df, field_name, baseline_trend, ARIMA_order=(2,2,3), train_pct=0.8, figsize=(12,10), bottom=None):
    result_list = []

    df_temp = df.dropna().copy()[field_name]


    # split for pseudo forecasting


    train_num = int(np.round(len(df_temp) * train_pct, 0))

    train = df_temp[:train_num]
    test = df_temp[train_num:]

    # Baseline model assuming constant percent increase in price
    baseline_start = train[-1]
    baseline = test.copy()
    baseline[0] = baseline_start * (1+baseline_trend)**(1/12)
    for i in range(1, len(baseline)):
        baseline[i] = baseline[i-1] * (1+baseline_trend)**(1/12)


    # Define and fit ARIMA model

    model = ARIMA(train, order=ARIMA_order, freq='MS')
    results_AR = model.fit(disp=-1)



    plt.figure(figsize=figsize)
    plt.title('Comparison of Baseline and ARIMA Model on Aggregate Data')
    plt.plot(train, label='Training Actual Price')
    plt.xlabel('Date')
    plt.ylabel('Mean Home Value ($)')
    y_pred = pd.Series(results_AR.forecast(steps=len(test))[0], index=test.index)
    plt.plot(test, label='Testing Actual Price')
    plt.plot(y_pred, color='red', label='ARIMA Predicted Price')
    plt.plot(baseline, label='Baseline Predicted Price')
    plt.legend()
    if bottom: plt.ylim(bottom=bottom) 
    plt.show()

    print('-'*77)
    print('Baseline Model Metrics on Test Data')
    print('='*77)
    report_metrics(test.squeeze(), baseline.squeeze())

    print('-'*77)
    print('ARIMA Model Metrics on Test Data')
    print('='*77)
    report_metrics(test.squeeze(), y_pred.squeeze())


# In[8]:


def change_in_changes(df, field_name, figsize=(12,8), sharex=True, hspace=0.25):
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=sharex)

    fig.subplots_adjust(hspace=hspace)
        
    axes[0].plot(df[field_name])
    axes[0].set_title('Mean Price')
    axes[0].set_ylabel('Mean Home Value ($)')
    axes[1].plot(df[field_name].diff())
    axes[1].set_title('Change in Mean Price')
    axes[1].set_ylabel(r'$\Delta$ Mean Home Value')
    axes[2].plot(df[field_name].diff().diff())
    axes[2].set_title('Change in Change in Mean Price')
    axes[2].set_ylabel(r'$\Delta$ $\Delta$ Mean Home Value');


# In[13]:


def format_results(df, sort_by_field=None):
    df_out = df.copy()
    div_by_curr_list  = ['1yr_95CI_Low', '1yr_95CI_High', '1yr_Forecast',
                     '10yr_95CI_Low', '10yr_95CI_High', '10yr_Forecast']

    for col in div_by_curr_list:
        df_out[col] = df_out[col] / df_out['curr_price'] - 1
        
    geom_return = {'10yr_95CI_High':10, '10yr_95CI_Low':10, '10yr_Forecast':10}

    for col, yrs in geom_return.items():
        df_out[col] = (df_out[col] + 1)**(1/yrs) - 1
        
        
    col_order = ['curr_price', '10yr_95CI_Low', '10yr_Forecast', '10yr_95CI_High', 
                               '1yr_95CI_Low' , '1yr_Forecast' , '1yr_95CI_High']

    df_out = df_out[col_order]
    if sort_by_field: df_out = df_out.sort_values(sort_by_field, ascending=False)
    return df_out


# In[ ]:

def sample_plots_by_scn(df, num_graphs, num_per_row, fig_width=16, hspace=0.6):
    """Print a sample of the data by Parking location, identified with the field SystemCodeNumber
    Parameters:
    num_graphs: Number of locations to make graphs for, ordered by appearance in the dataset.
    
    num_per_row: Number of columns in subplot.
    
    fig_width: Used to adjust the width of the subplots figure.  (default=16)
    
    hspace: Used to adjust whitespace between each row of subplots. (default=0.6)"""
    num_rows = int(np.ceil(num_graphs/num_per_row))
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_per_row, figsize=(fig_width, num_rows * fig_width/4))
    fig.subplots_adjust(hspace=hspace)
    plt.xticks(rotation=45)
    for i, scn in enumerate(df.SystemCodeNumber.unique()[:num_graphs]):
        temp_df = df[df.SystemCodeNumber==scn]
        ax = axes[i//num_per_row, i%num_per_row]
        ax.plot(temp_df.LastUpdated, temp_df.PercentOccupied)
        ax.set_title('Parking Area: {}'.format(scn))
        ax.set_xlabel('Date')
        ax.set_ylabel('Percent Occupied')
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1));
        
    for ax in fig.axes:
        plt.sca(ax)
        plt.xticks(rotation=45)


