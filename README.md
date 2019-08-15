
<a href="https://colab.research.google.com/github/JonSolow/Parking_TimeSeries/blob/master/Parking_TimeSeries.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Predicting Parking Occupancy Rates

Smart cities now have technology that can tell whether parking spots are occupied.  In some cities in the UK, such as Birmingham, nearly every parking spot has one of these detectors.  The detectors are connected to a server that can keep track of overall usage.

An obvious question for an individual driver may be: when and where will parking be available?

In order to predict an answer to this question, we will make a predictive model using Time Series analysis.

Specifically, we will try two models:
+ ARIMA (Autoregressive integrated moving average)
+ SARIMA (Seasonal ARIMA).


# Import Data

## Data Description

The dataset includes parking area occupancy and capacity for various locations in Birmingham, UK.

Each observation includes a date and time, ranging from October 4, 2016 through December 19, 2016 and for times from about 8:00AM to 4:30PM.  The measurements were taken every half hour, but not at exactly 0:00 and 0:30 times.


This dataset was obtained from the UCI Machine Learning Database and was compiled by Daniel H. Stolfi at the University of Malaga - Spain.  The original source of the data is from the Birmingham City Council government website.

https://data.birmingham.gov.uk/dataset/birmingham-parking/resource/bea04cd0-ea86-4d7e-ab3b-2da3368d1e01

https://archive.ics.uci.edu/ml/datasets/Parking+Birmingham#


+ Daniel H. Stolfi, Enrique Alba, and Xin Yao. Predicting Car Park Occupancy Rates in Smart Cities. In: Smart Cities: Second International Conference, Smart-CT 2017, MÃ¡laga, Spain, June 14-16, 2017, pp. 107â€“117. doi> 10.1007/978-3-319-59513-9_11 
+ Birmingham City Council. [Web Link]



## Download Data


```python
# Only needs to be run once to download the data, unless you are working on Google Colab
# uncomment below to download (For UNIX systems)

!curl https://archive.ics.uci.edu/ml/machine-learning-databases/00482/dataset.zip --create-dirs -o  ./data/dataset.zip
!unzip -o ./data/dataset.zip -d ./data/

```

      % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                     Dload  Upload   Total   Spent    Left  Speed
    100  234k  100  234k    0     0   180k      0  0:00:01  0:00:01 --:--:--  180k
    Archive:  ./data/dataset.zip
      inflating: ./data/dataset.csv      


# Import Data to Pandas Dataframe


```python
# To download the py script
!curl https://raw.githubusercontent.com/JonSolow/Parking_TimeSeries/master/Workbook_Init.py >> Workbook_Init.py

# Import libraries and custom functions defined in Workbook_Init.py
from Workbook_Init import *

```

      % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                     Dload  Upload   Total   Spent    Left  Speed
    100 11754  100 11754    0     0  29311      0 --:--:-- --:--:-- --:--:-- 29311



```python
df_raw = pd.read_csv('./data/dataset.csv')
```

## Data Inspection

Using the `.head` method, a sample of the dataset shows that we have the following information:
+ SystemCodeNumber - ID code for the parking area
+ Capacity - The number of parking spots available
+ Occupancy - The number of parkings spots actually occupied at the time
+ LastUpdated - The date and time of the observation


```python
df_raw.head(10)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SystemCodeNumber</th>
      <th>Capacity</th>
      <th>Occupancy</th>
      <th>LastUpdated</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>BHMBCCMKT01</td>
      <td>577</td>
      <td>61</td>
      <td>2016-10-04 07:59:42</td>
    </tr>
    <tr>
      <th>1</th>
      <td>BHMBCCMKT01</td>
      <td>577</td>
      <td>64</td>
      <td>2016-10-04 08:25:42</td>
    </tr>
    <tr>
      <th>2</th>
      <td>BHMBCCMKT01</td>
      <td>577</td>
      <td>80</td>
      <td>2016-10-04 08:59:42</td>
    </tr>
    <tr>
      <th>3</th>
      <td>BHMBCCMKT01</td>
      <td>577</td>
      <td>107</td>
      <td>2016-10-04 09:32:46</td>
    </tr>
    <tr>
      <th>4</th>
      <td>BHMBCCMKT01</td>
      <td>577</td>
      <td>150</td>
      <td>2016-10-04 09:59:48</td>
    </tr>
    <tr>
      <th>5</th>
      <td>BHMBCCMKT01</td>
      <td>577</td>
      <td>177</td>
      <td>2016-10-04 10:26:49</td>
    </tr>
    <tr>
      <th>6</th>
      <td>BHMBCCMKT01</td>
      <td>577</td>
      <td>219</td>
      <td>2016-10-04 10:59:48</td>
    </tr>
    <tr>
      <th>7</th>
      <td>BHMBCCMKT01</td>
      <td>577</td>
      <td>247</td>
      <td>2016-10-04 11:25:47</td>
    </tr>
    <tr>
      <th>8</th>
      <td>BHMBCCMKT01</td>
      <td>577</td>
      <td>259</td>
      <td>2016-10-04 11:59:44</td>
    </tr>
    <tr>
      <th>9</th>
      <td>BHMBCCMKT01</td>
      <td>577</td>
      <td>266</td>
      <td>2016-10-04 12:29:45</td>
    </tr>
  </tbody>
</table>
</div>



Using `len()` function, we can see how many observations there are:


```python
len(df_raw)
```




    35717



It is always a good idea to make sure your data is in the format that you would expect.  In our case, we would want the SystemCodeNumber to be a string (called object in Pandas).  The Capacity and Occupancy should be integers (int64 in Pandas).  And the LastUpdated field should be a datetime (datetime64 in Pandas).

Using the `.dtypes` attribute of the dataframe, we can see what the data type is for each column.

The LastUpdated field formatted as a string instead of a datetime.  This is common when importing dates from csv files with Pandas.  We can fix this in our next step.


```python
df_raw.dtypes
```




    SystemCodeNumber    object
    Capacity             int64
    Occupancy            int64
    LastUpdated         object
    dtype: object



## Data Cleaning

We already know that one thing we need to do is change LastUpdated to be a datetime type.

After doing so, there are a few additional features to add that will help with our analysis:
+ PercentOccupied - Ratio of Occupancy to Capacity.  We will want to make sure these values go from 0%-100%
+ Date - Only the date component of LastUpdated.  Used for checking the data.
+ DayOfWeek - An integer representing the day of the week.  (1 = Monday)
+ Date_Time_HalfHour - this field will round each time to the nearest half hour.  This will help when we aggregate accross all parking areas.
+ Time - Only the time component of the Date_Time_HalfHour.  Used for checking the data.


```python
df_clean = df_raw.copy()
df_clean.LastUpdated = df_clean.LastUpdated.astype('datetime64')
df_clean['PercentOccupied'] = df_clean.Occupancy / df_clean.Capacity
df_clean['date'] = df_clean.LastUpdated.dt.date
df_clean['dayofweek'] = df_clean.LastUpdated.dt.dayofweek
df_clean['date_time_halfhour'] = df_clean.LastUpdated.dt.round('30min')
df_clean['time'] = df_clean.date_time_halfhour.dt.time
```

Again, use `.head()` to inspect a sample of the data and make sure our new fields behave as expected.


```python
df_clean.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SystemCodeNumber</th>
      <th>Capacity</th>
      <th>Occupancy</th>
      <th>LastUpdated</th>
      <th>PercentOccupied</th>
      <th>date</th>
      <th>dayofweek</th>
      <th>date_time_halfhour</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>BHMBCCMKT01</td>
      <td>577</td>
      <td>61</td>
      <td>2016-10-04 07:59:42</td>
      <td>0.105719</td>
      <td>2016-10-04</td>
      <td>1</td>
      <td>2016-10-04 08:00:00</td>
      <td>08:00:00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>BHMBCCMKT01</td>
      <td>577</td>
      <td>64</td>
      <td>2016-10-04 08:25:42</td>
      <td>0.110919</td>
      <td>2016-10-04</td>
      <td>1</td>
      <td>2016-10-04 08:30:00</td>
      <td>08:30:00</td>
    </tr>
    <tr>
      <th>2</th>
      <td>BHMBCCMKT01</td>
      <td>577</td>
      <td>80</td>
      <td>2016-10-04 08:59:42</td>
      <td>0.138648</td>
      <td>2016-10-04</td>
      <td>1</td>
      <td>2016-10-04 09:00:00</td>
      <td>09:00:00</td>
    </tr>
    <tr>
      <th>3</th>
      <td>BHMBCCMKT01</td>
      <td>577</td>
      <td>107</td>
      <td>2016-10-04 09:32:46</td>
      <td>0.185442</td>
      <td>2016-10-04</td>
      <td>1</td>
      <td>2016-10-04 09:30:00</td>
      <td>09:30:00</td>
    </tr>
    <tr>
      <th>4</th>
      <td>BHMBCCMKT01</td>
      <td>577</td>
      <td>150</td>
      <td>2016-10-04 09:59:48</td>
      <td>0.259965</td>
      <td>2016-10-04</td>
      <td>1</td>
      <td>2016-10-04 10:00:00</td>
      <td>10:00:00</td>
    </tr>
  </tbody>
</table>
</div>



First, we will take a look at what times we have.  We expect to have times ranging from 8:00AM to 4:30PM (16:30 military time).

We can see from below taht there are a few times of 7:30AM, which we will discard.


```python
df_clean.groupby('time').size()
```




    time
    07:30:00      30
    08:00:00    2096
    08:30:00    1971
    09:00:00    1953
    09:30:00    1983
    10:00:00    1987
    10:30:00    1985
    11:00:00    1961
    11:30:00    1988
    12:00:00    1976
    12:30:00    1982
    13:00:00    1988
    13:30:00    1958
    14:00:00    1984
    14:30:00    1989
    15:00:00    1986
    15:30:00    1985
    16:00:00    1956
    16:30:00    1959
    dtype: int64




```python
# filter out few 7:30 measurements
df_clean = df_clean[df_clean.time > datetime.time(7,30)]
```

In case there are any duplicates, we will use pandas `.drop_duplicates()` method to automatically remove any. 

It happens that there are 207 duplicates removed.


```python
# drop duplicates
pre_len = len(df_clean)
df_clean = df_clean.drop_duplicates()

post_len = len(df_clean)

dropped_len = pre_len - post_len
print(dropped_len)
```

    207


We also wanted to make sure that the Occupancy ranged from a minimum value of 0 to a maximum value of Capacity.  We can check along all of the observations by seeing whether PercentOccupied is between 0% and 100%.


```python
# Note that some values are out of range of 0-100%
print('Minimum Percent Occupied: {:.2%}'.format(df_clean.PercentOccupied.min()))
print('Maximum Percent Occupied: {:.2%}'.format(df_clean.PercentOccupied.max()))
```

    Minimum Percent Occupied: -1.67%
    Maximum Percent Occupied: 104.13%


There were some values outside the range (probably due to malfunction equipment), so we will manually limit the data to be between 0 and the Capacity value.  After doing so, we do another check to make sure we are now ranging from 0% to 100% in the PercentOccupied field.


```python
# Limit Occupancy to the range of zero to Capacity
df_clean.Occupancy = df_clean.apply(lambda x: max(0, min(x['Capacity'], x['Occupancy'])), axis=1)
df_clean['PercentOccupied'] = df_clean.Occupancy / df_clean.Capacity

# Re-check range
print('Minimum Percent Occupied: {:.2%}'.format(df_clean.PercentOccupied.min()))
print('Maximum Percent Occupied: {:.2%}'.format(df_clean.PercentOccupied.max()))
```

    Minimum Percent Occupied: 0.00%
    Maximum Percent Occupied: 100.00%


Let's take a look at a few graphs for each location to see what the trend in parking occupancy rates looks like.


```python
sample_plots_by_scn(df=df_clean, num_graphs=6, num_per_row=2)
```



![png](images/output_32_1.png)


There seems to be a pattern to the parking occupancy, which makes sense.  People may use certain parking areas more often during lunch hours if there are places to eat nearby.   Most areas will also see higher parking rates during weekdays.


```python
df_agg_dthh = df_clean.groupby('date_time_halfhour').agg({'Occupancy':['sum','count'], 'Capacity':['sum','count']})
df_agg_dthh['PercentOccupied'] = df_agg_dthh.Occupancy['sum'] / df_agg_dthh.Capacity['sum']
```


```python
# Check for times when we dont have a big enough sample
df_agg_dthh[(df_agg_dthh.Occupancy['count']<20)|(df_agg_dthh.Capacity['sum']<25000)]


```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="2" halign="left">Occupancy</th>
      <th colspan="2" halign="left">Capacity</th>
      <th>PercentOccupied</th>
    </tr>
    <tr>
      <th></th>
      <th>sum</th>
      <th>count</th>
      <th>sum</th>
      <th>count</th>
      <th></th>
    </tr>
    <tr>
      <th>date_time_halfhour</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2016-10-28 08:00:00</th>
      <td>10</td>
      <td>1</td>
      <td>450</td>
      <td>1</td>
      <td>0.022222</td>
    </tr>
    <tr>
      <th>2016-12-13 13:30:00</th>
      <td>663</td>
      <td>1</td>
      <td>720</td>
      <td>1</td>
      <td>0.920833</td>
    </tr>
  </tbody>
</table>
</div>



There are two days where only one parking area was reported for a certain time.  We should replace these with more appropriate values to avoid confusing the model.  The next cell will drop these values, so they can be replaced.


```python
df_agg_dthh.drop(columns=['Occupancy', 'Capacity'], inplace=True)
df_agg_dthh.drop([pd.Timestamp('2016-10-28 08:00:00'), pd.Timestamp('2016-12-13 13:30:00')], inplace=True)
```


```python
df_agg_dthh.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th>PercentOccupied</th>
    </tr>
    <tr>
      <th></th>
      <th></th>
    </tr>
    <tr>
      <th>date_time_halfhour</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2016-10-04 08:00:00</th>
      <td>0.201437</td>
    </tr>
    <tr>
      <th>2016-10-04 08:30:00</th>
      <td>0.247512</td>
    </tr>
    <tr>
      <th>2016-10-04 09:00:00</th>
      <td>0.315444</td>
    </tr>
    <tr>
      <th>2016-10-04 09:30:00</th>
      <td>0.382376</td>
    </tr>
    <tr>
      <th>2016-10-04 10:00:00</th>
      <td>0.438917</td>
    </tr>
  </tbody>
</table>
</div>



The temporary dataframe below is used to inspect for more missing times.  The ARIMA and SARIMA models will require having every time filled in.


```python
temp = df_agg_dthh.reset_index()
temp['date'] = temp.date_time_halfhour.dt.date
temp['time'] = temp.date_time_halfhour.dt.time
temp = temp.groupby('date').count()
temp = pd.DataFrame(temp, index=pd.date_range('2016-10-04', '2016-12-19')).fillna(0)
temp[temp.date_time_halfhour<18]

# All of 10/20 and 10/21 are missing
# 10/30 missing 16:00 and 16:30
# 11/18 missing 9:00
# 11/25 missing 8:30
# 12/14 missing 11:00

# 10/28 and 12/13 dropped times as noted above
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th>date_time_halfhour</th>
      <th>PercentOccupied</th>
      <th>time</th>
    </tr>
    <tr>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2016-10-20</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2016-10-21</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2016-10-28</th>
      <td>17.0</td>
      <td>17.0</td>
      <td>17.0</td>
    </tr>
    <tr>
      <th>2016-10-30</th>
      <td>16.0</td>
      <td>16.0</td>
      <td>16.0</td>
    </tr>
    <tr>
      <th>2016-11-18</th>
      <td>17.0</td>
      <td>17.0</td>
      <td>17.0</td>
    </tr>
    <tr>
      <th>2016-11-25</th>
      <td>17.0</td>
      <td>17.0</td>
      <td>17.0</td>
    </tr>
    <tr>
      <th>2016-12-03</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2016-12-04</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2016-12-13</th>
      <td>17.0</td>
      <td>17.0</td>
      <td>17.0</td>
    </tr>
    <tr>
      <th>2016-12-14</th>
      <td>17.0</td>
      <td>17.0</td>
      <td>17.0</td>
    </tr>
  </tbody>
</table>
</div>



This function fills in the time with the occupancy at the same time from one week earlier, based on our knowledge that people tend to do things on a weekly basis.


```python
def fill_with_week_prior(df, column, year, month, day, hour, minutes):
  df.loc[pd.to_datetime(datetime.datetime(year, month, day, hour, minutes)), column] = \
      df.loc[pd.to_datetime(datetime.datetime(year, month, day, hour, minutes) + timedelta(days=-7)), column].values[0]

```


```python
# fill in missing Percent Occupied with prior week's value for same time

# Also fill in for the under-reported times noted above

df_agg_fillmissing = df_agg_dthh.copy()


# all day loop
for hour in range(8, 17):
  for half_hour in [0, 30]:
    fill_with_week_prior(df_agg_fillmissing, 'PercentOccupied', 2016, 10, 20, hour, half_hour)
    fill_with_week_prior(df_agg_fillmissing, 'PercentOccupied', 2016, 10, 21, hour, half_hour)
    fill_with_week_prior(df_agg_fillmissing, 'PercentOccupied', 2016, 12, 3, hour, half_hour)
    fill_with_week_prior(df_agg_fillmissing, 'PercentOccupied', 2016, 12, 4, hour, half_hour)

# fill_with_week_prior(df_agg_fillmissing, 'PercentOccupied', 2016, 10, 20, 8, 0)
# fill_with_week_prior(df_agg_fillmissing, 'PercentOccupied', 2016, 10, 21, 8, 0)
fill_with_week_prior(df_agg_fillmissing, 'PercentOccupied', 2016, 10, 30, 16, 0)
fill_with_week_prior(df_agg_fillmissing, 'PercentOccupied', 2016, 10, 30, 16, 30)
fill_with_week_prior(df_agg_fillmissing, 'PercentOccupied', 2016, 11, 18, 9, 0)
fill_with_week_prior(df_agg_fillmissing, 'PercentOccupied', 2016, 11, 25, 8, 30)
fill_with_week_prior(df_agg_fillmissing, 'PercentOccupied', 2016, 12, 14, 11, 0)
fill_with_week_prior(df_agg_fillmissing, 'PercentOccupied', 2016, 10, 28, 8, 0)
fill_with_week_prior(df_agg_fillmissing, 'PercentOccupied', 2016, 12, 13, 13, 30)

df_agg_fillmissing.sort_index(inplace=True)
```

Now we check to make sure that there are 18 times for each day.  The following code should return an empty series.


```python

temp = pd.Series(df_agg_fillmissing.index.date).value_counts()
temp[temp<18]
```




    Series([], dtype: int64)



Let's take another look to see that we don't have any more gaps in the data.


```python
plt.figure(figsize=(18,6))
plt.plot(df_agg_fillmissing)
plt.plot(df_agg_fillmissing.shift(18))
plt.xlabel('Date', fontsize=14);
```


![png](images/output_47_0.png)


# Look for Seasonality and Test for Stationarity

The function below graphs a plot of autocorrelation and partial correlation.  These graphs plot the correlation between the occupancy and that same measure X periods before (ACF is the single lag value, PACF is the moving average).

From the ACF graph, we can see the pattern of the movement through the day.  Each day has 18 periods for each half hour from (8:00am - 4:30pm).  The ACF peaks at 18, signifying that the period repeats daily.  We can see another peak at 36.

We can see from the PACF that the each time is positively correlated with the previous half hour.  We can see that the pattern restarts negatively at the 19th period, which is one entire day later.


```python
subplots_acf_pacf(df_agg_fillmissing)
```


![png](images/output_50_0.png)


One necessity for time series is to have stationarity.  This means that the mean and standard deviation in the endogenous variable are stable through time.

In order to achieve stationarity, we must remove the patterns in the data.

To start with, we know that the pattern repeats every 18 periods, so we use that as the period in the function below.

The Dickey-Fuller test is used to test for stationarity.  As shown below, the stationarity condition is met when using 18 periods, but we can probably do better.


```python
test_stationarity(df_agg_fillmissing.squeeze(), 18)
```


![png](images/output_52_0.png)


    Results of Dickey-Fuller Test:
    Test Statistic                   -4.028499
    p-value                           0.001268
    #Lags Used                       24.000000
    Number of Observations Used    1361.000000
    Critical Value (1%)              -3.435164
    Critical Value (5%)              -2.863666
    Critical Value (10%)             -2.567902
    dtype: float64


To try to make the data more stationary, the difference below aim to make the endogenous variable the change in occupancy for each time between each day.  It then uses a period of 7 to compare each of those differences each week.

The resulting data is very stationary.  The p-value (which is essentially the probability of seeing this stationarity by random chance) is incredibly low.


```python
test_stationarity(df_agg_dthh.diff(18).diff().dropna().squeeze(), 7)
```


![png](images/output_54_0.png)


    Results of Dickey-Fuller Test:
    Test Statistic                -1.407532e+01
    p-value                        2.891728e-26
    #Lags Used                     1.900000e+01
    Number of Observations Used    1.268000e+03
    Critical Value (1%)           -3.435518e+00
    Critical Value (5%)           -2.863822e+00
    Critical Value (10%)          -2.567985e+00
    dtype: float64


# Split data to train and test the Models

The train-test split used in this case is a TimeSeriesSplit, which uses a cutoff date.  All data prior to that date is training, and everything after is testing.


```python
# Train-Test Split
# Sklearn built in split for time series

from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
print(tscv)


data_use = df_agg_fillmissing.reset_index()['PercentOccupied']

for train_index, test_index in tscv.split(data_use):
  train = data_use[train_index]
  test = data_use[test_index]

```

    TimeSeriesSplit(max_train_size=None, n_splits=5)


We can check that the time split worked properly by making sure that the end of the training data lines up with the beginning of the test data.


```python
train.index = df_agg_fillmissing.index[:len(train)]
train.tail()
```




    date_time_halfhour
    2016-12-06 16:00:00    0.505548
    2016-12-06 16:30:00    0.458811
    2016-12-07 08:00:00    0.212089
    2016-12-07 08:30:00    0.256110
    2016-12-07 09:00:00    0.310685
    Name: PercentOccupied, dtype: float64




```python
test.index = df_agg_fillmissing.index[len(train):]
test.head()
```




    date_time_halfhour
    2016-12-07 09:30:00    0.368854
    2016-12-07 10:00:00    0.438171
    2016-12-07 10:30:00    0.494762
    2016-12-07 11:00:00    0.557329
    2016-12-07 11:30:00    0.589452
    Name: PercentOccupied, dtype: float64



# ARIMA Model

Based on our ACF and PACF, we will first try a baseline ARIMA model that uses 18 periods and a first order difference.  The ARIMA model with a period of 18 takes a while to run (about 6 minutes on google colab).


```python
%%time
# Define and fit ARIMA model
arima_model = ARIMA(train, order=(18, 1, 0))
results_AR = arima_model.fit(disp=-1)
```



    CPU times: user 4min 23s, sys: 1min 34s, total: 5min 57s
    Wall time: 3min 2s


Let's take a look at how the forecasted occupancy rates (purple) compare to the actual rates (orange).  The rates in blue are those that were used to train the model.

The function also shows several metrics on how well the function performed.
* Explained Variance - the amount of variance in the occupancy rates explained by the model (~77%)
* Mean Absolute Error (MAE) - the average percent error in the predictions (~5.5%)



```python
plt.figure(figsize=(16,6))
plt.title('ARIMA Model on Aggregate Data')
plt.plot(train, label='Training Actual Occupancy Rate')
plt.xlabel('Date')
plt.ylabel('Percent Occupied')
y_pred_AR = pd.Series(results_AR.forecast(steps=len(test))[0], index=test.index)
plt.plot(test, label='Testing Actual Occupancy Rate')
plt.plot(y_pred_AR, color='purple', label='ARIMA Predicted Occupancy Rate')
plt.legend()

plt.show()


print('-'*77)
print('ARIMA Model Metrics on Test Data')
print('='*77)
report_metrics(test.squeeze(), y_pred_AR.squeeze())
```


![png](images/output_65_0.png)


    -----------------------------------------------------------------------------
    ARIMA Model Metrics on Test Data
    =============================================================================
    Explained Variance:
    	 0.7690813638663216
    MAE:
    	 0.05530672421729186


# SARIMA Model

The model below reflects the seasonality of the occupancy rates.  The difference here is that a seasonal order is applied such that the 18 period-day is reflected.

This model is able to run much more quickly (about 30 seconds on google colab).


```python
%%time
# Define and fit SARIMA model
my_seasonal_order = (1, 1, 1, 18)
sarima_model = SARIMAX(train, order=(1, 0, 1), seasonal_order=my_seasonal_order)
results_SAR = sarima_model.fit(disp=-1)
```


    CPU times: user 21 s, sys: 13.3 s, total: 34.3 s
    Wall time: 17.5 s


We can see that the model seems to fit the actual data much more closely.  The metrics also show this:
* Explained Variance - increased to 85.8%
* Mean Absolute Error - decreased to 4.9%


```python
plt.figure(figsize=(16,6))
plt.title('SARIMA Model on Aggregate Data')
plt.plot(train, label='Training Actual Occupancy Rate')
plt.xlabel('Date')
plt.ylabel('Percent Occupied')
y_pred_sar = pd.Series(results_SAR.forecast(steps=len(test)).values, index=test.index)
plt.plot(test, label='Testing Actual Occupancy Rate')
plt.plot(y_pred_sar, color='red', label='SARIMA Predicted Occupancy Rate')
plt.legend()

plt.show()


print('-'*77)
print('SARIMA Model Metrics on Test Data')
print('='*77)
report_metrics(test.squeeze(), y_pred_sar.squeeze())
```




![png](images/output_69_1.png)


    -----------------------------------------------------------------------------
    SARIMA Model Metrics on Test Data
    =============================================================================
    Explained Variance:
    	 0.8579497936600348
    MAE:
    	 0.04914439301573438


# Model Comparison

We can take a look at how the actual predicted occupancy rates compare in the pandas dataframe below.


```python
df_SAR_results = pd.DataFrame(list(zip(test.index, y_pred_sar, test, y_pred_sar-test)), columns=['Date_Time', 'Predicted', 'Actual', 'Difference'])
df_SAR_results['Absolute_Diff'] = np.abs(df_SAR_results.Difference)
df_SAR_results.sort_values('Absolute_Diff', ascending=False)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date_Time</th>
      <th>Predicted</th>
      <th>Actual</th>
      <th>Difference</th>
      <th>Absolute_Diff</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>74</th>
      <td>2016-12-11 10:30:00</td>
      <td>0.440147</td>
      <td>0.266308</td>
      <td>0.173839</td>
      <td>0.173839</td>
    </tr>
    <tr>
      <th>194</th>
      <td>2016-12-17 16:30:00</td>
      <td>0.475947</td>
      <td>0.647990</td>
      <td>-0.172043</td>
      <td>0.172043</td>
    </tr>
    <tr>
      <th>73</th>
      <td>2016-12-11 10:00:00</td>
      <td>0.388501</td>
      <td>0.219690</td>
      <td>0.168811</td>
      <td>0.168811</td>
    </tr>
    <tr>
      <th>227</th>
      <td>2016-12-19 15:00:00</td>
      <td>0.580498</td>
      <td>0.746696</td>
      <td>-0.166198</td>
      <td>0.166198</td>
    </tr>
    <tr>
      <th>228</th>
      <td>2016-12-19 15:30:00</td>
      <td>0.554546</td>
      <td>0.718909</td>
      <td>-0.164363</td>
      <td>0.164363</td>
    </tr>
    <tr>
      <th>226</th>
      <td>2016-12-19 14:30:00</td>
      <td>0.602365</td>
      <td>0.762891</td>
      <td>-0.160527</td>
      <td>0.160527</td>
    </tr>
    <tr>
      <th>225</th>
      <td>2016-12-19 14:00:00</td>
      <td>0.611695</td>
      <td>0.761781</td>
      <td>-0.150086</td>
      <td>0.150086</td>
    </tr>
    <tr>
      <th>229</th>
      <td>2016-12-19 16:00:00</td>
      <td>0.519877</td>
      <td>0.667912</td>
      <td>-0.148036</td>
      <td>0.148036</td>
    </tr>
    <tr>
      <th>193</th>
      <td>2016-12-17 16:00:00</td>
      <td>0.519877</td>
      <td>0.667002</td>
      <td>-0.147125</td>
      <td>0.147125</td>
    </tr>
    <tr>
      <th>230</th>
      <td>2016-12-19 16:30:00</td>
      <td>0.475947</td>
      <td>0.619624</td>
      <td>-0.143677</td>
      <td>0.143677</td>
    </tr>
    <tr>
      <th>224</th>
      <td>2016-12-19 13:30:00</td>
      <td>0.608571</td>
      <td>0.752058</td>
      <td>-0.143488</td>
      <td>0.143488</td>
    </tr>
    <tr>
      <th>72</th>
      <td>2016-12-11 09:30:00</td>
      <td>0.331861</td>
      <td>0.191522</td>
      <td>0.140339</td>
      <td>0.140339</td>
    </tr>
    <tr>
      <th>199</th>
      <td>2016-12-18 10:00:00</td>
      <td>0.388188</td>
      <td>0.249156</td>
      <td>0.139032</td>
      <td>0.139032</td>
    </tr>
    <tr>
      <th>223</th>
      <td>2016-12-19 13:00:00</td>
      <td>0.601115</td>
      <td>0.737948</td>
      <td>-0.136833</td>
      <td>0.136833</td>
    </tr>
    <tr>
      <th>200</th>
      <td>2016-12-18 10:30:00</td>
      <td>0.439857</td>
      <td>0.304442</td>
      <td>0.135415</td>
      <td>0.135415</td>
    </tr>
    <tr>
      <th>75</th>
      <td>2016-12-11 11:00:00</td>
      <td>0.492604</td>
      <td>0.360669</td>
      <td>0.131935</td>
      <td>0.131935</td>
    </tr>
    <tr>
      <th>222</th>
      <td>2016-12-19 12:30:00</td>
      <td>0.583665</td>
      <td>0.708266</td>
      <td>-0.124601</td>
      <td>0.124601</td>
    </tr>
    <tr>
      <th>68</th>
      <td>2016-12-10 16:30:00</td>
      <td>0.476187</td>
      <td>0.600156</td>
      <td>-0.123969</td>
      <td>0.123969</td>
    </tr>
    <tr>
      <th>192</th>
      <td>2016-12-17 15:30:00</td>
      <td>0.554546</td>
      <td>0.671999</td>
      <td>-0.117453</td>
      <td>0.117453</td>
    </tr>
    <tr>
      <th>221</th>
      <td>2016-12-19 12:00:00</td>
      <td>0.562555</td>
      <td>0.678285</td>
      <td>-0.115730</td>
      <td>0.115730</td>
    </tr>
    <tr>
      <th>198</th>
      <td>2016-12-18 09:30:00</td>
      <td>0.331516</td>
      <td>0.226707</td>
      <td>0.104809</td>
      <td>0.104809</td>
    </tr>
    <tr>
      <th>220</th>
      <td>2016-12-19 11:30:00</td>
      <td>0.530302</td>
      <td>0.634790</td>
      <td>-0.104488</td>
      <td>0.104488</td>
    </tr>
    <tr>
      <th>201</th>
      <td>2016-12-18 11:00:00</td>
      <td>0.492314</td>
      <td>0.388268</td>
      <td>0.104045</td>
      <td>0.104045</td>
    </tr>
    <tr>
      <th>219</th>
      <td>2016-12-19 11:00:00</td>
      <td>0.492314</td>
      <td>0.594735</td>
      <td>-0.102421</td>
      <td>0.102421</td>
    </tr>
    <tr>
      <th>67</th>
      <td>2016-12-10 16:00:00</td>
      <td>0.520154</td>
      <td>0.619808</td>
      <td>-0.099654</td>
      <td>0.099654</td>
    </tr>
    <tr>
      <th>71</th>
      <td>2016-12-11 09:00:00</td>
      <td>0.282400</td>
      <td>0.182882</td>
      <td>0.099518</td>
      <td>0.099518</td>
    </tr>
    <tr>
      <th>191</th>
      <td>2016-12-17 15:00:00</td>
      <td>0.580498</td>
      <td>0.674796</td>
      <td>-0.094298</td>
      <td>0.094298</td>
    </tr>
    <tr>
      <th>218</th>
      <td>2016-12-19 10:30:00</td>
      <td>0.439857</td>
      <td>0.533474</td>
      <td>-0.093617</td>
      <td>0.093617</td>
    </tr>
    <tr>
      <th>76</th>
      <td>2016-12-11 11:30:00</td>
      <td>0.530566</td>
      <td>0.440561</td>
      <td>0.090005</td>
      <td>0.090005</td>
    </tr>
    <tr>
      <th>147</th>
      <td>2016-12-15 11:00:00</td>
      <td>0.492315</td>
      <td>0.577734</td>
      <td>-0.085418</td>
      <td>0.085418</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>120</th>
      <td>2016-12-13 15:30:00</td>
      <td>0.554554</td>
      <td>0.565863</td>
      <td>-0.011309</td>
      <td>0.011309</td>
    </tr>
    <tr>
      <th>176</th>
      <td>2016-12-16 16:30:00</td>
      <td>0.475947</td>
      <td>0.487233</td>
      <td>-0.011286</td>
      <td>0.011286</td>
    </tr>
    <tr>
      <th>102</th>
      <td>2016-12-12 15:30:00</td>
      <td>0.554575</td>
      <td>0.564366</td>
      <td>-0.009792</td>
      <td>0.009792</td>
    </tr>
    <tr>
      <th>16</th>
      <td>2016-12-08 08:30:00</td>
      <td>0.241196</td>
      <td>0.250975</td>
      <td>-0.009779</td>
      <td>0.009779</td>
    </tr>
    <tr>
      <th>62</th>
      <td>2016-12-10 13:30:00</td>
      <td>0.609210</td>
      <td>0.600240</td>
      <td>0.008970</td>
      <td>0.008970</td>
    </tr>
    <tr>
      <th>48</th>
      <td>2016-12-09 15:30:00</td>
      <td>0.555503</td>
      <td>0.564348</td>
      <td>-0.008846</td>
      <td>0.008846</td>
    </tr>
    <tr>
      <th>122</th>
      <td>2016-12-13 16:30:00</td>
      <td>0.475953</td>
      <td>0.484370</td>
      <td>-0.008417</td>
      <td>0.008417</td>
    </tr>
    <tr>
      <th>181</th>
      <td>2016-12-17 10:00:00</td>
      <td>0.388189</td>
      <td>0.396306</td>
      <td>-0.008118</td>
      <td>0.008118</td>
    </tr>
    <tr>
      <th>208</th>
      <td>2016-12-18 14:30:00</td>
      <td>0.602365</td>
      <td>0.610449</td>
      <td>-0.008084</td>
      <td>0.008084</td>
    </tr>
    <tr>
      <th>207</th>
      <td>2016-12-18 14:00:00</td>
      <td>0.611695</td>
      <td>0.604252</td>
      <td>0.007443</td>
      <td>0.007443</td>
    </tr>
    <tr>
      <th>178</th>
      <td>2016-12-17 08:30:00</td>
      <td>0.230015</td>
      <td>0.237154</td>
      <td>-0.007139</td>
      <td>0.007139</td>
    </tr>
    <tr>
      <th>180</th>
      <td>2016-12-17 09:30:00</td>
      <td>0.331516</td>
      <td>0.338464</td>
      <td>-0.006948</td>
      <td>0.006948</td>
    </tr>
    <tr>
      <th>167</th>
      <td>2016-12-16 12:00:00</td>
      <td>0.562556</td>
      <td>0.569485</td>
      <td>-0.006930</td>
      <td>0.006930</td>
    </tr>
    <tr>
      <th>81</th>
      <td>2016-12-11 14:00:00</td>
      <td>0.611860</td>
      <td>0.605535</td>
      <td>0.006325</td>
      <td>0.006325</td>
    </tr>
    <tr>
      <th>182</th>
      <td>2016-12-17 10:30:00</td>
      <td>0.439857</td>
      <td>0.445926</td>
      <td>-0.006069</td>
      <td>0.006069</td>
    </tr>
    <tr>
      <th>173</th>
      <td>2016-12-16 15:00:00</td>
      <td>0.580498</td>
      <td>0.574450</td>
      <td>0.006048</td>
      <td>0.006048</td>
    </tr>
    <tr>
      <th>15</th>
      <td>2016-12-08 08:00:00</td>
      <td>0.202960</td>
      <td>0.208799</td>
      <td>-0.005839</td>
      <td>0.005839</td>
    </tr>
    <tr>
      <th>103</th>
      <td>2016-12-12 16:00:00</td>
      <td>0.519902</td>
      <td>0.525577</td>
      <td>-0.005675</td>
      <td>0.005675</td>
    </tr>
    <tr>
      <th>0</th>
      <td>2016-12-07 09:30:00</td>
      <td>0.363275</td>
      <td>0.368854</td>
      <td>-0.005579</td>
      <td>0.005579</td>
    </tr>
    <tr>
      <th>187</th>
      <td>2016-12-17 13:00:00</td>
      <td>0.601115</td>
      <td>0.605432</td>
      <td>-0.004317</td>
      <td>0.004317</td>
    </tr>
    <tr>
      <th>121</th>
      <td>2016-12-13 16:00:00</td>
      <td>0.519884</td>
      <td>0.522875</td>
      <td>-0.002990</td>
      <td>0.002990</td>
    </tr>
    <tr>
      <th>49</th>
      <td>2016-12-09 16:00:00</td>
      <td>0.520690</td>
      <td>0.523670</td>
      <td>-0.002980</td>
      <td>0.002980</td>
    </tr>
    <tr>
      <th>179</th>
      <td>2016-12-17 09:00:00</td>
      <td>0.282048</td>
      <td>0.284807</td>
      <td>-0.002759</td>
      <td>0.002759</td>
    </tr>
    <tr>
      <th>50</th>
      <td>2016-12-09 16:30:00</td>
      <td>0.476635</td>
      <td>0.479228</td>
      <td>-0.002592</td>
      <td>0.002592</td>
    </tr>
    <tr>
      <th>174</th>
      <td>2016-12-16 15:30:00</td>
      <td>0.554546</td>
      <td>0.552082</td>
      <td>0.002464</td>
      <td>0.002464</td>
    </tr>
    <tr>
      <th>104</th>
      <td>2016-12-12 16:30:00</td>
      <td>0.475969</td>
      <td>0.477758</td>
      <td>-0.001788</td>
      <td>0.001788</td>
    </tr>
    <tr>
      <th>63</th>
      <td>2016-12-10 14:00:00</td>
      <td>0.612260</td>
      <td>0.613918</td>
      <td>-0.001658</td>
      <td>0.001658</td>
    </tr>
    <tr>
      <th>87</th>
      <td>2016-12-12 08:00:00</td>
      <td>0.193281</td>
      <td>0.192811</td>
      <td>0.000470</td>
      <td>0.000470</td>
    </tr>
    <tr>
      <th>175</th>
      <td>2016-12-16 16:00:00</td>
      <td>0.519877</td>
      <td>0.520088</td>
      <td>-0.000211</td>
      <td>0.000211</td>
    </tr>
    <tr>
      <th>168</th>
      <td>2016-12-16 12:30:00</td>
      <td>0.583665</td>
      <td>0.583848</td>
      <td>-0.000183</td>
      <td>0.000183</td>
    </tr>
  </tbody>
</table>
<p>231 rows × 5 columns</p>
</div>



This plot zooms in on the testing period and has the predictions from both models.  We can see that the SARIMA model fits much more closely to reality.


```python
plt.figure(figsize=(16,6))
plt.title('Comparison of ARIMA and SARIMA Models on Testing Data')
plt.xlabel('Date')
plt.ylabel('Percent Occupied')
y_pred_sar = pd.Series(results_SAR.forecast(steps=len(test)).values, index=test.index)
plt.plot(test, label='Testing Actual Occupancy Rate', color='orange')
plt.plot(y_pred_sar, color='red', label='SARIMA Occupancy Rate')

y_pred_AR = pd.Series(results_AR.forecast(steps=len(test))[0], index=test.index)
plt.plot(y_pred_AR, color='purple', label='ARIMA Occupancy Rate')

plt.legend()

plt.show()


```




![png](images/output_74_1.png)

