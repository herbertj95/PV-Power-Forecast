# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 16:57:15 2023
Residencial PV Power Production Forecast 
@author: Herbert Amezquita
"""
###############################################################################################################################
'Libraries'
###############################################################################################################################
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
import seaborn as sns
import holidays
import missingno as msno
import warnings
import xgboost as xgb
from xgboost import plot_importance
from sklearn.ensemble import RandomForestRegressor
from scipy import stats
from sklearn import metrics
from sklearn.model_selection import cross_val_score, KFold, TimeSeriesSplit
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, make_scorer
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from matplotlib.dates import DateFormatter, AutoDateLocator
from datetime import datetime
from datetime import timedelta

warnings.filterwarnings('ignore')

###############################################################################################################################
'Plot Parameters'
###############################################################################################################################
plt.rcParams['figure.figsize']=(20,10)
plt.style.use('ggplot')
plt.rcParams.update({'font.size': 20})
pd.set_option('display.max_columns', None)
warnings.filterwarnings("ignore", category=FutureWarning)

###############################################################################################################################
'Functions'
###############################################################################################################################
'Function define_season'
"""
Gives the number of the season based on the month 
    
Args:
    month_number
        
Returns:
    1 - Winter
    2 - Spring
    3 - Summer
    4 - Fall
"""

def define_season(month_number):
    if month_number in [12,1,2]:
        return 1
    
    elif month_number in [3,4,5]:
        return 2
    
    elif month_number in [6,7,8]:
        return 3
    
    elif month_number in [9,10,11]:
        return 4

'Function create_features'
"""
Creates date/time features from a dataframe 
    
Args:
    df - dataframe with a datetime index
        
Returns:
    df - dataframe with 'Weekofyear','Dayofyear','Month','Dayofmonth',
                        'Dayofweek','Weekend','Season','Hour' and 'Minute' features created
"""

def create_features(df):
    
    df['Date'] = df.index
    df['Weekofyear'] = df['Date'].dt.weekofyear   #Value: 1-52
    df['Dayofyear'] = df['Date'].dt.dayofyear    #Value: 1-365
    df['Month'] = df['Date'].dt.month   #Value: 1-12
    df['Dayofmonth'] = df['Date'].dt.day   #Value: 1-30/31
    df['Dayofweek'] = df['Date'].dt.weekday+1     #Value: 1-7 (Monday-Sunday)
    df['Weekend'] = np.where((df['Dayofweek'] == 6) | (df['Dayofweek'] == 7), 1, 0)    #Value: 1 if weekend, 0 if not
    df['Season'] = df.Month.apply(define_season)    #Value 1-4 (winter, spring, summer and fall)    
    df['Hour'] = df['Date'].dt.hour
    df['Hour'] = (df['Hour']+24).where(df['Hour'] == 0, df['Hour'])    #Value: 1-24
    df['Minute'] = df['Date'].dt.minute     #Value: 0, 15, 30 or 45
    df = df.drop(['Date'], axis=1)
    
    return df

'Function lag_features'
"""
Creates lag features for the target variable
    
Args:
    lag_dataset - dataframe 
    days_list - list with the number of days to lag
    var - name of the column to lag (target variable)
        
Returns:
    lag_dataset - dataframe with lag features created
"""

def lag_features(lag_dataset, days_list, var):
    
    temp_data = lag_dataset[var]
    
    for days in days_list:
        rows = 96 * days
        lag_dataset[var+"_lag_{}".format(days)] = temp_data.shift(rows)

    return lag_dataset 

'Function cyclical_features'
"""
Transforms (date/time) features into cyclical sine and cosine features
    
Args:
    df - dataframe with 'Weekofyear','Dayofyear','Season','Month',
                        'Dayofmonth','Dayofweek','Hour','Minute' columns
        
Returns:
    df - dataframe including the cyclical features (x and y for each column)
"""

def cyclical_features(df):

    df['Weekofyear_x']= np.cos(df['Weekofyear']*2*np.pi/52)
    df['Weekofyear_y']= np.sin(df['Weekofyear']*2*np.pi/52)
    df['Dayofyear_x']= np.cos(df['Dayofyear']*2*np.pi/365)
    df['Dayofyear_y']= np.sin(df['Dayofyear']*2*np.pi/365)
    df['Season_x']= np.cos(df['Season']*2*np.pi/4)
    df['Season_y']= np.sin(df['Season']*2*np.pi/4)
    df['Month_x']= np.cos(df['Month']*2*np.pi/12)
    df['Month_y']= np.sin(df['Month']*2*np.pi/12)
    df['Dayofmonth_x']= np.cos(df['Dayofmonth']*2*np.pi/31)
    df['Dayofmonth_y']= np.sin(df['Dayofmonth']*2*np.pi/31)
    df['Dayofweek_x']= np.cos(df['Dayofweek']*2*np.pi/7)
    df['Dayofweek_y']= np.sin(df['Dayofweek']*2*np.pi/7)
    df['Hour_x']= np.cos(df['Hour']*2*np.pi/24)
    df['Hour_y']= np.sin(df['Hour']*2*np.pi/24)
    df['Minute_x']= np.cos(df['Minute']*2*np.pi/45)
    df['Minute_y']= np.sin(df['Minute']*2*np.pi/45)
    
    df= df.drop(columns=['Weekofyear','Dayofyear','Season','Month','Dayofmonth',
                                         'Dayofweek','Hour','Minute'])
    
    return df

'Function compute_errors'
"""
Calculates the metrics to measure the perfomance of the forecasting models
    
Args:
    df - dataframe with 'Prediction' and 'Real' columns
    val - value used to normalized 
        
Returns:
    MAE - mean absolute error
    RMSE - root mean square error
    normRMSE - normalized root mean square error
    R2 - coefficient of determination
"""

def compute_errors(df, val):

    MAE = metrics.mean_absolute_error(df.Real, df.Prediction)
    RMSE = np.sqrt(metrics.mean_squared_error(df.Real, df.Prediction))
    normRMSE = 100 * RMSE / val
    R2 = metrics.r2_score(df.Real, df.Prediction)
    
    return MAE, RMSE, normRMSE, R2

'Function result_plots'
"""
Creates the plots to compare the real values with the predictions obtained
    
Args:
    df - dataframe with 'Prediction' and 'Real' columns
    model_name - string with the name of the model that will appear in the title of the plots
        
Returns:
    Two plots showed
"""

def result_plots(df, model_name):

    # Regression Plot
    sns.scatterplot(data= df, x= 'Real', y= 'Prediction')
    plt.plot(df.Real, df.Real, color = "dodgerblue", linewidth= 1) 
    plt.xlabel("Real Power (W)", alpha= 0.75, weight= "bold")
    plt.ylabel("Predicted Power (W)", alpha= 0.75, weight= "bold")
    plt.xticks(alpha= 0.75, weight= "bold", fontsize= 11)
    plt.yticks(alpha= 0.75, weight= "bold", fontsize= 11)
    plt.title(f"Correlation real vs predicted for {model_name}", alpha= 0.75, weight= "bold", pad= 10, loc= "left")
    plt.show()

    # Real vs predictions in the same plot
    fig, ax = plt.subplots()
    ax.plot(df.Real, label= "Real")
    ax.plot(df.Prediction, label= "Predicted")
    plt.xlabel("Date", alpha= 0.75, weight= "bold")
    plt.ylabel("Power (W)", alpha= 0.75, weight= "bold")
    plt.xticks(alpha= 0.75, weight= "bold", fontsize= 11)
    plt.yticks(alpha= 0.75, weight= "bold", fontsize= 11)
    plt.legend(frameon= False, loc= 'best')
    plt.title(f"Real vs predicted PV power production using {model_name}", alpha= 0.75, weight= "bold", pad= 10, loc= "left")
    locator = AutoDateLocator()
    date_form = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_formatter(date_form)
    fig.autofmt_xdate()
    plt.show()

###############################################################################################################################
'Start of the pipeline'
print(datetime.now())
print('###########################################PV Power Generation Forecast###########################################')

###############################################################################################################################
'Pre-processing and Feature Engineering'
print('########################Pre-Processing and Feature Engineering############################')
###############################################################################################################################

'Reading raw data of PV power production'
data = pd.read_csv('./Data/pv.csv', parse_dates= ['datetime_utc'])
data.drop(['load'], axis=1, inplace= True)
data.rename(columns= {'pv' : 'Power', 'datetime_utc' : 'Date'}, inplace= True)
data.set_index('Date', inplace=True)

#Plot of PV power timeseries
fig,ax= plt.subplots()
plt.plot(data.index, data.Power, color= 'darkcyan')
plt.xlabel("Date", alpha=0.75, weight="bold")
plt.ylabel("Power (W)", alpha=0.75, weight="bold")
plt.xticks(alpha=0.75, weight="bold", rotation= 45)
plt.yticks(alpha=0.75, weight="bold")
plt.title("PV power production every minute", alpha=0.75, weight="bold", loc="left", pad=10)
plt.show()

'Resampling power data from 1 min to 15 min resolution'
power_15min= data.resample('15min').mean()

'Reading the meteorological data'
data_meteo = pd.read_csv('./Data/solcast.csv', parse_dates= ['PeriodStart'])
data_meteo['Date'] = data_meteo['PeriodStart'].dt.tz_localize(None)
data_meteo.drop(['PeriodEnd','PeriodStart','Period','SnowDepth','AlbedoDaily','DewpointTemp','PrecipitableWater',
                 'RelativeHumidity','SurfacePressure','WindDirection10m','WindSpeed10m'], axis=1, inplace=True)
data_meteo.set_index('Date', inplace=True)

'Resampling meteo data from 30 min to 15 min resolution'
meteo_15min= data_meteo.resample('15min').ffill()

'Merging power_15min and meteo_15min dataframes to create the final dataframe'
data_final= pd.merge(power_15min, meteo_15min, left_index=True, right_index=True)
print(data_final.describe())

'Dealing with missing values'

# Number of missing values in each column
print(f'# of missing values in each column \n{data_final.isna().sum()}')

#Dataframe with the missing values of Power
missing_val= data_final[data_final['Power'].isna()]

#Plots of missing values
msno.bar(data_final)
plt.show()

msno.matrix(data_final)
plt.show()

#Delete rows with NaN values
data_final.dropna(axis=0, subset=['Power'], inplace=True)

#Replace NaN with last observed value - ffill
# data_final.fillna(method='ffill', inplace=True)

#Replace NaN with next observed value - bfill
# data_final.fillna(method='bfill', inplace=True)

#Replace NaN using linear interpolation
# data_final.interpolate(limit_direction="both", inplace=True)

'Creating date/time features using the datetime index'
data_final= create_features(data_final)

'Creating lag features of last 15 minutes (1 period before), last hour (4 periods before) and last day (96 periods before)'
data_final= lag_features(data_final,[1,7,30],'Power')
data_final.fillna(0, inplace= True)

'Dealing with outliers (if exist)'

# Converting negative values (if exist) to 0
data_final['Power'] = np.where(data_final.Power < 0, 0, data_final.Power)

# Removing outliers of PV generation before 05:00 am and after 20:00 pm
data_final['Power'] = np.where(data_final.Hour < 6, 0, data_final.Power)
data_final['Power'] = np.where(data_final.Hour > 19, 0, data_final.Power)

'Saving the final dataset'
data_final.to_csv('Final Dataset PV.csv', encoding='utf-8', index=True)
print('csv with the final dataset including date/time and lag features saved in the folder')

###############################################################################################################################
'Plots'
###############################################################################################################################

#PV power production timeseries (after Pre-Processing)
plt.plot(data_final.index, data_final.Power, color= 'darkcyan')
plt.xlabel("Date", alpha= 0.75, weight= "bold")
plt.ylabel("Power (W)", alpha= 0.75, weight= "bold")
plt.xticks(alpha= 0.75, weight= "bold", rotation = 45)
plt.yticks(alpha= 0.75, weight= "bold")
plt.title("PV power production after Pre-Processing", alpha= 0.75, weight= "bold", loc= "left", pad= 10)
plt.show()

#Irradiance
plt.plot(data_final.index, data_final.Ghi, color= 'darkcyan')
plt.xlabel("Date", alpha=0.75, weight="bold")
plt.ylabel("Irradiance (W/m2)", alpha=0.75, weight="bold")
plt.xticks(alpha=0.75, weight="bold")
plt.yticks(alpha=0.75, weight="bold")
plt.title("Global horizontal irradiance", alpha=0.75, weight="bold", loc="left", pad=10)
plt.show()

plt.plot(data_final.index, data_final.Dhi, color= 'darkcyan')
plt.xlabel("Date", alpha=0.75, weight="bold")
plt.ylabel("Irradiance (W/m2)", alpha=0.75, weight="bold")
plt.xticks(alpha=0.75, weight="bold")
plt.yticks(alpha=0.75, weight="bold")
plt.title("Direct normal irradiance", alpha=0.75, weight="bold", loc="left", pad=10)
plt.show()

plt.plot(data_final.index, data_final.Dni, color= 'darkcyan')
plt.xlabel("Date", alpha=0.75, weight="bold")
plt.ylabel("Irradiance (W/m2)", alpha=0.75, weight="bold")
plt.xticks(alpha=0.75, weight="bold")
plt.yticks(alpha=0.75, weight="bold")
plt.title("Diffuse horizontal irradiance", alpha=0.75, weight="bold", loc="left", pad=10)
plt.show()

#Temperature
plt.plot(data_final.index, data_final.AirTemp, color= 'darkcyan')
plt.xlabel("Date", alpha=0.75, weight="bold")
plt.ylabel("Irradiance (W/m2)", alpha=0.75, weight="bold")
plt.xticks(alpha=0.75, weight="bold")
plt.yticks(alpha=0.75, weight="bold")
plt.title("Air temperature", alpha=0.75, weight="bold", loc="left", pad=10)
plt.show()

#Histogram of power
non_zero= data_final[data_final.Power>0]
plt.hist(non_zero.Power, bins=50, color= 'darkcyan')
plt.xticks(alpha=0.75, weight="bold")
plt.yticks(alpha=0.75, weight="bold")
plt.xlabel("Power (W)",alpha=0.75, weight="bold")
plt.ylabel("Count",alpha=0.75, weight="bold")
plt.title("PV power production histrogram", alpha=0.75, weight="bold", loc="left", pad=10)
plt.show()

#Hourly power production scatterplot
sns.scatterplot(data= data_final, x='Hour', y= 'Power')
plt.xticks(range(1,25),alpha=0.75, weight="bold")
plt.yticks(alpha=0.75, weight="bold")
plt.xlabel("Hour",alpha=0.75, weight="bold")
plt.ylabel("Power (W)",alpha=0.75, weight="bold")
plt.title("Hourly PV power production", alpha=0.75, weight="bold", loc="left", pad=10)
plt.show()

#Mean hourly production barplot
mean_per_hour= data_final.groupby('Hour')['Power'].agg(["mean"])
plt.bar(mean_per_hour.index, mean_per_hour["mean"], color= 'darkcyan')
plt.xticks(range(1,25),alpha=0.75, weight="bold")
plt.yticks(alpha=0.75, weight="bold")
plt.xlabel("Hour",alpha=0.75, weight="bold")
plt.ylabel("Power (W)",alpha=0.75, weight="bold")
plt.title("Mean hourly power production", alpha=0.75, weight="bold", loc="left", pad=10)
plt.show()

# Boxplots
sns.boxplot(x= data_final.Hour, y= data_final.Power)
plt.xlabel("Hour", alpha= 0.75, weight= "bold")
plt.ylabel("Power (W)", alpha= 0.75, weight= "bold")
plt.title("Hourly distribution of PV power production", alpha= 0.75, weight= "bold", loc= "left", pad= 10)
plt.show()

sns.boxplot(x= data_final.Dayofweek, y= data_final.Power)
plt.xlabel("Day of Week", alpha= 0.75, weight= "bold")
plt.ylabel("Power (W)", alpha= 0.75, weight= "bold")
plt.xticks([0,1,2,3,4,5,6],['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
plt.xticks(rotation=0)
plt.title("Weekly distribution of PV power production", alpha= 0.75, weight= "bold", loc= "left", pad= 10)
plt.show()

sns.boxplot(x= data_final.Month, y= data_final.Power)
plt.xlabel("Month", alpha= 0.75, weight= "bold")
plt.ylabel("Power (W)", alpha= 0.75, weight= "bold")
plt.title("Monthly distribution of PV power production", alpha= 0.75, weight= "bold", loc= "left", pad= 10)
plt.show()

sns.boxplot(data= data_final, x= 'Season', y= 'Power')
plt.xlabel("Season", alpha= 0.75, weight= "bold")
plt.ylabel("Power (W)", alpha= 0.75, weight= "bold")
plt.xticks([0,1,2,3],['Winter', 'Spring', 'Summer','Fall'])
plt.title("Seasonal distribution of PV power production", alpha= 0.75, weight= "bold", loc= "left", pad= 10)
plt.show() 

###############################################################################################################################
'Clustering'
###############################################################################################################################

'Identifying daily patterns of PV power production'
df_clusters = data_final.copy()
df_clusters = df_clusters.loc[:,['Power', 'Hour']]
df_clusters.reset_index(inplace=True)
df_clusters['Date'] = df_clusters['Date'].dt.date

df_pivot = df_clusters.pivot_table(index= 'Date', columns= 'Hour', values= 'Power')
df_pivot = df_pivot.dropna()
df_pivot.T.plot(legend= False, color= 'dodgerblue', alpha= 0.02)
plt.xlabel('Hour', alpha=0.75, weight="bold")
plt.ylabel('Power (W)', alpha=0.75, weight="bold")
plt.title('Daily PV production profiles', alpha= 0.75, weight= "bold", loc= "left", pad= 10) 
plt.show()

'Defining optimal number of clusters using different methods'
X = df_pivot.values.copy() 
sc = MinMaxScaler()
X = sc.fit_transform(X)

n_cluster_list = np.arange(2,19).astype(int)

sillhoute_scores = []
davies_score = []
distortions = []

for n_cluster in n_cluster_list:
    kmeans = KMeans(n_clusters= n_cluster, init= 'k-means++', max_iter= 100, random_state= 0)
    kmeans.fit_predict(X)
    sillhoute_scores.append(silhouette_score(X, kmeans.labels_))
    davies_score.append(davies_bouldin_score(X,kmeans.labels_))
    distortions.append(kmeans.inertia_)

plt.plot(n_cluster_list, distortions, color= 'dodgerblue')
plt.title('SSE Scores Plot') 
plt.show()

plt.plot(n_cluster_list,sillhoute_scores, color= 'dodgerblue')
plt.title('Silhouete Scores Plot') 
plt.show()

plt.plot(n_cluster_list,davies_score, color= 'dodgerblue')
plt.title('Davies-Bouldin Scores Plot') 
plt.show()

'Number of clusters'
clusters = 3

kmeans = KMeans(n_clusters= clusters, random_state= 0)
cluster_found = kmeans.fit_predict(X)
cluster_found_sr = pd.Series(cluster_found, name= 'Cluster')
df_pivot['Cluster']= cluster_found 

# Plot of the median for each cluster
df_pivot2 = df_pivot.copy()
df_pivot2 = df_pivot2.set_index(cluster_found_sr, append= True )

fig, ax = plt.subplots(1,1)
color_list = ['dodgerblue','darkcyan','red'] 
cluster_values = sorted(df_pivot2.index.get_level_values('Cluster').unique())
for cluster, color in zip(cluster_values, color_list):
    df_pivot2.xs(cluster, level= 1).T.plot(ax= ax, legend= False, alpha= 0.01, color= color)
    df_pivot2.xs(cluster, level= 1).median().plot(ax= ax, color= color, alpha= 0.9, label= cluster)

c_0 = mpatches.Patch(color= 'dodgerblue', label= 'Cluster 0')
c_1 = mpatches.Patch(color= 'darkcyan', label= 'Cluster 1')
c_2 = mpatches.Patch(color= 'red', label= 'Cluster 2')

plt.xlabel("Hour", alpha= 0.75, weight= "bold")
plt.ylabel("Power (W)", alpha= 0.75, weight= "bold")
plt.xticks(alpha= 0.75, weight= "bold", fontsize= 11)
plt.yticks(alpha= 0.75, weight= "bold", fontsize= 11)
plt.legend(handles= [c_0, c_1, c_2])
plt.title('Daily PV power production for ' +str(clusters)+ ' clusters', alpha= 0.75, weight= "bold", loc= "left", pad= 10) 
plt.show()

# Plot of the profiles in each cluster
for cluster_id in sorted(df_pivot.Cluster.unique()):
    df_cluster = df_pivot.loc[df_pivot.Cluster == cluster_id].copy()
    df_cluster.drop(columns= ['Cluster'], inplace= True)
    df_cluster.T.plot(legend= None)
    plt.title(f'Cluster {cluster_id}', alpha= 0.75, weight= "bold", loc= "left", pad= 10)
    plt.xlabel('Hour', alpha= 0.75, weight= "bold")
    plt.ylabel('Power (W)', alpha= 0.75, weight= "bold")
    plt.show()
    
'Including clusters as a column (feature) in the dataframe'
cluster_column = df_pivot.loc[:, ['Cluster']]
cluster_column.index = pd.to_datetime(cluster_column.index)

data_final['Cluster'] = cluster_column
data_final.fillna(method= 'ffill', inplace= True)
data_final = data_final.astype({"Cluster": int})

# Plots of clustering
fig, ax = plt.subplots()
data_final.groupby(['Dayofweek', 'Cluster']).size().unstack().plot(ax=ax, kind='bar', stacked=False, alpha=0.7)
plt.xlabel("Day of Week", alpha= 0.75, weight= "bold")
plt.ylabel("Count", alpha= 0.75, weight= "bold")
plt.xticks([0,1,2,3,4,5,6],['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
plt.xticks(rotation=0)
ax.legend(ncol=3, loc= 'best')
plt.title("Clusters distribution for day of week", alpha= 0.75, weight= "bold", loc= "left", pad= 10)
plt.show()

fig, ax = plt.subplots()
data_final.groupby(['Month', 'Cluster']).size().unstack().plot(ax=ax, kind='bar', stacked=False, alpha=0.7)
plt.xlabel("Month", alpha= 0.75, weight= "bold")
plt.ylabel("Count", alpha= 0.75, weight= "bold")
plt.xticks(rotation=0)
ax.legend(ncol=3, loc= 'best')
plt.title("Clusters distribution for month", alpha= 0.75, weight= "bold", loc= "left", pad= 10)
plt.show()

fig, ax = plt.subplots()
data_final.groupby(['Season', 'Cluster']).size().unstack().plot(ax=ax, kind='bar', stacked=False, alpha=0.7)
plt.xlabel("Season", alpha= 0.75, weight= "bold")
plt.ylabel("Count", alpha= 0.75, weight= "bold")
plt.xticks([0,1,2,3],['Winter', 'Spring', 'Summer','Fall'])
plt.xticks(rotation=0)
ax.legend(ncol=3, loc= 'best')
plt.title("Clusters distribution for season", alpha= 0.75, weight= "bold", loc= "left", pad= 10)
plt.show()

###############################################################################################################################
'Feature engineering/selection'
print('######################################Feature Selection######################################')
###############################################################################################################################

'Transforming date/time features into two dimensional features'
data_final = cyclical_features(data_final)

'Correlation matrix'
corr = data_final.corr()[['Power']].sort_values(by= 'Power', ascending= False).round(2)

# Heatmap features correlation with Power
fig = plt.subplots()
heatmap = sns.heatmap(corr, vmin= -1, vmax= 1, annot= True, cmap= 'BrBG')
heatmap.set_title('Features correlation with PV power production', alpha= 0.75, weight= "bold", pad= 10)
plt.xticks(alpha= 0.75, weight= "bold")
plt.yticks(alpha= 0.75, weight= "bold")
plt.show()

# Bar plot of features correlation with Power
plt.bar(corr[1:].index, corr[1:].Power, color= 'darkcyan')
plt.xticks(rotation= 90, alpha= 0.75, weight= "bold")
plt.yticks(np.arange(-1, 1.1, step= 0.1), alpha=0.75,  weight= "bold")
plt.ylabel('Correlation', alpha= 0.75, weight= "bold")
plt.title("Features correlation with PV power production", alpha= 0.75, weight= "bold", loc= "left", pad= 10)
plt.show()

'Array containing the names of all features available'
all_features = data_final.columns.values.tolist()
all_features.remove('Power')
all_features = np.array(all_features)

'Target variable and features (Y: Power, X: all features)'
df_features = data_final.copy().values
Y = df_features[:, 0] 
X = df_features[:, [x for x in range(1, len(all_features)+1)]]

#Forecast Variable
'Column[0] = Power'

#Features
'Column[1] = AirTemp'
'Column[2] = Azimuth'
'Column[3] = CloudOpacity'
'Column[4] = Dhi'
'Column[5] = Dni'
'Column[6] = Ebh'
'Column[7] = Ghi'
'Column[8] = GtiFixedTilt'
'Column[9] = GtiTracking'
'Column[10] = Zenith'
'Column[11] = Weekend'
'Column[12] = Power_lag_1'
'Column[13] = Power_lag_7'
'Column[14] = Power_lag_30'
'Column[15] = Cluster'
'Column[16] = Weekofyear_x'
'Column[17] = Weekofyear_y'
'Column[18] = Dayofyear_x'
'Column[19] = Dayofyear_y'
'Column[20] = Season_x'
'Column[21] = Season_y'
'Column[22] = Month_x'
'Column[23] = Month_y'
'Column[24] = Dayofmonth_x'
'Column[25] = Dayofmonth_y'
'Column[26] = Dayofweek_x'
'Column[27] = Dayofweek_y'
'Column[28] = Hour_x'
'Column[29] = Hour_y'
'Column[30] = Minute_x'
'Column[31] = Minute_y'

'Filter method - K Best'
features1= SelectKBest(k=15, score_func=f_regression)
features2= SelectKBest(k=15, score_func=mutual_info_regression)

fit1= features1.fit(X,Y)
fit2= features2.fit(X,Y)
filter1= all_features[fit1.get_support()]
filter2= all_features[fit2.get_support()]

print('Best 15 features - Filter method (f_Regression)')
print(filter1)
print('Best 15 features - Filter method (Mutual Information)')
print(filter2)

'Feature Importance by model'
#Random Forest
model2 = RandomForestRegressor()
fit2 = model2.fit(X, Y)
importance2 = pd.DataFrame(data= {'Feature': all_features, 'Score': model2.feature_importances_})
importance2 = importance2.sort_values(by= ['Score'], ascending= False)
importance2.set_index('Feature', inplace= True)

print('Best 15 features - RF')
print(importance2.head(15))

#XGBOOST
model3 = xgb.XGBRegressor()
fit3 = model3.fit(X, Y)
importance3 = pd.DataFrame(data= {'Feature': all_features, 'Score': model3.feature_importances_})
importance3 = importance3.sort_values(by= ['Score'], ascending= False)
importance3.set_index('Feature', inplace= True)

print('Best 15 features - XGBOOST')
print(importance3.head(15))

###############################################################################################################################
'Forecasting Models'
print('######################################Forecasting Models######################################')
###############################################################################################################################

'INPUT the period to analyze (start and end date)'
date_start = '2021-01-10'
date_end = '2021-01-20'
print(f'Period defined: {date_start} to {date_end}')
    
data2 = data_final.loc[date_start:date_end, :].copy()
data2 = data2.resample('H').mean()
data2['Day'] = data2.index.date
data2['Hour'] = data2.index.hour
data2 = data2.loc[:, ['Power', 'Day', 'Hour']]

# Daily PV power profiles for the period defined
data2_pivot = data2.pivot_table(index='Day', values='Power', columns= 'Hour')     
data2_pivot.T.plot()
plt.xlabel("Hour", alpha= 0.75, weight= "bold")
plt.ylabel("Power (W)", alpha= 0.75, weight= "bold")
plt.xticks(alpha= 0.75, weight= "bold")
plt.yticks(alpha= 0.75, weight= "bold")
plt.title("Daily PV power production for the period", alpha= 0.75, weight= "bold", loc= "left", pad= 10)
plt.show()

# Individual profiles
size = len(data2.Day.unique())
fig = plt.figure(figsize= (30, size*5))
fig.subplots_adjust(wspace=0.2, hspace=0.4)
    
for i, day in enumerate(data2.Day.unique()):
    data_temp = data2[data2.Day == day]
    ax = fig.add_subplot(round(size+1/3), 3, i+1)
    sns.lineplot(data= data_temp, x= 'Hour', y='Power')
    plt.xlabel("Hour", alpha= 0.75, weight= "bold")
    plt.ylabel("Power (W)", alpha= 0.75, weight= "bold")
    plt.xticks(alpha= 0.75, weight= "bold")
    plt.yticks(alpha= 0.75, weight= "bold")
    plt.title(day, alpha= 0.75, weight= "bold", loc= "left", pad= 10, fontsize= 20)
    ax.set_ylim(bottom=0, top= data2.Power.max()+500)

plt.show()
     
'INPUTS FOR THE FORECAST'     
# Define the start date of the forecast
start_forecast = datetime.strptime('2021-01-14', '%Y-%m-%d')

# Define the number of days to forecast
days_to_forecast = 1

# Define forecast variable and features
FORECAST_COLUMN = ['Power']

FEATURE_COLUMNS =  ['GtiFixedTilt', 'GtiTracking', 'Ghi', 'Dhi', 'Dni', 'Ebh', 'Zenith', 'Azimuth',
                    'AirTemp', 'CloudOpacity', 'Hour_x', 'Hour_y',
                    'Power_lag_1', 'Power_lag_7', 'Power_lag_30']

print(f'Forecast variable: {FORECAST_COLUMN}')
print(f'Features: {FEATURE_COLUMNS}')

# The start of the training is always the beginning of the dataset
start_training = data_final.index[0]

# The end of the training is always 15min before the start of the forecast
end_training = start_forecast - timedelta(minutes= 15)
        
# The end of the forecast is obtained from the # of days to forecast defined
end_forecast = start_forecast + timedelta(days= days_to_forecast) - timedelta(minutes= 15)

# Print of training and forecast periods
print(f'Data available: {data_final.index[0].date()} to {data_final.index[-1].date()}')
print(f'Days to forecast: {days_to_forecast}')
print(f'Training period: {start_training.date()} to {end_training.date()}')
print(f'Forecast period: {start_forecast.date()} to {end_forecast.date()}')

'Dividing into train and test sets'
data_train = data_final.loc[start_training : end_training, :].copy()
data_test = data_final.loc[start_forecast : end_forecast, :].copy()

xtrain = data_train.loc[:, FEATURE_COLUMNS]
ytrain = data_train.loc[:, FORECAST_COLUMN]

xtest = data_test.loc[:, FEATURE_COLUMNS]
ytest = data_test.loc[:, FORECAST_COLUMN]
        
# Plot train-test split
fig, ax = plt.subplots()
coloring = data_final.Power.max()
plt.plot(data_train.index, data_train["Power"], color= "darkcyan", alpha= 0.75)
plt.fill_between(data_train.index, coloring, facecolor= "darkcyan", alpha= 0.2)
plt.plot(data_test.index, data_test["Power"], color= "dodgerblue", alpha= 0.60)
plt.fill_between(data_test.index, coloring, facecolor= "dodgerblue", alpha= 0.2)
plt.xlabel("Date", alpha= 0.75, weight= "bold")
plt.ylabel("PV Power generation (W)", alpha= 0.75, weight= "bold")
plt.xticks(alpha= 0.75,weight= "bold", fontsize= 11, rotation = 45)
plt.yticks(alpha= 0.75,weight= "bold", fontsize= 11)
plt.legend(['Train','Test'], frameon= False, loc= 'upper center', ncol= 2)
plt.title("Train - Test Split", alpha=0.75, weight="bold", pad=10, loc="left")
plt.show()

###############################################################################################################################
'Implementing Random Forest model'
print('###################################Random Forest####################################')
###############################################################################################################################

# Model
parameters_RF = {'bootstrap': True,
              'min_samples_leaf': 3,
              'n_estimators': 200, 
              'min_samples_split': 7,
              'max_depth': 30,
              'max_leaf_nodes': None,
              'random_state': 18}

reg_RF = RandomForestRegressor(**parameters_RF)
reg_RF.fit(xtrain, np.ravel(ytrain))

# Feature importance
importances = reg_RF.feature_importances_
sorted_index = np.argsort(importances)[::-1]
sorted_index_top = sorted_index[:len(FEATURE_COLUMNS)]
x = range(len(sorted_index_top))

labels = np.array(FEATURE_COLUMNS)[sorted_index_top]
plt.bar(x, importances[sorted_index_top], tick_label= labels)
plt.title("Feature importance RF")
plt.xticks(rotation= 45)
plt.show()

# Predictions and pos-processing
df_RF = pd.DataFrame(reg_RF.predict(xtest), columns= ['Prediction'], index= xtest.index)
df_RF['Real'] = ytest
df_RF['Prediction'] = np.where(df_RF['Prediction'] < 0, 0 , df_RF['Prediction'])
df_RF['Prediction'] = np.where((df_RF.index.hour < 6) | (df_RF.index.hour > 19) , 0, df_RF['Prediction'])

# Errors
MAE_RF, RMSE_RF, normRMSE_RF, R2_RF = compute_errors(df_RF, ytest.Power.max())

print(f'RF - Mean Absolute Error (MAE): {MAE_RF:.2f} W')
print(f'RF - Root Mean Square Error (RMSE): {RMSE_RF:.2f} W')
print(f'RF - Normalized RMSE: {normRMSE_RF:.2f} %')
print(f'RF - R square: {R2_RF:.2f}')

# Plots real vs predicted
result_plots(df_RF, 'Random Forest')

#Saving the predictions
predictions_RF= df_RF[['Prediction']]
predictions_RF.to_csv('./Forecast/Predictions_RF.csv', encoding='utf-8', index=True)
print('csv with the predictions saved')

###############################################################################################################################
'Implementing XGBOOST model'
print('##############################Extreme Gradient Boosting##############################')
###############################################################################################################################

# Model
parameters_XGBOOST = {'n_estimators' : 300,
                      'learning_rate' : 0.01,
                      'verbosity' : 0,
                      'n_jobs' : -1,
                      'gamma' : 0,
                      'min_child_weight' : 1,
                      'max_delta_step' : 0,
                      'subsample' : 0.7,
                      'colsample_bytree' : 1,
                      'colsample_bylevel' : 1,
                      'colsample_bynode' : 1,
                      'reg_alpha' : 0,
                      'reg_lambda' : 1,
                      'random_state' : 18,
                      'objective' : 'reg:linear',
                      'booster' : 'gbtree'}

reg_XGBOOST = xgb.XGBRegressor(**parameters_XGBOOST)
reg_XGBOOST.fit(xtrain, ytrain, 
                eval_set= [(xtrain, ytrain), (xtest, ytest)],
                verbose= 50)

# Feature Importance
plot_importance(reg_XGBOOST)
plt.title("Feature importance XGBOOST")
plt.show()

# Predictions and pos-processing
df_xgboost = pd.DataFrame(reg_XGBOOST.predict(xtest), columns= ['Prediction'], index= xtest.index)
df_xgboost['Real'] = ytest
df_xgboost['Prediction'] = np.where(df_xgboost['Prediction'] < 0, 0 , df_xgboost['Prediction'])
df_xgboost['Prediction'] = np.where((df_xgboost.index.hour < 6) | (df_xgboost.index.hour > 19) , 0, df_xgboost['Prediction'])

# Errors
MAE_XGBOOST, RMSE_XGBOOST, normRMSE_XGBOOST, R2_XGBOOST = compute_errors(df_xgboost, ytest.Power.max())

print(f'XGBOOST - Mean Absolute Error (MAE): {MAE_XGBOOST:.2f} W')
print(f'XGBOOST - Root Mean Square Error (RMSE): {RMSE_XGBOOST:.2f} W')
print(f'XGBOOST - Normalized RMSE: {normRMSE_XGBOOST:.2f} %')
print(f'XGBOOST - R square: {R2_XGBOOST:.2f}')

# Plots real vs predicted
result_plots(df_xgboost, 'XGBOOST')

#Saving the predictions
predictions_XGBOOST= df_xgboost[['Prediction']]
predictions_XGBOOST.to_csv('./Forecast/Predictions_XGBOOST.csv', encoding='utf-8', index=True)
print('csv with the predictions saved')

###############################################################################################################################
'Future Forecasting'
print('##############################Future Forecasting##############################')
###############################################################################################################################

'Dividing all data between features (X) and target variable (y)'
X_all = data_final.loc[:, FEATURE_COLUMNS].copy()
y_all = data_final.loc[:, FORECAST_COLUMN].copy()

'XGBOOST model training with all data'
parameters_XGBOOST = {'n_estimators' : 500,
                      'learning_rate' : 0.01,
                      'verbosity' : 0,
                      'n_jobs' : -1,
                      'gamma' : 0,
                      'min_child_weight' : 1,
                      'max_delta_step' : 0,
                      'subsample' : 0.7,
                      'colsample_bytree' : 1,
                      'colsample_bylevel' : 1,
                      'colsample_bynode' : 1,
                      'reg_alpha' : 0,
                      'reg_lambda' : 1,
                      'random_state' : 18,
                      'objective' : 'reg:linear',
                      'booster' : 'gbtree'}

reg_XGBOOST_future = xgb.XGBRegressor(**parameters_XGBOOST)
reg_XGBOOST_future.fit(X_all, y_all, 
                       eval_set= [(X_all, y_all)],
                       verbose= 100)

'INPUT the number of days to forecast in the future'
days_future = 1

'Creating future dataframe'
df_future = pd.DataFrame(index=pd.date_range(start= data_final.index[-1] + timedelta(minutes = 15), freq= '15min', periods= 96 * days_future))
df_future['isFuture'] = True

power_15min.dropna(axis= 0, subset= ['Power'], inplace= True)
power_15min['isFuture'] = False

data_future = pd.concat([power_15min, df_future])
data_future = create_features(data_future)
data_future = lag_features(data_future,[1,7,30], 'Power')
data_future = cyclical_features(data_future)
data_future = pd.merge(data_future, meteo_15min, left_index= True, right_index= True)

test_future = data_future.query('isFuture').loc[:, FEATURE_COLUMNS]

'Generating the predictions and pos-processing'
xgboost_future = pd.DataFrame(reg_XGBOOST_future.predict(test_future), columns= ['Prediction'], index= test_future.index)
xgboost_future['Prediction'] = np.where(xgboost_future['Prediction'] < 0, 0 , xgboost_future['Prediction'])
xgboost_future['Prediction'] = np.where((xgboost_future.index.hour < 6) | (xgboost_future.index.hour > 19) , 0, xgboost_future['Prediction'])

# Plot of future forecast
fig, ax = plt.subplots()
ax.plot(xgboost_future.Prediction)
plt.xlabel("Date", alpha= 0.75, weight= "bold")
plt.ylabel("Power (W)", alpha= 0.75, weight= "bold")
plt.xticks(alpha= 0.75, weight= "bold", fontsize= 11, rotation = 45)
plt.yticks(alpha= 0.75, weight= "bold", fontsize= 11)
plt.title(f"Future forecast: {days_future} day(s)", alpha= 0.75, weight= "bold", pad= 10, loc= "left")
locator = AutoDateLocator()
date_form = mdates.ConciseDateFormatter(locator)
ax.xaxis.set_major_formatter(date_form)
fig.autofmt_xdate()
plt.show()

# Plot of last past_days (defined) + future forecast
past_days = 7
fig, ax = plt.subplots()
ax.plot(xgboost_future.Prediction)
ax.plot(data_future.Power[-96 * (past_days + days_future) : -96* days_future])
plt.xlabel("Date", alpha= 0.75, weight= "bold")
plt.ylabel("Power (W)", alpha= 0.75, weight= "bold")
plt.xticks(alpha= 0.75, weight= "bold", fontsize= 11)
plt.yticks(alpha= 0.75, weight= "bold", fontsize= 11)
plt.title(f"PV Power production of last {past_days} day(s) + forecast obtained", alpha= 0.75, weight= "bold", pad= 10, loc= "left")
locator = AutoDateLocator()
date_form = mdates.ConciseDateFormatter(locator)
ax.xaxis.set_major_formatter(date_form)
plt.show()

'Saving the future predictions'
xgboost_future.to_csv('./Forecast/Future_Predictions_XGBOOST.csv', encoding='utf-8', index=True)
print('csv with the predictions saved')

###############################################################################################################################
# 'Cross Validation'
# print('##############################Cross Validation##############################')
###############################################################################################################################

# df_cv = data_final.copy()

# parameters_XGBOOST = {'n_estimators' : 1000,
#                       'learning_rate' : 0.01,
#                       'verbosity' : 0,
#                       'n_jobs' : -1,
#                       'gamma' : 0,
#                       'min_child_weight' : 1,
#                       'max_delta_step' : 0,
#                       'subsample' : 0.7,
#                       'colsample_bytree' : 1,
#                       'colsample_bylevel' : 1,
#                       'colsample_bynode' : 1,
#                       'reg_alpha' : 0,
#                       'reg_lambda' : 1,
#                       'random_state' : 18,
#                       'objective' : 'reg:linear',
#                       'booster' : 'gbtree'}

# reg_XGBOOST_cv = xgb.XGBRegressor(**parameters_XGBOOST)

# 'Cross_val_score'
# X_cv = df_cv[FEATURE_COLUMNS]
# y_cv = df_cv[FORECAST_COLUMN]

# cv_results = cross_val_score(reg_XGBOOST_cv, X_cv, y_cv, cv=5, scoring='neg_root_mean_squared_error')

# rmse_mean = -1 * cv_results.mean()
# norm_rmse_mean = 100 * (rmse_mean / ytest.Power.max())

# print(f'Cross_val_mean RMSE: {rmse_mean} W')
# print(f'Cross_val_mean Normalized RMSE: {norm_rmse_mean:.2f} %')

# 'Time Series Split'
# tss = TimeSeriesSplit(n_splits= 5, test_size= 96 * days_to_forecast)

# xgboost_cv = pd.DataFrame()

# for train_idx, val_idx in tss.split(df_cv):
    
#     train_cv = df_cv.iloc[train_idx]
#     test_cv = df_cv.iloc[val_idx]

#     xtrain_cv = train_cv[FEATURE_COLUMNS]
#     ytrain_cv = train_cv[FORECAST_COLUMN]

#     xtest_cv = test_cv[FEATURE_COLUMNS]
#     ytest_cv = test_cv[FORECAST_COLUMN]

#     reg_XGBOOST_cv.fit(xtrain_cv, ytrain_cv, 
#                     eval_set= [(xtrain_cv, ytrain_cv), (xtest_cv, ytest_cv)],
#                     early_stopping_rounds= 50,
#                     verbose= 100)

#     ypred_cv = pd.DataFrame(reg_XGBOOST_cv.predict(xtest_cv), columns= ['Prediction'], index= xtest_cv.index)
#     ypred_cv['Real'] = ytest_cv
#     xgboost_cv = xgboost_cv.append(ypred_cv)

# xgboost_cv['Prediction'] = np.where(xgboost_cv['Prediction'] < 0, 0 , xgboost_cv['Prediction'])
# xgboost_cv['Prediction'] = np.where((xgboost_cv.index.hour < 6) | (xgboost_cv.index.hour > 19) , 0, xgboost_cv['Prediction'])

# # Errors
# MAE_CV, RMSE_CV, normRMSE_CV, R2_CV = compute_errors(xgboost_cv, max(ytest_cv.Power))

# print(f'CV Time Series Split XGBOOST - Mean Absolute Error (MAE): {MAE_CV:.2f} W')
# print(f'CV Time Series Split XGBOOST - Root Mean Square Error (RMSE): {RMSE_CV:.2f} W')
# print(f'CV Time Series Split XGBOOST - Normalized RMSE: {normRMSE_CV:.2f} %')
# print(f'CV Time Series Split XGBOOST - R square: {R2_CV:.2f}')

# # Plots
# result_plots(xgboost_cv, 'CV Time Series Split XGBOOST')

# 'K-Fold'
# kf = KFold(n_splits= 5, shuffle= True, random_state= 18)

# xgboost_cv = pd.DataFrame()

# for train_idx, val_idx in kf.split(df_cv):
    
#     train_cv = df_cv.iloc[train_idx]
#     test_cv = df_cv.iloc[val_idx]

#     xtrain_cv = train_cv[FEATURE_COLUMNS]
#     ytrain_cv = train_cv[FORECAST_COLUMN]

#     xtest_cv = test_cv[FEATURE_COLUMNS]
#     ytest_cv = test_cv[FORECAST_COLUMN]

#     reg_XGBOOST_cv.fit(xtrain_cv, ytrain_cv, 
#                     eval_set= [(xtrain_cv, ytrain_cv), (xtest_cv, ytest_cv)],
#                     early_stopping_rounds= 50,
#                     verbose= 100)

#     ypred_cv = pd.DataFrame(reg_XGBOOST_cv.predict(xtest_cv), columns= ['Prediction'], index= xtest_cv.index)
#     ypred_cv['Real'] = ytest_cv
#     xgboost_cv = xgboost_cv.append(ypred_cv)

# xgboost_cv['Prediction'] = np.where(xgboost_cv['Prediction'] < 0, 0 , xgboost_cv['Prediction'])
# xgboost_cv['Prediction'] = np.where((xgboost_cv.index.hour < 6) | (xgboost_cv.index.hour > 19) , 0, xgboost_cv['Prediction'])

# # Errors
# MAE_CV, RMSE_CV, normRMSE_CV, R2_CV = compute_errors(xgboost_cv, max(ytest_cv.Power))

# print(f'CV K-Fold XGBOOST - Mean Absolute Error (MAE): {MAE_CV:.2f} W')
# print(f'CV K-Fold XGBOOST - Root Mean Square Error (RMSE): {RMSE_CV:.2f} W')
# print(f'CV K-Fold XGBOOST - Normalized RMSE: {normRMSE_CV:.2f} %')
# print(f'CV K-Fold XGBOOST - R square: {R2_CV:.2f}')

###############################################################################################################################
'End of the pipeline'
print(datetime.now())
###############################################################################################################################