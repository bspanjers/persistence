#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 13:45:12 2024

@author: admin
"""
from mpl_toolkits.basemap import Basemap
from scipy.interpolate import griddata
from matplotlib.colors import TwoSlopeNorm
from sklearn.preprocessing import StandardScaler
import os
import numpy as np
import statsmodels.api as sm
import pandas as pd
import matplotlib.pyplot as plt
from QAR_persistence_precip import QAR_precipitation
from statsmodels.tools.sm_exceptions import PerfectSeparationError
import copy


def read_climate_data(file_path):
    # Initialize variables
    station_name = None
    starting_date = None
    
    # Open the file for reading
    with open(file_path, 'r', encoding='ISO-8859-1') as file:
        # Read the file line by line
        for line in file:
            line = line.strip()  # Remove leading and trailing whitespace
            if line.startswith("This is the blended series of station"):
                station_info = line.split("(")
                if len(station_info) > 1:
                    station_name = station_info[1].split(")")[0].split(",")[-1].strip()[7:]
                break  # Exit the loop after finding station info

        # Open the file again for reading
        # Skip the header lines
        for line in file:
            if line.startswith("STAID"):
                break
        
        # Read the first observation to get the starting date
        for line in file:
            line = line.strip()
            if line:
                data = line.split(",")
                starting_date = data[2][:4]  # Extract the year part of the date
                break  # Exit the loop after finding starting date

    return station_name, starting_date

def map_station_with_city(station_name, file_name):
    # Open the file for reading
    with open(file_name, 'r', encoding='ISO-8859-1') as file:
        # Skip the header lines
        next(file)
        next(file)

        # Read the file line by line
        for line in file:
            line = line.strip()  # Remove leading and trailing whitespace
            if line:
                # Extract station information
                station_data = line.split(",")
                if len(station_data) >= 5:  # Ensure there are enough elements in the list
                    current_station_name = station_data[0].strip()
                    if current_station_name == station_name:
                        city_name = station_data[1].strip()
                        latitude = station_data[3].strip()
                        longitude = station_data[4].strip()
                        return city_name, latitude, longitude

    return None, None, None  # Return None if station not found


# Define function to assign seasons
def get_season(month):
    if month in [12, 1, 2]:
        return 'winter'
    elif month in [3, 4, 5]:
        return 'spring'
    elif month in [6, 7, 8]:
        return 'summer'
    elif month in [9, 10, 11]:
        return 'autumn'

# Create lag features for each season, including shifting nao_index_cdas and creating dummies
def create_lagged_features(data, n_lags):
    lagged_data = data.copy()
    
    # Create lagged features for 'Temp'
    for lag in range(1, n_lags + 1):
        lagged_data[f'lag_{lag}'] = lagged_data['Temp'].shift(lag)
    
    # Create categorical indicators for 'nao_index_cdas'
    lagged_data['nao_index_cdas'] = lagged_data['nao_index_cdas'].shift(1)  # Shift nao_index_cdas
    quantiles = [0, 0.5, 0.8, 1.0]  # Define quantiles for categorical conversion
    lagged_data['nao_index_cdas_cat'] = pd.qcut(lagged_data['nao_index_cdas'], quantiles, labels=False)
    
    # Convert categorical indicator into dummy variables
    nao_index_dummies = pd.get_dummies(lagged_data['nao_index_cdas_cat'], prefix='nao_index_cat', drop_first=True)
    lagged_data = pd.concat([lagged_data, nao_index_dummies], axis=1)
    
    # Drop original categorical indicator column and rows with NaN values due to lagging and shifting
    lagged_data = lagged_data.drop(['nao_index_cdas', 'nao_index_cdas_cat'], axis=1).dropna()
    
    return lagged_data

# Function to fit AR logistic regression for all winter months
def fit_ar_logistic_regression(data, n_lags, months=[12, 1, 2]):
    winter_data = data[data.index.month.isin(months)].copy()
    seasonal_data = create_lagged_features(winter_data, n_lags)

    # Prepare predictors (X) and response variable (y)
    X = seasonal_data[[f'lag_{lag}' for lag in range(1, n_lags + 1)] + list(seasonal_data.filter(like='nao_index_cat').columns)].values
    y = seasonal_data['Temp'].values

    # Add a constant term for the intercept
    X = sm.add_constant(X)
    
    # Standardize the predictors (optional but recommended)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X[:, 1:])  # Exclude the constant column
    X_scaled = np.column_stack((np.ones(X_scaled.shape[0]), X_scaled))  # Add back the constant column
    
    # Fit the autoregressive logistic regression model
    model = sm.Logit(y, X)
    result = model.fit(disp=0)

    return result


def plot_heatmap_precip(df_results, sSeason, sType='_unc'):
    
    try:
        df_results.set_index('STANAME', inplace=True)
        df_results = df_results.drop(['IZANA','ELAT', 'ELAT-1', 'STA. CRUZ DE TENERIFE', 'TENERIFE/LOS RODEOS'],axis=0)
        df_results.reset_index(inplace=True)
    except KeyError:
        pass
    
    # Create a Basemap of Europe
    plt.figure(figsize=(12, 5), dpi=200)
    m = Basemap(projection='merc', llcrnrlat=30, urcrnrlat=75, llcrnrlon=-20, urcrnrlon=60, resolution='l')
    
    # Draw coastlines and countries
    m.drawcoastlines()
    m.drawcountries()
    
    # Convert latitudes and longitudes to x, y coordinates
    latitude = df_results['latitude'].apply(lambda x: float(x.split(':')[0]) + float(x.split(':')[1])/60 + float(x.split(':')[2])/3600)
    longitude = df_results['longitude'].apply(lambda x: float(x.split(':')[0]) + float(x.split(':')[1])/60 + float(x.split(':')[2])/3600)
    x, y = m(list(longitude), list(latitude))
    
    # Define colormap and normalization
    cmap = plt.cm.RdYlGn_r
    if sType != '_unc':
        vmin, vcenter, vmax = -.5, 0, .5
    else: 
        vmin, vcenter, vmax = -.151, 0, .151

    norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
    
    # Plot heatmap
    if sType != '_unc':
        sc = m.scatter(x, y, c=df_results['mean_diff_' + sSeason], cmap=cmap, norm=norm, s=30, marker='o', alpha=1)
    else: 
        sc = m.scatter(x, y, c=df_results['mean_diff_' + sSeason + '_unc'], cmap=cmap, norm=norm, s=30, marker='o', alpha=1)

    # Add colorbar with shrink factor to make it smaller
    cbar = plt.colorbar(sc, shrink=0.85)  # Use shrink to make the colorbar smaller
        
    parallels = np.arange(30, 81, 10)
    m.drawparallels(parallels, labels=[1,0,0,0], fontsize=8, linewidth=0)  # labels on the left
    meridians = np.arange(-20, 60, 20)
    m.drawmeridians(meridians, labels=[0,0,0,1], fontsize=8, linewidth=0)  # labels on the bottom

    # Set colorbar label
    if sType != '_unc':
        cbar.set_label('$\Delta_{\\gamma_5}$')
    else: 
        cbar.set_label('$\Delta_{Q_5}$')
    # Show the plot
    #plt.savefig(f'./plots_persistence/probability_of_rain_without_{loyo_year}_for_{setting_save}.png')  # You can also specify other formats like 'pdf', 'jpg', etc.

    plt.show()

start_date = 1950
start_year_old = start_date
end_year_old = start_date + 30
start_year_new = 1990
end_year_new = start_year_new + 30
tau = .5
drop_na_larger_than = 0.05

folder_path = '/Users/admin/Downloads/ECA_blend_rr/'
lendata = len(np.sort(os.listdir(folder_path))[:-4])
lat_long = pd.DataFrame(np.zeros((lendata, 5)))
lat_long[:] = np.nan
for (i, file_name) in enumerate(np.sort(os.listdir(folder_path))[1:-4]):
    station_name, starting_date = read_climate_data(folder_path + file_name)
    city_name, latitude, longitude = map_station_with_city(station_name, folder_path + 'stations.txt')
    if type(starting_date) != type(None):
        if int(starting_date) <= start_date:
                lat_long.iloc[i,:] = [file_name, station_name, latitude, longitude, city_name]

df = lat_long.dropna()  
df.columns =  ['file_name', 'STAID', 'latitude', 'longitude', 'city_name']
df_results = pd.DataFrame(np.zeros((len(df), 12)), columns=['STANAME', 'STAID', 'latitude', 'longitude', 
                                                            'mean_diff_winter', 'mean_diff_spring', 'mean_diff_summer', 'mean_diff_autumn',
                                                            'mean_diff_winter_unc', 'mean_diff_spring_unc', 'mean_diff_summer_unc', 'mean_diff_autumn_unc'])
df_results[:] = np.nan
for (i, file_name) in enumerate(df.file_name):
    print(f'\rCurrently calculating station {i+1} out of {len(df.file_name)}', end='')
    
    try:
        test = QAR_precipitation(sFile=file_name, dropna=drop_na_larger_than,
                       oldend = str(end_year_old) + '-', oldstart=str(start_year_old) + '-', 
                       newend = str(end_year_new) + '-', newstart=str(start_year_new) + '-', include_nao=True
                      )
        season_list_unc = ['mean_diff_winter_unc', 'mean_diff_spring_unc', 'mean_diff_summer_unc', 'mean_diff_autumn_unc']
        season_list_mean = ['mean_diff_winter_', 'mean_diff_spring_', 'mean_diff_summer_', 'mean_diff_autumn_']
        test.prepare_data()  
        # Generate example binary time series data for test.old
        y_prec_old = (test.old.Temp >= 5) * 1
        data_old = pd.DataFrame(y_prec_old, columns=['Temp'])
        data_old['nao_index_cdas'] = test.old.nao_index_cdas
        
        # Generate example binary time series data for test.new
        y_prec_new = (test.new.Temp >= 5) * 1
        data_new = pd.DataFrame(y_prec_new, columns=['Temp'])
        data_new['nao_index_cdas'] = test.new.nao_index_cdas
        
        # Assign season to each row
        data_old['season'] = data_old.index.month.map(get_season).values
        data_new['season'] = data_new.index.month.map(get_season).values
        
        
        data_winter_new, data_winter_old = data_new.loc[data_new.season == 'winter'],  data_old.loc[data_old.season == 'winter']
        data_spring_new, data_spring_old = data_new.loc[data_new.season == 'spring'],  data_old.loc[data_old.season == 'spring']
        data_summer_new, data_summer_old = data_new.loc[data_new.season == 'summer'],  data_old.loc[data_old.season=='summer']
        data_autumn_new, data_autumn_old = data_new.loc[data_new.season == 'autumn'],  data_old.loc[data_old.season=='autumn']

                
        p_rain_cond_nao_new_winter = data_winter_new.loc[data_winter_new.nao_index_cdas.shift(1) > np.quantile(data_winter_new.nao_index_cdas, .8)].mean().Temp
        p_rain_cond_nao_old_winter = data_winter_old.loc[data_winter_old.nao_index_cdas.shift(1) > np.quantile(data_winter_old.nao_index_cdas, .8)].mean().Temp
        p_rain_cond_nao_new_spring = data_spring_new.loc[data_spring_new.nao_index_cdas.shift(1) > np.quantile(data_spring_new.nao_index_cdas, .8)].mean().Temp
        p_rain_cond_nao_old_spring = data_spring_old.loc[data_spring_old.nao_index_cdas.shift(1) > np.quantile(data_spring_old.nao_index_cdas, .8)].mean().Temp
        p_rain_cond_nao_new_summer = data_summer_new.loc[data_summer_new.nao_index_cdas.shift(1) > np.quantile(data_summer_new.nao_index_cdas, .8)].mean().Temp
        p_rain_cond_nao_old_summer = data_summer_old.loc[data_summer_old.nao_index_cdas.shift(1) > np.quantile(data_summer_old.nao_index_cdas, .8)].mean().Temp
        p_rain_cond_nao_new_autumn = data_autumn_new.loc[data_autumn_new.nao_index_cdas.shift(1) > np.quantile(data_autumn_new.nao_index_cdas, .8)].mean().Temp
        p_rain_cond_nao_old_autumn = data_autumn_old.loc[data_autumn_old.nao_index_cdas.shift(1) > np.quantile(data_autumn_old.nao_index_cdas, .8)].mean().Temp

        diff_winter_unc = p_rain_cond_nao_new_winter - p_rain_cond_nao_old_winter
        diff_spring_unc = p_rain_cond_nao_new_spring - p_rain_cond_nao_old_spring
        diff_summer_unc = p_rain_cond_nao_new_summer - p_rain_cond_nao_old_summer
        diff_autumn_unc = p_rain_cond_nao_new_autumn - p_rain_cond_nao_old_autumn
        
        # Number of lags
        n_lags = 1
        
        # Fit models for both datasets
        seasonal_results_old_params_winter = fit_ar_logistic_regression(data_old, n_lags, months=[12, 1, 2]).params[-1]
        seasonal_results_new_params_winter = fit_ar_logistic_regression(data_new, n_lags, months=[12, 1, 2]).params[-1]
        seasonal_results_old_params_spring = fit_ar_logistic_regression(data_old, n_lags, months=[3, 4, 5]).params[-1]
        seasonal_results_new_params_spring = fit_ar_logistic_regression(data_new, n_lags, months=[3, 4, 5]).params[-1]
        seasonal_results_old_params_summer = fit_ar_logistic_regression(data_old, n_lags, months=[6, 7, 8]).params[-1]
        seasonal_results_new_params_summer = fit_ar_logistic_regression(data_new, n_lags, months=[6, 7, 8]).params[-1]
        seasonal_results_old_params_autumn = fit_ar_logistic_regression(data_old, n_lags, months=[9, 10, 11]).params[-1]
        seasonal_results_new_params_autumn = fit_ar_logistic_regression(data_new, n_lags, months=[9, 10, 11]).params[-1]
        
        diff_winter_gamma5 = seasonal_results_new_params_winter - seasonal_results_old_params_winter
        diff_spring_gamma5 = seasonal_results_new_params_spring - seasonal_results_old_params_spring
        diff_summer_gamma5 = seasonal_results_new_params_summer - seasonal_results_old_params_summer
        diff_autumn_gamma5 = seasonal_results_new_params_autumn - seasonal_results_old_params_autumn
        
        df_results.iloc[i, :] = [df.city_name.iloc[i], df.STAID.iloc[i], df.latitude.iloc[i], df.longitude.iloc[i], \
                                 diff_winter_gamma5, diff_spring_gamma5, diff_summer_gamma5, diff_autumn_gamma5, \
                                 diff_winter_unc, diff_spring_unc, diff_summer_unc, diff_autumn_unc]
    except (ValueError, np.linalg.LinAlgError, PerfectSeparationError) as e: 
        pass 

df_results = df_results.dropna().set_index('STANAME')
#df_results = df_results.drop(['IZANA','ELAT', 'ELAT-1', 'STA. CRUZ DE TENERIFE', 'TENERIFE/LOS RODEOS'],axis=0)
df_results.to_csv('/Users/admin/Documents/PhD/persistence/data_persistence/results_precipitation_' + str(start_date) + 'WithUncProbabilities.csv')

plot_heatmap_precip(df_results, 'winter', sType='_unc')
plot_heatmap_precip(df_results, 'winter', sType='')
