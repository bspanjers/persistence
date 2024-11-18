#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 13:45:12 2024

@author: admin
"""
from mpl_toolkits.basemap import Basemap
from matplotlib.colors import Normalize
import os
import numpy as np
from QAR import *
import pandas as pd
import matplotlib.pyplot as plt


def read_climate_data(file_path):
    # Initialize variables
    station_name = None
    starting_date = None
    
    # Open the file for reading
    with open(file_path, 'r') as file:
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
    with open(file_name, 'r') as file:
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

    

def plot_combined(df_results_list, sSeason, l_sType=['pers_', 'pers_', 'pers_'], tau_list=[0.05, .5, .95]):
    fig, axs = plt.subplots(3, 3, figsize=(15, 17), dpi=100, sharey=True, sharex=False)
    fig.subplots_adjust(hspace=0.1, wspace=0.01)
    
    scatter_plots = []

    # First loop for the first row (NAO-)
    for i, (df_results, sType, tau) in enumerate(zip(df_results_list[:3], l_sType, tau_list)):
        ax = axs[0, i]
        
        # Create a Basemap of Europe
        m = Basemap(projection='merc', llcrnrlat=30, urcrnrlat=75, llcrnrlon=-20, urcrnrlon=59, resolution='c', ax=ax)

        # Draw coastlines and countries
        m.drawcoastlines()
        m.drawcountries()

        # Convert latitudes and longitudes to x, y coordinates
        latitude = df_results['latitude'].apply(lambda x: float(x.split(':')[0]) + float(x.split(':')[1])/60 + float(x.split(':')[2])/3600)
        longitude = df_results['longitude'].apply(lambda x: float(x.split(':')[0]) + float(x.split(':')[1])/60 + float(x.split(':')[2])/3600)
        x, y = m(list(longitude), list(latitude))

        # Define colormap and normalization
        cmap = plt.cm.RdYlGn_r if sType == 'pers_' else plt.cm.RdBu_r
        vmin, vmax = (-0.151, 0.151) if sType == 'pers_' else (-15, 15)
        norm = Normalize(vmin=vmin, vmax=vmax)

        # Plot scattered dots
        if sType == 'pers_':
            sc = m.scatter(x, y, c=df_results[f'mean_diff_pers_{sSeason}'], cmap=cmap, norm=norm, s=100, marker='o', alpha=0.7)
        else: 
            sc = m.scatter(x, y, c=df_results[f'mean_diff_{sSeason}'], cmap=cmap, norm=norm, s=100, marker='o', alpha=0.7)

        scatter_plots.append(sc)

        # Add title with subplot numbering
        ax.set_title(f'({chr(97 + i)})' + ' $\\overline{\\Delta}_{\\phi}(\\tau)$ with' + f' $\\tau$ = {tau}', fontsize=12)

        # Add lat and long labels without gridlines
        meridians = np.arange(-20, 61, 20)
        #m.drawmeridians(meridians, labels=[0, 0, 0, 1], fontsize=8, linewidth=0)  # labels on the bottom
        if i == 0:
            parallels = np.arange(30, 81, 10)
            m.drawparallels(parallels, labels=[1, 0, 0, 0], fontsize=8, linewidth=0)  # labels on the left

    # Second loop for the second and third rows (NAO+)
    for i, (df_results, sType, tau) in enumerate(zip(df_results_list[3:], l_sType, tau_list)):
        for j, pm in enumerate(['0', '1']):
            ax = axs[j+1, i]
            
            # Create a Basemap of Europe
            m = Basemap(projection='merc', llcrnrlat=30, urcrnrlat=75, llcrnrlon=-20, urcrnrlon=59, resolution='c', ax=ax)

            # Draw coastlines and countries
            m.drawcoastlines()
            m.drawcountries()

            # Convert latitudes and longitudes to x, y coordinates
            latitude = df_results['latitude'].apply(lambda x: float(x.split(':')[0]) + float(x.split(':')[1])/60 + float(x.split(':')[2])/3600)
            longitude = df_results['longitude'].apply(lambda x: float(x.split(':')[0]) + float(x.split(':')[1])/60 + float(x.split(':')[2])/3600)
            x, y = m(list(longitude), list(latitude))

            # Define colormap and normalization
            cmap = plt.cm.RdYlGn_r if sType == 'pers_' else plt.cm.RdBu_r
            vmin, vmax = (-0.151, 0.151) if sType == 'pers_' else (-15, 15)
            norm = Normalize(vmin=vmin, vmax=vmax)

            # Plot scattered dots
            if sType == 'pers_':
                sc = m.scatter(x, y, c=df_results[f'mean_diff_pers_{sSeason}_{pm}'], cmap=cmap, norm=norm, s=100, marker='o', alpha=0.7)
            else: 
                sc = m.scatter(x, y, c=df_results[f'mean_diff_{sSeason}_{pm}'], cmap=cmap, norm=norm, s=100, marker='o', alpha=0.7)

            scatter_plots.append(sc)

            # Add title with subplot numbering
            sign = '+' if pm==str(1) else '-'
            ax.set_title(f'({chr(97 + 3 + 3*j + i)})' + ' $\\overline{\\Delta}_{\\psi_s}(\\tau)$ with $s=$NAO' + sign + f' and $\\tau$ = {tau}', fontsize=12)

            # Add lat and long labels without gridlines
            if j == 1:
                meridians = np.arange(-20, 61, 20)
                m.drawmeridians(meridians, labels=[0, 0, 0, 1], fontsize=8, linewidth=0)  # labels on the bottom
            if i == 0:
                parallels = np.arange(30, 81, 10)
                m.drawparallels(parallels, labels=[1, 0, 0, 0], fontsize=8, linewidth=0)  # labels on the left

    # Create a single colorbar horizontally at the bottom of the plots
    cbar_ax = fig.add_axes([0.2, 0.075, 0.6, 0.02])  # [left, bottom, width, height]
    cbar = fig.colorbar(scatter_plots[0], cax=cbar_ax, orientation='horizontal')
    plt.show()
    

#for replication purposes, here one can obtain the data for fig 3.


start_date = 1950
start_year_old = start_date
end_year_old = start_date + 30
start_year_new = 1990
end_year_new = start_year_new + 30
tau = .95
drop_na_larger_than = 0.05

folder_path = '/Users/admin/Downloads/ECA_blend_tg/'
lendata = len(np.sort(os.listdir(folder_path))[:-4])
lat_long = pd.DataFrame(np.zeros((lendata, 5)))
lat_long[:] = np.nan
for (i, file_name) in enumerate(np.sort(os.listdir(folder_path))[1:-4]):
    station_name, starting_date = read_climate_data(folder_path + file_name)
    city_name, latitude, longitude = map_station_with_city(station_name, folder_path + 'stations.txt')
    if type(starting_date) != type(None):
        if int(starting_date)<=start_date:
                lat_long.iloc[i,:] = [file_name, station_name, latitude, longitude, city_name]

df = lat_long.dropna()  
df.columns =  ['file_name', 'STAID', 'latitude', 'longitude', 'city_name']
df_results = pd.DataFrame(np.zeros((len(df), 8)), columns=['STANAME', 'STAID', 'latitude', 'longitude', 
                                                            'mean_diff_winter', 'mean_diff_spring', 'mean_diff_summer', 'mean_diff_autumn'])
df_results[:] = np.nan
datetime_index_2019 = pd.date_range(start='2019-01-01', end='2019-12-31', freq='D')

for (i, file_name) in enumerate(df.file_name):
    print(f'\rCurrently calculating station {i+1} out of {len(df.file_name)}', end='')
    test = QAR_temperature(sFile=file_name, dropna=drop_na_larger_than, fTau=tau, 
                       oldend = str(end_year_old) + '-', oldstart=str(start_year_old) + '-', 
                       newend = str(end_year_new) + '-', newstart= str(start_year_new) + '-',
                       include_nao=True, split_nao=False, iLeafs=1)
    if test.iLeafs >= 2:
        season_list_pers = ['mean_diff_pers_winter_', 'mean_diff_pers_spring_', 'mean_diff_pers_summer_', 'mean_diff_pers_autumn_']
    else: 
        season_list_pers = ['mean_diff_pers_winter', 'mean_diff_pers_spring', 'mean_diff_pers_summer', 'mean_diff_pers_autumn']
    season_list_mean = ['mean_diff_winter_', 'mean_diff_spring_', 'mean_diff_summer_', 'mean_diff_autumn_']
    try: 
        if test.iLeafs >= 2:    
            test.plot_paths_with_nao(2019, conf_intervals=True, plot=False)   
        else: 
            test.results()
        for leaf in range(test.iLeafs):
            #differences in persistence for NAO+
            diff_pers = test.mCurves_new - test.mCurves_old
            diff_pers.index = datetime_index_2019
            mean_diff_pers_winter = diff_pers.loc[diff_pers.index.month.isin([12, 1, 2])].mean()
            mean_diff_pers_spring = diff_pers.loc[diff_pers.index.month.isin([3, 4, 5])].mean()
            mean_diff_pers_summer = diff_pers.loc[diff_pers.index.month.isin([6, 7, 8])].mean()
            mean_diff_pers_autumn = diff_pers.loc[diff_pers.index.month.isin([9, 10, 11])].mean()
            if test.iLeafs >= 2:
                df_results.loc[i, [season_list_pers[i] + str(leaf) for i in range(len(season_list_pers))]] = mean_diff_pers_winter.values[leaf], mean_diff_pers_spring.values[leaf], mean_diff_pers_summer.values[leaf], mean_diff_pers_autumn.values[leaf]
            else: 
                df_results.loc[i, [season_list_pers[i] for i in range(len(season_list_pers))]] = mean_diff_pers_winter, mean_diff_pers_spring, mean_diff_pers_summer, mean_diff_pers_autumn

        #mean differences in temperature per season
        mean_diff_winter = test.new.loc[test.new.index.month.isin([12, 1, 2])].mean() - test.old.loc[test.old.index.month.isin([12,1,2])].mean()
        mean_diff_spring = test.new.loc[test.new.index.month.isin([3, 4, 5])].mean() - test.old.loc[test.old.index.month.isin([3, 4, 5])].mean()
        mean_diff_summer = test.new.loc[test.new.index.month.isin([6, 7, 8])].mean() - test.old.loc[test.old.index.month.isin([6, 7, 8])].mean()
        mean_diff_autumn = test.new.loc[test.new.index.month.isin([9, 10, 11])].mean() - test.old.loc[test.old.index.month.isin([9, 10, 11])].mean()
        df_results.iloc[i, :8] = df.city_name.iloc[i], df.STAID.iloc[i], df.latitude.iloc[i], df.longitude.iloc[i], mean_diff_winter.values[0], mean_diff_spring.values[0], mean_diff_summer.values[0], mean_diff_autumn.values[0]
    except ValueError: 
        pass 


#df_results = df_results.dropna().set_index('STANAME')
#df_results = df_results.drop(['IZANA','ELAT', 'ELAT-1', 'STA. CRUZ DE TENERIFE', 'TENERIFE/LOS RODEOS'],axis=0)
#df_results.to_csv('/Users/admin/Documents/PhD/persistence/data_persistence/results_' + str(tau)[-2:] + '_' + str(start_date) + '.csv')
    

