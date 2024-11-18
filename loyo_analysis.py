#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 09:36:28 2024

@author: admin
"""

from mpl_toolkits.basemap import Basemap
from scipy.interpolate import griddata
from matplotlib.colors import TwoSlopeNorm, Normalize
import os
import numpy as np
from QAR import *
import warnings
from pandas.errors import SettingWithCopyWarning
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from plots_europe import read_climate_data, map_station_with_city


######## LOYO ANALYSIS PRECIP #############


# Suppress specific warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)

# Initialize variables
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

# Load station metadata and populate lat_long DataFrame
for (i, file_name) in enumerate(np.sort(os.listdir(folder_path))[1:-4]):
    station_name, starting_date = read_climate_data(folder_path + file_name)
    city_name, latitude, longitude = map_station_with_city(station_name, folder_path + 'stations.txt')
    if type(starting_date) != type(None):
        if int(starting_date) <= start_date:
            lat_long.iloc[i, :] = [file_name, station_name, latitude, longitude, city_name]

df = lat_long.dropna()  
df.columns = ['file_name', 'STAID', 'latitude', 'longitude', 'city_name']

# Initialize DataFrame to store results
datetime_index_2019 = pd.date_range(start='2019-01-01', end='2019-12-31', freq='D')


season_list_pers = ['mean_diff_pers_winter', 'mean_diff_pers_spring', 'mean_diff_pers_summer', 'mean_diff_pers_autumn']

# Perform LOYO: Leave-One-Year-Out cross-validation for each year in the new dataset (1990–2020)
for exclude_year in tqdm(range(start_year_new, end_year_new)):
    print(f"\nCurrently excluding year: {exclude_year}")

    # Initialize DataFrame for the results of this iteration
    df_results = pd.DataFrame(np.zeros((len(df), 8)), columns=['STANAME', 'STAID', 'latitude', 'longitude', 
                                                                'mean_diff_winter', 'mean_diff_spring', 'mean_diff_summer', 'mean_diff_autumn'])
    df_results[:] = np.nan

    for (i, file_name) in enumerate(df.file_name):
        print(f'\rCurrently calculating station {i+1} out of {len(df.file_name)}', end='')


        try:
            test = QAR_climate(sFile=file_name, dropna=drop_na_larger_than, fTau=tau, 
                               oldend = str(end_year_old) + '-', oldstart=str(start_year_old) + '-', 
                               newend = str(end_year_new) + '-', newstart=str(start_year_new) + '-',
                               include_nao=True, split_nao=False, iLeafs=1)
            test.prepare_data()
            
            # Filter out the 'exclude_year' from the new data
            test.new = test.new[test.new.index.year != exclude_year]
            test.results()

            for leaf in range(test.iLeafs):
                # Differences in persistence for NAO+
                diff_pers = test.mCurves_new - test.mCurves_old
                diff_pers.index = datetime_index_2019
                mean_diff_pers_winter = diff_pers.loc[diff_pers.index.month.isin([12, 1, 2])].mean()
                mean_diff_pers_spring = diff_pers.loc[diff_pers.index.month.isin([3, 4, 5])].mean()
                mean_diff_pers_summer = diff_pers.loc[diff_pers.index.month.isin([6, 7, 8])].mean()
                mean_diff_pers_autumn = diff_pers.loc[diff_pers.index.month.isin([9, 10, 11])].mean()


                df_results.loc[i, [season_list_pers[i] for i in range(len(season_list_pers))]] = (
                    mean_diff_pers_winter, mean_diff_pers_spring,
                    mean_diff_pers_summer, mean_diff_pers_autumn
                )

            # Mean differences in temperature per season
            mean_diff_winter = test.new.loc[test.new.index.month.isin([12, 1, 2])].mean() - test.old.loc[test.old.index.month.isin([12, 1, 2])].mean()
            mean_diff_spring = test.new.loc[test.new.index.month.isin([3, 4, 5])].mean() - test.old.loc[test.old.index.month.isin([3, 4, 5])].mean()
            mean_diff_summer = test.new.loc[test.new.index.month.isin([6, 7, 8])].mean() - test.old.loc[test.old.index.month.isin([6, 7, 8])].mean()
            mean_diff_autumn = test.new.loc[test.new.index.month.isin([9, 10, 11])].mean() - test.old.loc[test.old.index.month.isin([9, 10, 11])].mean()

            # Store the results
            df_results.iloc[i, :8] = (df.city_name.iloc[i], df.STAID.iloc[i], df.latitude.iloc[i], df.longitude.iloc[i],
                                       mean_diff_winter.values[0], mean_diff_spring.values[0], mean_diff_summer.values[0], mean_diff_autumn.values[0])

        except ValueError:
            pass  # Handle any errors in processing

    # Finalize and save the results for this excluded year
    df_results = df_results.dropna().set_index('STANAME')
    

    # Save the results for this exclude year to a CSV file
    df_results.to_csv(f'/Users/admin/Documents/PhD/persistence/data_persistence/results_{exclude_year}_{tau}_{start_date}.csv')
    
############ END LOYO ANALYSIS ###############    



