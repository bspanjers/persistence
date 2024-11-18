#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 28 June 

@author: bspanjers
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.optimize as opt
from matplotlib.dates import MonthLocator, DateFormatter
from scipy import stats
np.set_printoptions(suppress=True)
from scipy.signal import savgol_filter
import matplotlib.colors as mcolors
import matplotlib.lines as mlines



class QAR_precipitation:
    def __init__(self, sFile=None, sCity=None, dropna=0.05, 
                 fTau=.5, month='01', day='-01', 
                 oldstart='1950-', oldend='1980-', 
                 newstart='1990-',  newend='2020-', mid=False, 
                 split_nao=False, include_nao=False, power_pers_nao='linear', positive_is_one=True,
                 num_terms_level=2, num_terms_pers=2, use_statsmodels=True,
                 path='/Users/admin/Downloads/ECA_blend_rr/'):
        self.sCity = sCity
        self.dropna = dropna
        self.sFile = sFile
        self.day = day
        self.month = month
        self.oldstart = oldstart
        self.oldend = oldend
        self.newstart = newstart
        self.newend = newend
        self.path = path
        self.old = None
        self.new = None
        self.control = None
        self.split_nao = split_nao
        self.positive_is_one = positive_is_one
        self.mid = mid
        self.include_nao = include_nao

        
    def prepare_data(self):
        nao_index = pd.read_csv('/Users/admin/Documents/PhD/persistence/data_persistence/norm.daily.nao.cdas.z500.19500101_current.csv')
        nao_index['date'] = pd.to_datetime(nao_index[['year', 'month', 'day']])
        nao_index.set_index('date', inplace=True)
        nao_index.drop(columns=['year', 'month', 'day'], inplace=True)        
        nao_index = nao_index.replace(np.nan, 0)
        nao_index.columns = ['nao_index_cdas']
        self.nao_index = nao_index

        missing = -9999
        lower = 0
        upper = 600
        stations = pd.read_csv(self.path + 'stations.txt', header=13, encoding='ISO-8859-1').iloc[:,:2]
        stations.columns = ['STAID', 'STANAME']
        stations['STANAME'] = stations['STANAME'].astype(str).str.rstrip()
        stations['STAID'] = stations['STAID'].astype(str).str.lstrip()
        stations.set_index('STAID', inplace=True)
        
        if self.sFile == None and self.sCity == None:
            raise ValueError('Provide file name or city name.')
        elif type(self.sFile) != type(None) and type(self.sCity) != type(None):
            raise ValueError('Provide either file name or city name, not both.')
        elif type(self.sFile) != type(None) and type(self.sCity) == type(None):
            temp = pd.read_csv(self.path + self.sFile, header=15)
            station_ID = temp.STAID[0]
            self.sCity = stations.loc[str(temp.STAID[0])].STANAME
        elif type(self.sFile) == type(None) and type(self.sCity) != type(None):
            station_ID = stations.index[np.where(stations.STANAME == self.sCity)[0]][0].strip()
            sFileCity = 'RR_STAID0' + '0' * (5 - len(station_ID)) + station_ID + '.txt'
            temp = pd.read_csv(self.path + sFileCity, header=15)
        self.station_ID = station_ID
        temp = temp[['    DATE', '   RR']]
        temp.columns = ['Date', 'Temp']
        

            
        temp.set_index('Date', inplace=True, drop=True)
        temp.index = pd.to_datetime(temp.index, format='%Y%m%d')
    
        temp = temp[~((temp.index.month == 2) & (temp.index.day == 29))] #delete leap days
        if (self.split_nao == True) or (self.include_nao==True):
            temp = temp.merge(nao_index, left_index=True, right_index=True)
        #if self.include_nao == False:
           # temp = temp.loc[temp.index.year >=1910]
        self.temp = temp

        old = temp[temp.index >= self.oldstart + self.month + self.day]
        old = old[old.index < self.oldend + self.month + self.day]
        if len(old) <= 1000:
            raise ValueError('Old dataset has not enough observations')
        if len(np.where(old.Temp == missing)[0]) / len(old) >= self.dropna:
            raise ValueError('Not enough observations')
        old.drop(old.index[np.where(old.Temp == missing)[0]], inplace=True)
        old = old.iloc[np.where(old <= upper)[0]]
        old = old.iloc[np.where(old >= lower)[0]]
        old = old[~old.index.duplicated(keep='first')]
        self.old = old
        
        if self.mid == True:
            mid = temp[temp.index < self.newstart + self.month + self.day]
            mid.drop(mid.index[np.where(mid.Temp == missing)[0]], inplace=True)
            self.mid = mid[mid.index >= self.oldend + self.month + self.day]
        
        new = temp[temp.index < str(int(self.newend[:-1]) + 1)+'-' + self.month + self.day]
        
        if len(new) <= 1000:
            raise ValueError('New dataset has not enough observations')
        if len(np.where(new.Temp == missing)[0]) / len(new) >= self.dropna:
            raise ValueError('Not enough observations')
            
        new.drop(new.index[np.where(new.Temp == missing)[0]], inplace=True)
        new = new.iloc[np.where(new <= upper)[0]]
        new = new.iloc[np.where(new >= lower)[0]]
        new = new[~new.index.duplicated(keep='first')]
        
        predict = new[new.index.year>=int(self.newend[:-1])]
        self.predict = predict
        
        new = temp[temp.index < self.newend + self.month + self.day]
        new = new[new.index >= self.newstart + self.month + self.day]
        
        y_prec_old = (old.Temp >= 5) * 1
        data_old = pd.DataFrame(y_prec_old, columns=['Temp'])
        # Generate example binary time series data for test.new
        y_prec_new = (new.Temp >= 5) * 1
        data_new = pd.DataFrame(y_prec_new, columns=['Temp'])
        if self.include_nao == True:
            data_old['nao_index_cdas'] = old.nao_index_cdas
            data_new['nao_index_cdas'] = new.nao_index_cdas
        
        # Assign season to each row
       # old['season'] = data_old.index.month.map(self.get_season).values
       # new['season'] = data_new.index.month.map(self.get_season).values
        
        self.new = new
        self.old = old

    # Define function to assign seasons
    def get_season(self, month):
        if month in [12, 1, 2]:
            return 'winter'
        elif month in [3, 4, 5]:
            return 'spring'
        elif month in [6, 7, 8]:
            return 'summer'
        elif month in [9, 10, 11]:
            return 'autumn'
    
    def create_fourier_terms(self, dates, num_terms, prefix):
        t = (dates - dates.min()) / pd.Timedelta(1, 'D') + dates.dayofyear[0] - 1
        fourier_terms = pd.DataFrame(np.ones(len(dates)))
        fourier_terms.columns = [prefix + 'const']
        for i in range(1, num_terms + 1):
            sin_term = pd.Series(np.sin(2 * np.pi * i * t / 365), name=f'{prefix}sin_{i}')
            cos_term = pd.Series(np.cos(2 * np.pi * i * t / 365), name=f'{prefix}cos_{i}')
            fourier_terms = pd.concat([fourier_terms, sin_term, cos_term], axis=1)
        return fourier_terms
    
    def makeX_uni(self, df):
        df = df.copy()
        index = df.index
        num_params_pers = self.num_terms_pers * 2 + 1
        num_params_const = self.num_terms_level * 2 + 1
        
        if self.split_nao == True:
            #define nao states
            df_leafs = self.dfleafs.loc[(self.dfleafs.index<=df.index[-1]) & (self.dfleafs.index>=df.index[0])]
            df_leafs = df_leafs.loc[df_leafs.index.isin(df.index)]
            self.df_leafs=df_leafs
            
            for bin_idx in range(self.iLeafs):
                prefix_trend = f'trend_{bin_idx+1}'
                prefix_const = f'constant_{bin_idx+1}_'
                # Repeat the column bin_idx `num_params_const` times and take the transpose
                repeated_col = pd.DataFrame([df_leafs.iloc[:, bin_idx]] * num_params_const).T.values
                # Generate the Fourier terms
                fourier_terms = repeated_col * self.create_fourier_terms(index, self.num_terms_level, prefix=prefix_const)
                # Store in the dictionary
                df[prefix_trend] = np.linspace(index.year[0], index.year[-1], len(df)) - index.year[0]
                df[fourier_terms.columns] = fourier_terms.values 
                
                prefix_pers = f'pers_{bin_idx + 1}_'
                # Repeat the column bin_idx `num_params_const` times and take the transpose
                repeated_col = pd.DataFrame([df_leafs.iloc[:, bin_idx]] * num_params_pers).T.values
                # Generate the Fourier terms
                fourier_terms_pers = repeated_col * self.create_fourier_terms(index, self.num_terms_pers, prefix=prefix_pers)
                # Store in the dictionary
                df[fourier_terms_pers.columns] = fourier_terms_pers.values * pd.concat([df.loc[:,'Temp']] * fourier_terms_pers.shape[1], axis=1).values
        else:
              
            #constant variables
            fourier_terms_constant = self.create_fourier_terms(index, self.num_terms_level, prefix='constant_')
            df[fourier_terms_constant.columns] = fourier_terms_constant.values
            
            #persistence variables
            fourier_terms_pers = self.create_fourier_terms(index, self.num_terms_pers, prefix='pers_')
            df[fourier_terms_pers.columns] = fourier_terms_pers.values * pd.concat([df.loc[:,'Temp']] * fourier_terms_pers.shape[1], axis=1).values
            df.insert(loc=0, column='Trend', value=np.linspace(index.year[0], index.year[-1], len(df)) - index.year[0])
            if self.include_nao == True:    
                df.insert(1, 'nao_index_cdas_winter', df.nao_index_cdas * df.nao_index_cdas.index.month.isin([12,1,2]) * 1)
                df.drop('nao_index_cdas', inplace=True, axis=1)
                
        #drop nao index
        if self.split_nao == True:
            df.drop('nao_index_cdas', inplace=True, axis=1)
        
        #define mX
        mX = df.iloc[:-1,:].drop('Temp', axis=1)
        self.mX = mX
        return mX
    
    # Create lag features for each season, including shifting nao_index_cdas and creating dummies
    def create_lagged_features(self, data, n_lags):
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
    def fit_ar_logistic_regression(self, data, n_lags, months=[12, 1, 2]):
        winter_data = data[data.index.month.isin(months)].copy()
        seasonal_data = self.create_lagged_features(winter_data, n_lags)
    
        # Prepare predictors (X) and response variable (y)
        X = seasonal_data[[f'lag_{lag}' for lag in range(1, n_lags + 1)] + list(seasonal_data.filter(like='nao_index_cat').columns)].values
        y = seasonal_data['Temp'].values
    
        # Add a constant term for the intercept
        X = sm.add_constant(X)
        
        # Standardize the predictors (optional but recommended)
        #scaler = StandardScaler()
        #X_scaled = scaler.fit_transform(X[:, 1:])  # Exclude the constant column
        #X_scaled = np.column_stack((np.ones(X_scaled.shape[0]), X_scaled))  # Add back the constant column
        
        # Fit the autoregressive logistic regression model
        model = sm.Logit(y, X)
        result = model.fit(disp=0)
    
        return result

