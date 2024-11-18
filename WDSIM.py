#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu
@author: admin
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.optimize as opt
from matplotlib.dates import MonthLocator, DateFormatter
from scipy import stats
#np.set_printoptions(suppress=True)
from scipy.signal import savgol_filter
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import seaborn as sns


class WDSIM:
    def __init__(self, sFile=None, sCity=None, dropna=0.05, 
                 fTau=.5, month='01', day='-01', 
                 oldstart='1950-', oldend='1980-', 
                 newstart='1990-',  newend='2020-', 
                 num_terms_level=2, num_terms_pers=2,  num_lags=1,
                 path='/Users/admin/Downloads/ECA_blend_tg/'):
        self.sCity = sCity
        self.dropna = dropna
        self.sFile = sFile
        self.fTau = fTau
        self.month = month
        self.day = day
        self.oldstart = oldstart
        self.oldend = oldend
        self.newstart = newstart
        self.newend = newend
        self.num_terms_level = num_terms_level
        self.num_terms_pers = num_terms_pers
        self.path = path
        self.old = None
        self.new = None
        self.num_lags = num_lags

        
    def prepare_data(self):
        missing = -999.9
        lower = -100
        upper = 60
        stations = pd.read_csv(self.path + 'stations.txt', header=13).iloc[:,:2]
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
            sFileCity = 'TG_STAID0' + '0' * (5 - len(station_ID)) + station_ID + '.txt'
            temp = pd.read_csv(self.path + sFileCity, header=15)
        self.station_ID = station_ID
            
        temp = temp[['    DATE', '   TG']]
        temp.columns = ['Date', 'Temp']
        temp.Temp /= 10
    
            
        temp.set_index('Date', inplace=True, drop=True)
        temp.index = pd.to_datetime(temp.index, format='%Y%m%d')
        self.temp = temp
    
        old = temp[temp.index >= self.oldstart + self.month + self.day]
        old = old[old.index < self.oldend + self.month + self.day]
        if len(old) <= 5000:
            raise ValueError('Old dataset has not enough observations')
        if len(np.where(old.Temp == missing)[0]) / len(old) >= self.dropna:
            raise ValueError('Not enough observations')
        old.drop(old.index[np.where(old.Temp == missing)[0]], inplace=True)
        old = old.iloc[np.where(old <= upper)[0]]
        old = old.iloc[np.where(old >= lower)[0]]
        old = old[~old.index.duplicated(keep='first')]
        self.old = old

        
        new = temp[temp.index < self.newend + self.month + self.day]
        new = new[new.index >= self.newstart + self.month + self.day]
        
        if len(new) <= 5000:
            raise ValueError('New dataset has not enough observations')
        if len(np.where(new.Temp == missing)[0]) / len(new) >= self.dropna:
            raise ValueError('Not enough observations')
            
        new.drop(new.index[np.where(new.Temp == missing)[0]], inplace=True)
        new = new.iloc[np.where(new <= upper)[0]]
        new = new.iloc[np.where(new >= lower)[0]]
        new = new[~new.index.duplicated(keep='first')]
        self.new = new
        
  
 
      
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

        # Constant variables
        fourier_terms_constant = self.create_fourier_terms(index, self.num_terms_level, prefix='constant_')
        df[fourier_terms_constant.columns] = fourier_terms_constant.values
    
        # Persistence variables for each lag
        for lag in range(1, self.num_lags + 1):
            prefix_pers = f'pers_lag{lag}_'
            fourier_terms_pers = self.create_fourier_terms(index, self.num_terms_pers, prefix=prefix_pers)
            lagged_temp = df['Temp'].shift(lag).fillna(0)
            df[fourier_terms_pers.columns] = fourier_terms_pers.values * pd.concat([lagged_temp] * fourier_terms_pers.shape[1], axis=1).values
        df.insert(loc=0, column='Trend', value=np.linspace(index.year[0], index.year[-1], len(df)) - index.year[0])

        # Define mX
        mX = df.iloc[self.num_lags:, :].drop('Temp', axis=1)  # Drop rows according to the number of lags
        self.mX = mX
        return mX



    def create_year_df(self, year):
        # Create a DatetimeIndex for the given year, excluding February 29 for leap years
        if pd.Timestamp(str(year)).is_leap_year:
            dates = pd.date_range(start=f"{year}-01-01", end=f"{year}-12-31").to_series()
            dates = dates[~((dates.dt.month == 2) & (dates.dt.day == 29))]
        else:
            dates = pd.date_range(start=f"{year}-01-01", end=f"{year}-12-31")
    
        # Create a DataFrame with zeros and the DatetimeIndex
        df = pd.DataFrame(0, index=dates, columns=['Value'])
        return df
    
    def results(self):
        if type(self.old) == type(None):
            self.prepare_data()
        num_params_pers = self.num_lags * (self.num_terms_pers * 2 + 1)
        num_params_const = self.num_terms_level * 2 + 2
        ### OLD RESULTS ###
        # Adjust vY_old to skip initial rows based on number of lags
        vY_old = self.old.iloc[self.num_lags:,:1].reset_index(drop=True)
        mX_old = self.makeX_uni(self.old).reset_index(drop=True)
        modelold = sm.QuantReg(vY_old, mX_old)
        resultold = modelold.fit(q=self.fTau, vcov='robust', max_iter=5000)
    
        ### NEW RESULTS ###
        # Adjust vY_new to skip initial rows based on number of lags
        vY_new = self.new.iloc[self.num_lags:,:1].reset_index(drop=True)
        mX_new = self.makeX_uni(self.new).reset_index(drop=True)
        modelnew = sm.QuantReg(vY_new, mX_new)
        resultnew = modelnew.fit(q=self.fTau, vcov='robust', max_iter=5000)
        
        results = resultnew, resultold
        self.mX_new, self.mX_old = mX_new, mX_old
        self.oldfitted = resultold.fittedvalues
        self.newfitted = resultnew.fittedvalues
    
        # Store the parameters
        self.vThetanew = resultnew.params.values
        self.vThetaold = resultold.params.values
    
        # Define the yearly persistence curves for each lag
        df_old, df_new = self.create_year_df(self.old.index.year[-1]), self.create_year_df(self.new.index.year[-1])
        
        # Compute persistence terms for each lag in old and new periods
        self.mCurves_old = {}
        self.mCurves_new = {}
    
        for lag in range(1, self.num_lags + 1):
            lag_prefix = f'pers_lag{lag}_'
            # Compute Fourier terms and persistence curves for each lag
            fourier_terms_old = self.create_fourier_terms(df_old.index, self.num_terms_pers, prefix=lag_prefix)
            fourier_terms_new = self.create_fourier_terms(df_new.index, self.num_terms_pers, prefix=lag_prefix)
    
            # Compute persistence curves for the old and new periods for each lag
            start_index = num_params_const + int((num_params_pers / self.num_lags)) * (lag - 1)
            end_index = num_params_const + int((num_params_pers / self.num_lags)) * lag
            self.mCurves_old[lag] = self.vThetaold[start_index: end_index] @ fourier_terms_old.T
            self.mCurves_new[lag] = self.vThetanew[start_index: end_index] @ fourier_terms_new.T
    
            # Set indices for proper time alignment
            self.mCurves_old[lag].index = df_old.index
            self.mCurves_new[lag].index = df_new.index
        return results

    def return_params_quantiles(self, vQuantiles):
        if type(self.old) == type(None):
            self.prepare_data()
        
        # Define dimensions to handle multiple lags in the persistence terms
        num_params_total = 2 + 2 * self.num_terms_level + self.num_lags * (2 * self.num_terms_pers + 1)
        df_params_new = pd.DataFrame(np.zeros((num_params_total, len(vQuantiles))), columns=[str(round(t, 2)) for t in vQuantiles])
        df_params_old = df_params_new.copy()
    
        for tau in vQuantiles:
            self.fTau = tau
            resultnew, resultold = self.results()
            df_params_new.loc[:, str(round(tau, 2))] = resultnew.params.values
            df_params_old.loc[:, str(round(tau, 2))] = resultold.params.values
        return df_params_new, df_params_old
    
    def bootstrap_one_path_original(self, df, df_params, vQuantiles):
        U_t = np.random.choice(vQuantiles, df.shape[0] - 1)
        corresponding_columns = df_params.loc[:, [str(round(u, 2)) for u in U_t]].T.reset_index(drop=True)
    
        Ystar = np.zeros(len(U_t) + 1)
        Ystar[0] = df.Temp.values[0]
        
        # Create Fourier terms for each lag
        fourier_pers_all_lags = []
        for lag in range(1, self.num_lags + 1):
            fourier_pers_all_lags.append(self.create_fourier_terms(df.index, self.num_terms_pers, f'pers_lag{lag}_').values)
        fourier_pers = self.create_fourier_terms(df.index, self.num_terms_pers, 'pers_').values
        mX = self.makeX_uni(df).reset_index(drop=True).iloc[:, :-self.num_lags * (2 * self.num_terms_pers + 1)]
    
        # Constants and other non-persistence terms are at the beginning
        num_params_const = 2 + 2 * self.num_terms_level
    
        # Precompute the persistence terms for all lags to reduce repeated computation inside the loop
        persistence_indices = [
            (num_params_const + int(2 * self.num_terms_pers + 1) * (lag - 1),
             num_params_const + int(2 * self.num_terms_pers + 1) * lag)
            for lag in range(1, self.num_lags + 1)
        ]
        
        # Convert corresponding_columns and mX to numpy arrays for faster row access
        corresponding_columns_np = corresponding_columns.to_numpy()
        mX_np = mX.to_numpy()
        
        # Loop through each row in mX (vectorized approach)
        for i in range(1, mX_np.shape[0] + 1):
            # Sum the persistence terms across all lags for this row
            pers_terms_sum = sum(
                np.dot(corresponding_columns_np[i - 1, start:end], fourier_pers[i - 1, :] * Ystar[i - lag])
                for lag, (start, end) in enumerate(persistence_indices, start=1)
            )
        
            # Compute Ystar using the constant terms and the sum of persistence terms
            Ystar[i] = np.dot(corresponding_columns_np[i - 1, :num_params_const], mX_np[i - 1, :]) + pers_terms_sum

        return Ystar

    
    def bootstrap_unconditional_quantiles(self, iQuantiles=49, iB=500):
        if type(self.new) == type(None):
            self.prepare_data()
        vQuantiles = np.linspace(0, 1, iQuantiles + 2)[1: -1]
        df_params_new, df_params_old = self.return_params_quantiles(vQuantiles)
       
        mB_new = pd.DataFrame(np.zeros((iB, self.new.shape[0])))
        for b in range(iB):
            print(f'\rCurrently performing bootstrap {b+1} out of {iB}.', end='')
            mB_new.iloc[b, :] = self.bootstrap_one_path_original(self.new, df_params_new, vQuantiles)
        self.mB_new = mB_new
        



