#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 13:50:30 2023

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


class QAR_temperature:
    def __init__(self, sFile=None, sCity=None, dropna=0.05, 
                 fTau=.5, month='01', day='-01', 
                 oldstart='1950-', oldend='1980-', 
                 newstart='1990-',  newend='2020-', 
                 Kelvin=False, mid=False, 
                 split_nao=False, include_nao=False, iLeafs=1, power_pers_nao='linear', positive_is_one=True,
                 num_terms_level=2, num_terms_pers=2, use_statsmodels=True,
                 path='/Users/admin/Downloads/ECA_blend_tg/'):
        self.Kelvin = Kelvin
        self.sCity = sCity
        self.dropna = dropna
        self.sFile = sFile
        self.iLeafs = iLeafs
        self.fTau = fTau
        self.month = month
        self.power_pers_nao = power_pers_nao
        self.day = day
        self.oldstart = oldstart
        self.oldend = oldend
        self.newstart = newstart
        self.newend = newend
        self.num_terms_level = num_terms_level
        self.num_terms_pers = num_terms_pers
        self.path = path
        self.use_statsmodels = use_statsmodels
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
        quantiles = np.quantile(self.nao_index.nao_index_cdas, np.arange(0, 1, 1 / self.iLeafs)[1:])
        if self.iLeafs == 2:
           quantiles = quantiles - quantiles

        # Use np.digitize to find the bin indices for each value in nao_index
        bin_indices = np.digitize(self.nao_index.nao_index_cdas, quantiles)
        
        # Create an empty DataFrame with the same number of rows as nao_index and self.iLeafs columns
        dfleafs = pd.DataFrame(0, index=np.arange(len(self.nao_index.nao_index_cdas)), columns=np.arange(self.iLeafs))
        
        # Assign the values to the corresponding bins
        for i, bin_idx in enumerate(bin_indices):
            dfleafs.iloc[i, bin_idx] = 1
        dfleafs.columns = np.arange(1, self.iLeafs + 1)
        dfleafs.index = nao_index.index
        self.dfleafs = dfleafs

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
        
        if self.Kelvin == True:
            temp.Temp += 273.15
            
        temp.set_index('Date', inplace=True, drop=True)
        temp.index = pd.to_datetime(temp.index, format='%Y%m%d')
        temp = temp[~((temp.index.month == 2) & (temp.index.day == 29))] #delete leap days
        if (self.split_nao == True) or (self.include_nao==True):
            temp = temp.merge(nao_index, left_index=True, right_index=True)
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
        
        if self.mid == True:
            mid = temp[temp.index < self.newstart + self.month + self.day]
            mid.drop(mid.index[np.where(mid.Temp == missing)[0]], inplace=True)
            self.mid = mid[mid.index >= self.oldend + self.month + self.day]
        
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
                if self.include_nao == True:    
                    df['nao_index_cdas_winter_' + str(bin_idx+1)] = df_leafs.iloc[:, bin_idx] *  df.nao_index_cdas * df.nao_index_cdas.index.month.isin([12,1,2]) * 1
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
            #df.insert(loc=0, column='Trend', value=0)
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
    
    def rho_tau(self, x):
        return x * (self.fTau - (x < 0) * 1) 

    def quantregmulti(self, vTheta, args):
        if (self.fixed_params == False) or (self.data == 'old'):
            mX, vY = args
            vY_temp = vY.Temp.values
            rho_args = vY_temp - np.dot(vTheta, mX.values.T)
            quantsum = np.sum(self.rho_tau(rho_args))       
        else: 
            mX, vY, fixed = args
            vY_temp = vY.Temp.values
            vTheta[:-(self.num_terms_pers * 2 + 1)] = fixed
            rho_args = vY_temp - np.dot(vTheta, mX.values.T)
            quantsum = np.sum(self.rho_tau(rho_args))    
        return quantsum

    def CalcParams(self, fixed_params=False):
        self.fixed_params = fixed_params
        num_params_pers = self.num_terms_pers * 2 + 1
        self.prepare_data()
        self.mX_new = self.makeX_uni(self.new).reset_index(drop=True)
        self.vY_new = self.new[1:].reset_index(drop=True)
        self.mX_old = self.makeX_uni(self.old).reset_index(drop=True)
        self.vY_old = self.old[1:].reset_index(drop=True)
        
        vTheta0 = list(np.repeat(-.1, self.mX_new.shape[1]))

        self.data = 'old'
        args_old = [self.mX_old, self.vY_old]
        #generate results
        lResults_old = opt.minimize(self.quantregmulti, vTheta0, args=(args_old),
                            method='SLSQP', options={'disp': False})
        self.vThetaold = lResults_old.x
        
        self.data = 'new'
        if fixed_params == True:
            fixed = lResults_old.x[:-num_params_pers]
            vTheta0[:-num_params_pers] = fixed
            args_new = [self.mX_new, self.vY_new, fixed]
        else: 
            args_new = [self.mX_new, self.vY_new]
            
        self.iT_new = len(self.new)
        #generate results
        lResults_new = opt.minimize(self.quantregmulti, vTheta0, args=(args_new),
                            method='SLSQP', options={'disp': False})
        self.vThetanew = lResults_new.x
        

        self.oldfitted, self.newfitted = pd.Series(np.dot(self.vThetaold, self.mX_old.values.T)), pd.Series(np.dot(self.vThetanew, self.mX_new.values.T))
        return lResults_new.x, lResults_old.x
    
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
    
    def results(self, mid=False):
        if type(self.old) == type(None):
            self.prepare_data()
        num_params_pers = self.num_terms_pers * 2 + 1

        ### OLD RESULTS ###
        vY_old = self.old.iloc[1:,:1].reset_index(drop=True)
        mX_old = self.makeX_uni(self.old).reset_index(drop=True)
        modelold = sm.QuantReg(vY_old, mX_old)
        resultold = modelold.fit(q=self.fTau, vcov='robust', max_iter=5000)
        
        
        ### NEW RESULTS ###
        vY_new = self.new.iloc[1:,:1].reset_index(drop=True)
        mX_new = self.makeX_uni(self.new).reset_index(drop=True)
        modelnew = sm.QuantReg(vY_new, mX_new)
        resultnew = modelnew.fit(q=self.fTau, vcov='robust', max_iter=5000)
        results = resultnew, resultold
        self.mX_new, self.mX_old = mX_new, mX_old
        self.oldfitted = resultold.fittedvalues
        self.newfitted = resultnew.fittedvalues
        #self.mX_new = mX_new
        #self.mX_old = mX_old
        
        if mid == True:
            vY_mid = self.mid[1:].reset_index(drop=True)
            mX_mid = self.makeX_uni(self.mid).reset_index(drop=True)
            modelmid = sm.QuantReg(vY_mid, mX_mid)
            resultmid = modelmid.fit(q=self.fTau, vcov='robust', max_iter=5000)
            results = resultnew, resultold, resultmid
            self.midfitted = resultmid.fittedvalues
        
        self.vThetanew = resultnew.params.values
        self.vThetaold = resultold.params.values
        df_old, df_new = self.create_year_df(self.old.index.year[-1]), self.create_year_df(self.new.index.year[-1])
        self.mCurves_old = self.vThetaold[-num_params_pers:] @ self.create_fourier_terms(df_old.index, self.num_terms_pers, 'pers_').T
        self.mCurves_new = self.vThetanew[-num_params_pers:] @ self.create_fourier_terms(df_new.index, self.num_terms_pers, 'pers_').T          
        self.mCurves_old.index = df_new.index
        self.mCurves_new.index = df_new.index
        if self.split_nao == True:
            self.curve_old_plus = self.vThetaold[-2*num_params_pers:-num_params_pers] @ self.create_fourier_terms(df_old.index, self.num_terms_pers, 'pers_').T
            self.curve_old_min = self.vThetaold[-num_params_pers:] @ self.create_fourier_terms(df_old.index, self.num_terms_pers, 'pers_').T
            self.curve_new_plus = self.vThetanew[-2*num_params_pers:-num_params_pers] @ self.create_fourier_terms(df_old.index, self.num_terms_pers, 'pers_').T
            self.curve_new_min = self.vThetanew[-num_params_pers:] @ self.create_fourier_terms(df_new.index, self.num_terms_pers, 'pers_').T          
            self.curve_old_plus.index = df_new.index
            self.curve_new_plus.index = df_new.index
            self.curve_old_min.index = df_new.index
            self.curve_new_min.index = df_new.index
        return results
    
    def make_plots(self, mid=False, alpha=.1):
        iPers = 2 * self.num_terms_pers + 1
        if self.use_statsmodels == True:
            if mid == False:
                resultnew, resultold = self.results()
            else:
                 resultnew, resultold, resultmid = self.results(mid)
                 coef_mid = resultmid.params.values[-iPers:]
                 conf_int_mid = resultmid.conf_int(alpha=alpha).values[-iPers:]
            
            # Extract coefficients and confidence intervals from the summary table        
            coef_new = resultnew.params.values[-iPers:]
            conf_int_new = resultnew.conf_int(alpha=alpha).values[-iPers:]
            # Extract coefficients and confidence intervals from the old result
            coef_old = resultold.params.values[-iPers:]
            conf_int_old = resultold.conf_int(alpha=alpha).values[-iPers:]
            variable_names = resultnew.params.index[-iPers:]
        else:
            vThetanew, vThetaold = self.CalcParams()
            coef_old = vThetaold[-iPers:]
            coef_new = vThetanew[-iPers:]
        # Extract variable names
        
        # Number of coefficients
        num_coefs = len(coef_new)
        
        # Plot the estimates and confidence intervals
        fig, ax = plt.subplots(dpi=700, figsize=(12, 8))

        
        # Plot point estimates
        ax.plot(coef_new, marker='o', color='red', linestyle='', label='New Estimates: ' + str(self.new.index.year[0]) + '-' + str(self.new.index.year[-1]))
        

        if mid == True:
            # Plot point estimates
            ax.plot(np.arange(num_coefs) + 0.4, coef_mid, marker='o', color='black', linestyle='', label='Mid Estimates')
            
            # Plot confidence intervals for the mid result
            for i in range(num_coefs):
                ax.plot([i + 0.4, i + 0.4], conf_int_mid[i], color='black', linewidth=2)
        # Plot point estimates
        ax.plot(np.arange(num_coefs) + 0.2, coef_old, marker='o', color='orange', linestyle='', label='Old Estimates: ' + str(self.old.index.year[0]) + '-' + str(self.old.index.year[-1]))
        
        if self.use_statsmodels == True:
            # Plot confidence intervals
            for i in range(num_coefs):
                ax.plot([i, i], conf_int_new[i], color='red', linewidth=2)
                    
            # Plot confidence intervals for the old result
            for i in range(num_coefs):
                ax.plot([i + 0.2, i + 0.2], conf_int_old[i], color='orange', linewidth=2)
        

        
        # Customize the plot
        ax.set_xticks(range(num_coefs))
        if self.use_statsmodels == True:
            ax.set_xticklabels(variable_names, rotation=90)
        ax.set_ylabel('Coefficient Value')
        ax.legend()
        ax.set_title(self.sCity + ': Regression Coefficients and 90% Confidence Intervals for $\\tau$=' + str(self.fTau))
        plt.grid(True)

        plt.tight_layout()
        plt.show()
        
    def plot_fourier_fit(self, mid=False, conf_intervals=True, alpha=.1, fixed_params=False, plot=True):
        num_params_pers = self.num_terms_pers * 2 + 1
        num_params_const = self.num_terms_level * 2 + 1
        
        if type(self.old) == type(None):
            self.prepare_data()
            
        if self.use_statsmodels == True:
            if mid == False:
                resultnew, resultold = self.results()
            else: 
                resultnew, resultold, resultsmid = self.results(mid)
                self.vThetamid = resultsmid.params.values
        else:
            
            self.vThetanew, self.vThetaold = self.CalcParams(fixed_params)
       
        df_old, df_new = self.create_year_df(self.old.index.year[-1]), self.create_year_df(self.new.index.year[-1])
        if plot == True:
            plt.figure(dpi=700, figsize=(12, 8))
            plt.plot(df_new.index, self.vThetanew[1: num_params_const + 1] @ self.create_fourier_terms(df_new.index, self.num_terms_level, 'constant_').iloc[-365:].T, color='red', label='New Estimates: ' + str(self.new.index.year[0]) + '-' + str(self.new.index.year[-1] + 1))
            plt.plot(df_new.index[-365:], self.vThetaold[1: num_params_const + 1] @ self.create_fourier_terms(df_old.index, self.num_terms_level, 'constant_').iloc[-365:].T, color='orange', label='Old Estimates: ' + str(self.old.index.year[0]) + '-' + str(self.old.index.year[-1] + 1))
            if mid == True:
                plt.plot(self.new.Temp.index[-365:], self.vThetamid[1: 2 * self.num_terms_level + 2] @ self.create_fourier_terms(self.mid.index, self.num_terms_level, 'constant_').iloc[-365:].T, color='green', label='Mid Estimates: ' + str(self.old.index.year[-1]) + '-' + str(self.old.index.year[0] + 1))
            plt.legend()
            ax = plt.gca()
            plt.title(self.sCity + ': Regression Coefficients for the constant $\\alpha(\\tau)$ with $\\tau$=' + str(self.fTau))
            ax.xaxis.set_major_locator(MonthLocator())
            ax.xaxis.set_major_formatter(DateFormatter('%b'))
            plt.show()    
        
        curve_old = self.vThetaold[-num_params_pers:] @ self.create_fourier_terms(df_old.index, self.num_terms_pers, 'pers_').iloc[-365:].T
        curve_new = self.vThetanew[-num_params_pers:] @ self.create_fourier_terms(df_new.index, self.num_terms_pers, 'pers_').iloc[-365:].T          
        curve_old.index = curve_new.index
            
        if conf_intervals == True:
            cov_new, cov_old = resultnew.cov_params().iloc[-num_params_pers:, -num_params_pers:], resultold.cov_params().iloc[-num_params_pers:, -num_params_pers:]
            fourier_terms_old, fourier_terms_new = self.create_fourier_terms(self.old.index, self.num_terms_pers, 'pers_'), self.create_fourier_terms(self.new.index, self.num_terms_pers, 'pers_')
            std_old, std_new = np.sqrt(np.diag(fourier_terms_old @ cov_old @ fourier_terms_old.T)), np.sqrt(np.diag(fourier_terms_new @ cov_new @ fourier_terms_new.T))      
            print(np.diag(fourier_terms_old @ cov_old @ fourier_terms_old.T))
            curve_old.index = curve_new.index
            lower_old, lower_new = curve_old - stats.norm.ppf(1 - alpha / 2) * std_old[-365:], curve_new - stats.norm.ppf(1 - alpha / 2) * std_new[-365:]
            upper_old, upper_new = curve_old + stats.norm.ppf(1 - alpha / 2) * std_old[-365:], curve_new + stats.norm.ppf(1 - alpha / 2) * std_new[-365:]
            
            if mid == True:
                cov_mid = resultsmid.cov_params().iloc[-num_params_pers:, -num_params_pers:]
                fourier_terms_mid = self.create_fourier_terms(self.mid.index, self.num_terms_pers, 'pers_')
                std_mid = np.sqrt(np.diag(fourier_terms_mid @ cov_mid @ fourier_terms_mid.T))
                curve_mid = self.vThetamid[-num_params_pers:] @ self.create_fourier_terms(self.mid.index, self.num_terms_pers, 'pers_').iloc[-365:].T        
                curve_mid.index = curve_new.index
                lower_mid, upper_mid = curve_mid - stats.norm.ppf(1 - alpha / 2) * std_mid[-365:], curve_mid + stats.norm.ppf(1 - alpha / 2) * std_mid[-365:]
        if plot == True: 
            plt.figure(figsize=(8, 4), dpi=700)
            plt.plot(curve_new, label='Estimated Curve New Data', color='red')
            plt.plot(curve_old, label='Estimated Curve Old Data', color='orange')
            if mid == True:
                plt.plot(curve_mid, label='Estimated Curve Mid Data', color='green')
                plt.fill_between(lower_mid.index, lower_mid, upper_mid, color='green', alpha=alpha)
            
            if conf_intervals == True:    
                plt.fill_between(lower_new.index, lower_new, upper_new, color='red', alpha=alpha)
                plt.fill_between(lower_old.index, lower_old, upper_old, color='orange', alpha=alpha)
            plt.xlabel('Month')
            plt.ylabel('Persistence parameter $\\phi(\\tau)$')
            plt.title(self.sCity + ': Regression Coefficients with ' + str(int(100 * (1 - alpha))) + '% confidence intervals for the persistence\n parameter $\phi(\\tau)$ with $\\tau$=' + str(self.fTau))
            plt.legend()
            plt.grid(True)
            # Setting ticks to be at the beginning of each month
            ax = plt.gca()
            ax.xaxis.set_major_locator(MonthLocator())
            ax.xaxis.set_major_formatter(DateFormatter('%b'))
            plt.show()
        self.mCurves_new = curve_new
        self.mCurves_old = curve_old
        if conf_intervals == True:
            self.mCurves_old_conf_low, self.mCurves_old_conf_up = lower_old, upper_old
            self.mCurves_new_conf_low, self.mCurves_new_conf_up = lower_new, upper_new

        """
        if self.split_nao == True:
            curve_old2 = self.vThetaold[-2*num_params_pers: -num_params_pers] @ self.create_fourier_terms(df_old.index, self.num_terms_pers, 'pers_').iloc[-365:].T
            curve_new2 = self.vThetanew[-2*num_params_pers: -num_params_pers] @ self.create_fourier_terms(df_new.index, self.num_terms_pers, 'pers_').iloc[-365:].T          
            curve_old2.index = curve_new2.index 
            plt.figure(figsize=(8, 4), dpi=700)
            plt.plot(curve_new2, label='Estimated Curve New Data', color='red')
            plt.plot(curve_old2, label='Estimated Curve Old Data', color='orange')
            #if conf_intervals == True:    
             #   plt.fill_between(lower_new.index, lower_new, upper_new, color='red', alpha=alpha)
              #  plt.fill_between(lower_old.index, lower_old, upper_old, color='orange', alpha=alpha)
            plt.xlabel('Month')
            plt.ylabel('Persistence parameter $\\phi(\\tau)$')
            plt.title(self.sCity + ': PLUS Regression Coefficients with ' + str(int(100 * (1 - alpha))) + '% confidence intervals for the persistence\n parameter $\phi(\\tau)$ with $\\tau$=' + str(self.fTau))
            plt.legend()
            plt.grid(True)
            # Setting ticks to be at the beginning of each month
            ax = plt.gca()
            ax.xaxis.set_major_locator(MonthLocator())
            ax.xaxis.set_major_formatter(DateFormatter('%b'))
            plt.show()
    """
    def plot_fourier_fit_full(self, vTau=[0.05, 0.1, .5, .9, .95], mid=False, conf_intervals=True, alpha=.1, fixed_params=False):
        num_params_pers = self.num_terms_pers * 2 + 1
        
        # Create a single figure and subplots
        fig, axs = plt.subplots(2, 2, figsize=(12, 6), dpi=200)
        fig.subplots_adjust(hspace=0.15)
        self.plot_all_phi_coefs(plot=False)
        ax = axs[0,0]
        ax.plot(self.mCurves_new, label=['$\\tau$=' + str(round(tau, 2)) for tau in self.vQuantiles])
        ax.set_ylabel('Persistence coefficient $\\phi(\\tau)$')
        ax.set_title('(' + chr(97 + 0) + ') $\phi(\\tau)$ for varying $\\tau$ (new data)')
        ax.legend(fontsize='small')
        ax.grid(True)
        ax.xaxis.set_major_locator(MonthLocator())
        ax.xaxis.set_major_formatter(DateFormatter('%b'))
        
        # Loop through each value of tau in vTau
        for i, tau in enumerate(vTau):
            if type(self.old) == type(None):
                self.prepare_data()
        
            self.fTau = tau
            
            if self.use_statsmodels == True:
                if mid == False:
                    resultnew, resultold = self.results()
                else: 
                    resultnew, resultold, resultsmid = self.results(mid)
                    self.vThetamid = resultsmid.params.values
            else:
                self.vThetanew, self.vThetaold = self.CalcParams(fixed_params)
            
            df_old, df_new = self.create_year_df(self.old.index.year[-1]), self.create_year_df(self.new.index.year[-1])
            curve_old = self.vThetaold[-num_params_pers:] @ self.create_fourier_terms(df_old.index, self.num_terms_pers, 'pers_').iloc[-365:].T
            curve_new = self.vThetanew[-num_params_pers:] @ self.create_fourier_terms(df_new.index, self.num_terms_pers, 'pers_').iloc[-365:].T          
            curve_old.index = curve_new.index
            
            if conf_intervals == True:
                cov_new, cov_old = resultnew.cov_params().iloc[-num_params_pers:, -num_params_pers:], resultold.cov_params().iloc[-num_params_pers:, -num_params_pers:]
                fourier_terms_old, fourier_terms_new = self.create_fourier_terms(self.old.index, self.num_terms_pers, 'pers_'), self.create_fourier_terms(self.new.index, self.num_terms_pers, 'pers_')
                std_old, std_new = np.sqrt(np.diag(fourier_terms_old @ cov_old @ fourier_terms_old.T)), np.sqrt(np.diag(fourier_terms_new @ cov_new @ fourier_terms_new.T))      
                curve_old.index = curve_new.index
                lower_old, lower_new = curve_old - stats.norm.ppf(1 - alpha / 2) * std_old[-365:], curve_new - stats.norm.ppf(1 - alpha / 2) * std_new[-365:]
                upper_old, upper_new = curve_old + stats.norm.ppf(1 - alpha / 2) * std_old[-365:], curve_new + stats.norm.ppf(1 - alpha / 2) * std_new[-365:]
                
                if mid == True:
                    cov_mid = resultsmid.cov_params().iloc[-num_params_pers:, -num_params_pers:]
                    fourier_terms_mid = self.create_fourier_terms(self.mid.index, self.num_terms_pers, 'pers_')
                    std_mid = np.sqrt(np.diag(fourier_terms_mid @ cov_mid @ fourier_terms_mid.T))
                    curve_mid = self.vThetamid[-num_params_pers:] @ self.create_fourier_terms(self.mid.index, self.num_terms_pers, 'pers_').iloc[-365:].T        
                    curve_mid.index = curve_new.index
                    lower_mid, upper_mid = curve_mid - stats.norm.ppf(1 - alpha / 2) * std_mid[-365:], curve_mid + stats.norm.ppf(1 - alpha / 2) * std_mid[-365:]
            
            # Plot on the corresponding subplot
            row = (i+1) // 2
            col = (i+1) % 2
            ax = axs[row, col]
            ax.plot(curve_new, label='Estimated Curve New Data', color='red')
            ax.plot(curve_old, label='Estimated Curve Old Data', color='orange')
            if mid == True:
                ax.plot(curve_mid, label='Estimated Curve Mid Data', color='green')
                ax.fill_between(lower_mid.index, lower_mid, upper_mid, color='green', alpha=.1)
            
            if conf_intervals == True:    
                ax.fill_between(lower_new.index, lower_new, upper_new, color='red', alpha=.1)
                ax.fill_between(lower_old.index, lower_old, upper_old, color='orange', alpha=.1)
            ax.set_ylabel('Persistence coefficient $\\phi(\\tau)$')
            ax.set_title('(' + chr(97 + i + 1) + ') $\\tau$=' + str(self.fTau))
            ax.legend()
            ax.grid(True)
            # Setting ticks to be at the beginning of each month
            ax.xaxis.set_major_locator(MonthLocator())
            ax.xaxis.set_major_formatter(DateFormatter('%b'))
            # Call plot_all_phi_coefs function and add its plot to the final subplot
        plt.tight_layout()        
        plt.show()    
        
        
    def plot_all_phi_coefs(self, iQuantiles=9, plot=True):
        if type(self.old) == type(None):
            self.prepare_data()
        num_params_pers = self.num_terms_pers * 2 + 1
        num_params_level = self.num_terms_level * 2 + 1

        vQuantiles = np.linspace(0, 1, iQuantiles + 2)[1: -1]
        self.vQuantiles = vQuantiles
        mCurves_new = pd.DataFrame(np.zeros(shape=(365, len(vQuantiles))), index = self.new.index[-365:], columns=[str(round(tau, 2)) for tau in vQuantiles])
        mCurves_old = pd.DataFrame(np.zeros(shape=(365, len(vQuantiles))), index = self.old.index[-365:], columns=[str(round(tau, 2)) for tau in vQuantiles])
        mIntercepts_new = pd.DataFrame(np.zeros(shape=(365, len(vQuantiles))), index = self.new.index[-365:], columns=[str(round(tau, 2)) for tau in vQuantiles])
        mIntercepts_old = pd.DataFrame(np.zeros(shape=(365, len(vQuantiles))), index = self.old.index[-365:], columns=[str(round(tau, 2)) for tau in vQuantiles])
        mTrends_old = pd.DataFrame(np.zeros(shape=(365, len(vQuantiles))), index = self.old.index[-365:], columns=[str(round(tau, 2)) for tau in vQuantiles])
        mTrends_new = pd.DataFrame(np.zeros(shape=(365, len(vQuantiles))), index = self.new.index[-365:], columns=[str(round(tau, 2)) for tau in vQuantiles])
        mNAO_new_min, mNAO_new_plus = pd.DataFrame(np.zeros(shape=(365, len(vQuantiles))), index = self.new.index[-365:], columns=[str(round(tau, 2)) for tau in vQuantiles]), pd.DataFrame(np.zeros(shape=(365, len(vQuantiles))), index = self.new.index[-365:], columns=[str(round(tau, 2)) for tau in vQuantiles])
        mNAO_old_min, mNAO_old_plus = pd.DataFrame(np.zeros(shape=(365, len(vQuantiles))), index = self.old.index[-365:], columns=[str(round(tau, 2)) for tau in vQuantiles]), pd.DataFrame(np.zeros(shape=(365, len(vQuantiles))), index = self.old.index[-365:], columns=[str(round(tau, 2)) for tau in vQuantiles])
        df_old, df_new = self.create_year_df(self.old.index.year[-1]), self.create_year_df(self.new.index.year[-1])
        for tau in vQuantiles:
            self.fTau = tau
            resultnew, resultold = self.results()
            mCurves_new.loc[:, str(round(tau, 2))] = resultnew.params.values[-num_params_pers:] @ self.create_fourier_terms(df_new.index, self.num_terms_pers, 'pers_').iloc[-365:].T.values        
            mCurves_old.loc[:, str(round(tau, 2))] = resultold.params.values[-num_params_pers:] @ self.create_fourier_terms(df_old.index, self.num_terms_pers, 'pers_').iloc[-365:].T.values        
            mIntercepts_new.loc[:, str(round(tau, 2))] = resultnew.params.values[3: num_params_level + 3] @ self.create_fourier_terms(df_new.index, self.num_terms_level, 'pers_').iloc[-365:].T.values        
            mIntercepts_old.loc[:, str(round(tau, 2))] = resultold.params.values[3: num_params_level + 3] @ self.create_fourier_terms(df_old.index, self.num_terms_level, 'pers_').iloc[-365:].T.values        
            mNAO_new_min.loc[:, str(round(tau, 2))] = resultnew.params[1] 
            mNAO_old_min.loc[:, str(round(tau, 2))] = resultold.params[1] 
            mNAO_new_plus.loc[:, str(round(tau, 2))] = resultnew.params[2] 
            mNAO_old_plus.loc[:, str(round(tau, 2))] = resultold.params[2]             
            mTrends_new.loc[:, str(round(tau, 2))] = resultnew.params[0] * (np.linspace(self.new.index.year[0], self.new.index.year[-1], len(self.new)) - self.new.index.year[0])[-365:]
            mTrends_old.loc[:, str(round(tau, 2))] = resultold.params[0] * (np.linspace(self.new.index.year[0], self.new.index.year[-1], len(self.new)) - self.new.index.year[0])[-365:]
            
        self.mCurves_new, self.mCurves_old = mCurves_new, mCurves_old
        self.mIntercepts_new, self.mIntercepts_old = mIntercepts_new, mIntercepts_old
        self.mTrends_new, self.mTrends_old = mTrends_new, mTrends_old
        self.mNAO_new_plus, self.mNAO_new_min = mNAO_new_plus, mNAO_new_min
        self.mNAO_old_min, self.mNAO_old_plus = mNAO_old_plus, mNAO_old_min
        
        if plot == True:
            plt.figure(figsize=(8, 3), dpi=100) 
            plt.plot(mCurves_new, label=['$\\tau$=' + str(round(tau, 2)) for tau in vQuantiles])
            plt.xlabel(str(self.new.index.year[-1]))
            plt.ylabel('$\\phi(\\tau)$')
            plt.title(self.sCity + ': Persistence coefficient $\phi(\\tau)$ for varying $\\tau$')
            plt.legend()
            plt.grid(True)
            # Setting ticks to be at the beginning of each month
            ax = plt.gca()
            ax.xaxis.set_major_locator(MonthLocator())
            ax.xaxis.set_major_formatter(DateFormatter('%b'))
            plt.show()
            
            plt.figure(figsize=(8, 3), dpi=400)
            plt.plot(mCurves_old, label=['$\\tau$=' + str(round(tau, 2)) for tau in vQuantiles])
            plt.xlabel(str(self.old.index.year[-1]))
            plt.ylabel('$\\phi(\\tau)$')
            plt.title(self.sCity + ': Persistence coefficient $\phi(\\tau)$ for varying $\\tau$')
            plt.legend()
            plt.grid(True)
            # Setting ticks to be at the beginning of each month
            ax = plt.gca()
            ax.xaxis.set_major_locator(MonthLocator())
            ax.xaxis.set_major_formatter(DateFormatter('%b'))
            plt.show()
                    
    def gen_phi_alpha_with_states(self, iQuantiles=9):
        if type(self.old) == type(None):
            self.prepare_data()
        num_params_pers = self.num_terms_pers * 2 + 1
        num_params_level = self.num_terms_level * 2 + 1

        vQuantiles = np.linspace(0, 1, iQuantiles + 2)[1: -1]
        self.vQuantiles = vQuantiles

        # Initialize multi-index DataFrames to store curves and intercepts for all states
        index = pd.MultiIndex.from_product([range(self.iLeafs), self.new.index[-365:]], names=["State", "Date"])
        columns = [str(round(tau, 2)) for tau in vQuantiles]
        
        mCurves_new = pd.DataFrame(np.zeros((self.iLeafs * 365, len(vQuantiles))), index=index, columns=columns)
        mCurves_old = pd.DataFrame(np.zeros((self.iLeafs * 365, len(vQuantiles))), index=index, columns=columns)
        mIntercepts_new = pd.DataFrame(np.zeros((self.iLeafs * 365, len(vQuantiles))), index=index, columns=columns)
        mIntercepts_old = pd.DataFrame(np.zeros((self.iLeafs * 365, len(vQuantiles))), index=index, columns=columns)
        mStds_old = pd.DataFrame(np.zeros((self.iLeafs * 365, len(vQuantiles))), index=index, columns=columns)
        mStds_new = pd.DataFrame(np.zeros((self.iLeafs * 365, len(vQuantiles))), index=index, columns=columns)
        for state in range(self.iLeafs):
            for tau in vQuantiles:
                # Set the current quantile
                self.fTau = tau
                
                # Obtain results for the current quantile
                resultnew, resultold = self.results()
                df_results_new = self.gen_curves(self.new, resultnew, 'pers_new_', num_params_pers, num_params_level, state)
                df_results_old = self.gen_curves(self.old, resultold, 'pers_old_', num_params_pers, num_params_level, state, period='old')

                # Calculate the curves and intercepts for the new data
                mCurves_new.loc[(state,), str(round(tau, 2))] = df_results_new.iloc[:, 0].values
                mCurves_old.loc[(state,), str(round(tau, 2))] = df_results_old.iloc[:, 0].values
                
                mIntercepts_new.loc[(state,), str(round(tau, 2))] = df_results_new.iloc[:, 3].values
                mIntercepts_old.loc[(state,), str(round(tau, 2))] = df_results_old.iloc[:, 3].values
                
                mStds_new.loc[(state,), str(round(tau, 2))] = df_results_new.iloc[:, -1].values
                mStds_old.loc[(state,), str(round(tau, 2))] = df_results_old.iloc[:, -1].values
                
        # Assign the calculated curves and intercepts to the class attributes
        self.mCurves_new = mCurves_new
        self.mCurves_old = mCurves_old
        self.mIntercepts_new = mIntercepts_new
        self.mIntercepts_old = mIntercepts_old
        self.mStds_new = mStds_new
        self.mStds_old = mStds_old
        
    def plot_conditional_quantiles(self, state=1, ylag=9, date='-01-15', alpha=0.1, dpi=200, fixed_params=False):
        if state == 1:
            nao = 'NAO$+$'
        else: 
            nao= 'NAO$-$'
        

        # Extract the data for plotting
        old_data = self.mIntercepts_old.loc[state,'2019' + date] + self.mCurves_old.loc[state,'2019' + date] * ylag
        if fixed_params == True:
            new_data = self.mIntercepts_new.loc[state,'2019' + date] + self.mCurves_old.loc[state,'2019' + date] * ylag
        else: 
            new_data = self.mIntercepts_new.loc[state,'2019' + date] + self.mCurves_new.loc[state,'2019' + date] * ylag
            
        lower_old, upper_old = old_data - stats.norm.ppf(1 - alpha/2) * self.mStds_old.loc[state,'2019' + date], old_data + stats.norm.ppf(1 - alpha/2) * self.mStds_old.loc[state,'2019' + date]
        lower_new, upper_new = new_data - stats.norm.ppf(1 - alpha/2) * self.mStds_new.loc[state,'2019' + date], new_data + stats.norm.ppf(1 - alpha/2) * self.mStds_new.loc[state,'2019' + date]

        # Create the figure
        plt.figure(dpi=dpi)
        
        # Plot the old and new data
        plt.plot(old_data, label='old', color='orange')
        plt.plot(new_data, label='new', color='red')
        
        # Add the horizontal line at y=ylag
        plt.axhline(y=ylag, color='gray', linestyle='--', label='$y_{t-1}$')
        
        # Add legend
        plt.legend()
        
        # Function to find the exact crossing points
        def find_crossings(data, ylag):
            crossings = []
            for i in range(len(data) - 1):
                if (data[i] < ylag and data[i + 1] > ylag) or (data[i] > ylag and data[i + 1] < ylag):
                    # Linear interpolation to find the exact crossing point
                    crossing = i + (ylag - data[i]) / (data[i + 1] - data[i])
                    crossings.append(crossing)
            return crossings
        
        # Find the points where the curves cross ylag
        crossings_old = find_crossings(old_data, ylag)
        crossings_new = find_crossings(new_data, ylag)
        
        # Plot vertical lines at the crossing points
        for crossing in crossings_old:
            plt.axvline(x=crossing, color='orange', linestyle='--', alpha=0.5)
        
        for crossing in crossings_new:
            plt.axvline(x=crossing, color='red', linestyle='--', alpha=0.5)
        if fixed_params == False:
            plt.fill_between(old_data.index, lower_old, upper_old, color='orange', alpha=alpha)
            plt.fill_between(new_data.index, lower_new, upper_new, color='red', alpha=alpha)
        plt.title('Conditional temperature distribution with $y_{t-1}$=' + str(ylag) + ' for ' + self.sCity + '\n during ' + nao + ' for date ' + date[1:] + ' in year 2019 (new) and 1950 (old)')
        plt.xticks(rotation=90)
        plt.xlabel('Quantile $\\tau$')
        plt.ylabel('Next period\'s temperature $y_t$')
        # Show the plot
        plt.show()
        
    def hit_series(self, actual, forecast):
        if self.fTau <= .5:
            return (actual < forecast) * 1
        else:
            return (actual > forecast) * 1

    def plot_backtest(self, df, fitted, alpha, sTime='old', on='yearly'):    
        ###for the new data
        hits = pd.DataFrame(self.hit_series(df.Temp.values[1:], fitted))
        hits.index = df[1:].index
        if on == 'daily':
            daily_hits = hits.groupby(hits.index.strftime("%m-%d")).sum()
        elif on == 'yearly': 
            daily_hits = hits.groupby(hits.index.strftime("%y-%m")).sum()
            
        daily_probabilities =  daily_hits / np.sum(daily_hits)
        cumulative_probabilities = np.cumsum(daily_probabilities)
        #uniform_samples_new = stats.uniform.ppf(cumulative_probabilities_new[0].values)
        #plot empirical cdf
        plt.figure(figsize=(12, 8), dpi=100)
        multiply = self.fTau if self.fTau<= 0.5 else 1-self.fTau

        if on == 'daily':
            x = np.linspace(1, 365, len(cumulative_probabilities))
            plt.plot([1, 365], [0, 1], color='red', linestyle='--')  # y=x line
            plt.xlabel('Day in year')
            expected_counts = np.full(365, len(df) / 365) * multiply
            expected_counts[:] = np.sum(daily_hits.iloc[:,0]) / 365
        elif on == 'yearly':
            x = np.linspace(df.index.year[0], df.index.year[-1] + 1, len(cumulative_probabilities))
            plt.xlabel('Year')
            plt.plot([df.index.year[0],df.index.year[-1] + 1],[0, 1], color='red', linestyle='--')  # y=x line
            expected_counts = np.full(360, len(df) / 360) * multiply
            expected_counts[:] = np.sum(daily_hits.iloc[:,0]) / 360
            
        plt.ylabel('Cumulative density')
        plt.plot(x, cumulative_probabilities.values)

        plt.title('Empirical cdf based on ' + on + ' data of the ' + sTime + ' dataset')
        plt.grid(True)
        
        plt.show()
        # Plot the histogram of the generated uniform sample
        plt.figure(figsize=(12,8))
        plt.hist(cumulative_probabilities, bins=10, density=True, alpha=0.7, color='blue', label='Empirical Distribution')
        
        # Plot the theoretical uniform distribution
        x = np.linspace(0, 1, 100)
        plt.plot(x, stats.uniform.pdf(x), 'r-', lw=2, label='Theoretical Uniform Distribution')
        
        plt.title('Theoretical uniform pdf vs empirical pdf based on ' + on + ' data of the ' + sTime + ' dataset')
        plt.xlabel('Values')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True)
        plt.show()
        #print(daily_hits, expected_counts)
        # Perform the Kolmogorov-Smirnov test
        self.daily_hits=daily_hits
        self.expected_counts = expected_counts
        test_statistic, p_value = stats.chisquare(daily_hits, expected_counts)

        # Define significance level (e.g., 0.05)        
        # Check the result
        print('The p-value is ' + str(p_value[0]))
        if p_value[0] > alpha:
            print('The ' + on + ' averaged hit sequence for the ' + sTime + ' data appears to follow a uniform distribution (fail to reject H0)')
        else:
            print('The ' + on + ' averaged hit sequence for the ' + sTime + ' data does not follow a uniform distribution (reject H0)')
        
    def backtest(self, alpha=.05):
        self.plot_backtest(self.new, self.newfitted, alpha, sTime='new', on='daily')
        self.plot_backtest(self.old, self.oldfitted, alpha, sTime='old', on='daily')
        self.plot_backtest(self.new, self.newfitted, alpha, sTime='new', on='yearly')
        self.plot_backtest(self.old, self.oldfitted, alpha, sTime='old', on='yearly')

    def return_params_quantiles(self, vQuantiles):
        if type(self.old) == type(None):
            self.prepare_data()
        df_params_new = pd.DataFrame(np.zeros((1 + 2 * self.num_terms_level + 2 * self.num_terms_pers + 2, len(vQuantiles))), columns=[str(round(t,2)) for t in vQuantiles])
        df_params_old = df_params_new.copy()
        for tau in vQuantiles:
            self.fTau = tau
            resultnew, resultold = self.results()
            df_params_new.loc[:, str(round(tau, 2))] = resultnew.params.values
            df_params_old.loc[:, str(round(tau, 2))] = resultold.params.values
        return df_params_new, df_params_old

    
    def bootstrap_one_path(self, df, df_params, vQuantiles):
        U_t = np.random.choice(vQuantiles, 730)
        corresponding_columns = df_params.loc[:, [str(round(u, 2)) for u in U_t]].T.reset_index(drop=True)
 
        Ystar = np.zeros(len(U_t) + 1)
        Ystar[0] = df.Temp.values[0]
        fourier_pers = self.create_fourier_terms(df.index[-730:], self.num_terms_pers, 'pers_').values
        mX = self.makeX_uni(df).reset_index(drop=True).iloc[-730:, :-2 * self.num_terms_pers - 1].reset_index(drop=True)

        for i in np.arange(1, mX.shape[0]):
            Ystar[i] = np.dot(corresponding_columns.iloc[i-1, :-2 * self.num_terms_pers - 1], mX.values[i-1, :].T) + np.dot(corresponding_columns.iloc[i-1, -2 * self.num_terms_pers - 1:], fourier_pers[i-1, :] * Ystar[i-1]) #+ corresponding_resids[i-1]
        return Ystar[:-1]
    
    def bootstrap_one_path_original(self, df, df_params, vQuantiles):
        U_t = np.random.choice(vQuantiles, df.shape[0]-1)
        corresponding_columns = df_params.loc[:, [str(round(u, 2)) for u in U_t]].T.reset_index(drop=True)

        Ystar = np.zeros(len(U_t) + 1)
        Ystar[0] = df.Temp.values[0]
        fourier_pers = self.create_fourier_terms(df.index, self.num_terms_pers, 'pers_').values
        mX = self.makeX_uni(df).reset_index(drop=True).iloc[:, :-(2 * self.num_terms_pers + 1)]

        for i in np.arange(1, mX.shape[0] + 1):
            Ystar[i] = np.dot(corresponding_columns.iloc[i-1, :-2 * self.num_terms_pers - 1], mX.values[i-1, :].T) + np.dot(corresponding_columns.iloc[i-1, -(2 * self.num_terms_pers + 1):], fourier_pers[i-1, :] * Ystar[i-1])
        return Ystar
    
    def bootstrap_unconditional_quantiles(self, iQuantiles=9, iB=500, set_old_pers_params=False, old=True):
        if type(self.old) == type(None):
            self.prepare_data()
        self.iB = iB
        vQuantiles = np.linspace(0, 1, iQuantiles + 2)[1: -1]
        df_params_new, df_params_old = self.return_params_quantiles(vQuantiles)
        if set_old_pers_params == True:
            df_params_new_fix = df_params_new.copy()
            df_params_new_fix.iloc[:, -(1 + 2 * self.num_terms_pers):] = df_params_old.iloc[:, -1 - 2 * self.num_terms_pers:] 
            mB_new_fix = pd.DataFrame(np.zeros((iB, self.new.shape[0])))
        if old == True:
            mB_old = pd.DataFrame(np.zeros((iB, self.old.shape[0])))
        mB_new = pd.DataFrame(np.zeros((iB, self.new.shape[0])))
        for b in range(iB):
            print(f'\rCurrently performing bootstrap {b} out of {iB}', end='')
            mB_new.iloc[b, :] = self.bootstrap_one_path_original(self.new, df_params_new, vQuantiles)
            if old ==True:
                mB_old.iloc[b, :] = self.bootstrap_one_path_original(self.old, df_params_old, vQuantiles)
            if set_old_pers_params == True:
                mB_new_fix.iloc[b, :] = self.bootstrap_one_path_original(self.new, df_params_new_fix, vQuantiles)
        
        if set_old_pers_params==True: 
            self.mB_new_fix = mB_new_fix
        if old == True:
            self.mB_old = mB_old
        self.mB_new = mB_new
        
    def CalcParams_simu(self, vB):
        mX_b = self.makeX_uni(vB).reset_index(drop=True)
        vY_b = vB[1:].reset_index(drop=True)
        model_b = sm.QuantReg(vY_b, mX_b)
        result_b = model_b.fit(q=self.fTau, vcov='robust', max_iter=5000)
        return result_b
    
    def loop_simu(self, alpha=0.05):
        # Initialize variables
        vTheta_0 = self.CalcParams_simu(self.new).params  # Initial parameters
        self.fixed_params = False
        self.data = 'new'
        
        # Matrix to store bootstrapped parameters
        mTheta_B = np.zeros((self.iB, len(vTheta_0)))  # Assuming 11 as in your example
        mCov_B = []  # Store covariance matrices
        
        # Perform bootstrapping
        for b in range(self.iB):
            print(f'\rCurrently performing bootstrap {b+1} out of {self.iB}', end='')
            
            # Bootstrap sample
            vB = self.mB_new.iloc[b, :].copy()
            vB.name = 'Temp'
            vB = pd.DataFrame(vB)
            
            # Calculate parameters for bootstrap sample
            results_b = self.CalcParams_simu(vB)
            mTheta_B[b, :] = results_b.params
            mCov_B.append(results_b.cov_params())
        
        # Calculate coverage for each parameter in vTheta_0
        coverage = np.zeros(len(vTheta_0))  # Store coverage for each parameter
        
        for b in range(self.iB):
            # Extract bootstrap parameters and covariance matrix
            theta_b = mTheta_B[b, :]
            cov_b = mCov_B[b]
            
            # Calculate 95% confidence intervals for this bootstrap
            lower_bound = theta_b - stats.norm.ppf(1 - alpha/2) * np.sqrt(np.diag(cov_b))
            upper_bound = theta_b + stats.norm.ppf(1 - alpha/2) * np.sqrt(np.diag(cov_b))
            
            # Check if vTheta_0 is within the confidence interval
            coverage += np.logical_and(vTheta_0 >= lower_bound, vTheta_0 <= upper_bound)
        
        # Convert to percentage coverage
        coverage_percentage = (coverage / self.iB) * 100
        
        # Output the coverage table
        coverage_table = pd.DataFrame({
            'Parameter': np.arange(len(vTheta_0)),  # Assuming parameter indices
            'vTheta_0': vTheta_0,
            'Coverage (%)': coverage_percentage
        })
        
        print("\nCoverage Table:")
        print(coverage_table)        
        return mTheta_B, mCov_B, coverage_table

    def plot_unconditional_quantiles(self):
        plt.figure(figsize=(12,6), dpi=800)
        plt.plot(self.new.index[-366:-1], np.quantile(self.mB_new,.05,axis=0)[-366:-1], label='Quantile=0.05')
        plt.plot(self.new.index[-366:-1], np.quantile(self.mB_new,.1,axis=0)[-366:-1], label='Quantile=0.1')
        plt.plot(self.new.index[-366:-1], np.quantile(self.mB_new,.5,axis=0)[-366:-1], label='Quantile=0.5')
        plt.plot(self.new.index[-366:-1], np.quantile(self.mB_new,.9,axis=0)[-366:-1], label='Quantile=0.9')
        plt.plot(self.new.index[-366:-1], np.quantile(self.mB_new,.95,axis=0)[-366:-1], label='Quantile=0.95')
        plt.plot(self.new.index[-366:-1], self.new.Temp.values[-366:-1], label='Observed temperatures')
        plt.title('Bootstrapped quantiles obtained by performing {self.iB} bootstrap replications and observed temperatures (new data)') 
        plt.legend()
        plt.show()
        
        plt.figure(figsize=(12,6), dpi=800)
        plt.plot(self.old.index[-366:-1], np.quantile(self.mB_old,.05,axis=0)[-366:-1], label='Quantile=0.05')
        plt.plot(self.old.index[-366:-1], np.quantile(self.mB_old,.1,axis=0)[-366:-1], label='Quantile=0.1')
        plt.plot(self.old.index[-366:-1], np.quantile(self.mB_old,.5,axis=0)[-366:-1], label='Quantile=0.5')
        plt.plot(self.old.index[-366:-1], np.quantile(self.mB_old,.9,axis=0)[-366:-1], label='Quantile=0.9')
        plt.plot(self.old.index[-366:-1], np.quantile(self.mB_old,.95,axis=0)[-366:-1], label='Quantile=0.95')
        plt.plot(self.old.index[-366:-1], self.old.Temp.values[-366:-1], label='Observed temperatures')
        plt.title('Bootstrapped quantiles obtained by performing {self.iB} bootstrap replications and observed temperatures (old data)') 
        plt.legend()
        plt.show()
    
    def generate_color_codes(self, k, colormap='tab20'):
        """
        Generate k unique color codes using a specified colormap.
        
        Parameters:
            k (int): Number of color codes to generate.
            colormap (str): Name of the matplotlib colormap to use.
        
        Returns:
            list: A list of color codes in hex format.
        """
        cmap = plt.get_cmap(colormap)
        colors = [cmap(i / k) for i in range(k)]
        hex_colors = [mcolors.rgb2hex(color) for color in colors]
        return hex_colors


    def compare_unconditional_quantiles(self, vQuantiles=[0.05, 0.1, 0.5, 0.9, 0.95], plot=True, mB_fixed=None):
        dfSmoothed_new_fixed = pd.DataFrame(np.zeros((365, len(vQuantiles))), columns=[str(q) for q in vQuantiles])
        dfSmoothed_old = pd.DataFrame(np.zeros((365, len(vQuantiles))), columns=[str(q) for q in vQuantiles])
        dfSmoothed_new = pd.DataFrame(np.zeros((365, len(vQuantiles))), columns=[str(q) for q in vQuantiles])
        
        if plot==True:
            plt.figure(figsize=(10,6), dpi=100)
            color_codes = self.generate_color_codes(len(vQuantiles))
            if len(vQuantiles) == 3:
                color_codes = ['blue', 'purple', 'red']
            elif len(vQuantiles) == 1:
                color_codes = ['red']
        for (i,q) in enumerate(vQuantiles):
            dfSmoothed_old.loc[:, str(q)] = savgol_filter(np.quantile(self.mB_old.T, q, axis=1)[-366:-1], window_length=21, polyorder=2)
            dfSmoothed_new.loc[:, str(q)] = savgol_filter(np.quantile(self.mB_new.T, q, axis=1)[-366:-1], window_length=21, polyorder=2)
            self.dfSmoothed_old,  self.dfSmoothed_new = dfSmoothed_old, dfSmoothed_new
            if type(mB_fixed) != type(None):
                dfSmoothed_new_fixed.loc[:, str(q)] = savgol_filter(np.quantile(mB_fixed.T, q, axis=1)[-366:-1], window_length=21, polyorder=2)     
                self.dfSmoothed_new_fixed = dfSmoothed_new_fixed
            if plot == True:
                plt.plot(self.new.index[-366:-1], dfSmoothed_old.loc[:, str(q)], linewidth=1, color=color_codes[i], linestyle='dashdot')
                plt.plot(self.new.index[-366:-1], dfSmoothed_new.loc[:, str(q)], linewidth=1, color=color_codes[i], label='Quantile $\\tau=$' + str(q))
                plt.plot(self.new.index[-366:-1], dfSmoothed_new_fixed.loc[:, str(q)], linewidth=1, color=color_codes[i], linestyle='--')
        plt.gca().xaxis.set_major_formatter(DateFormatter('%b'))
        old_legend = mlines.Line2D([], [], color='black', linestyle='dashdot', label='Old $\mu_t(\\tau)$ and $\phi_t(\\tau)$')
        new_legend = mlines.Line2D([], [], color='black', linestyle='-', label='New $\mu_t(\\tau)$ and $\phi_t(\\tau)$')
        if type(mB_fixed) != type(None):
            newfixed_legend = mlines.Line2D([], [], color='black', linestyle='--', label='New $\mu_t(\\tau)$, old $\phi_t(\\tau)$')
        plt.yticks(np.arange(-6, 25 + 1, 1))
        plt.grid(which='both', axis='y', color='gray', linestyle='-', linewidth=0.5, alpha=0.2)

        plt.title(f'Quantile curves for {self.sCity} obtained by $S=${self.iB} model simulations for $\\tau\in$' + '{' + str(vQuantiles)[1:-1] + '}')
        # Add custom legend
        plt.legend(handles=[old_legend, new_legend, newfixed_legend], loc='best', handlelength=3)
        plt.show()
        
    def gen_curves(self, df, results, sSetting, num_params_pers, num_params_const, iLeaf, alpha=.1, period='new'):
        dfindex = self.create_year_df(self.new.index.year[-1]).index
        
        def find_first_number(string):
            for char in string:
                if char.isdigit():
                    return int(char)
            return None
        
        params_index_set = results.params.index
        # Get the columns where '1' is the first number
        selected_columns = [param for param in params_index_set if find_first_number(param) == iLeaf + 1]
        vTheta = results.params[selected_columns]
        vTheta_pers, vTheta_const = vTheta[-num_params_pers:], vTheta[:num_params_pers+2]
        cov = results.cov_params().loc[selected_columns, selected_columns]
        cov_pers = cov.iloc[-num_params_pers:, -num_params_pers:]
        cov_const = cov.iloc[:num_params_const + 2, :num_params_const + 2]
        
        fourier_terms_pers = self.create_fourier_terms(dfindex, self.num_terms_pers, sSetting)
        std_pers = np.sqrt(np.diag(fourier_terms_pers.values @ cov_pers.values @ fourier_terms_pers.values.T))   
        curve_pers = vTheta_pers @ self.create_fourier_terms(dfindex, self.num_terms_pers, sSetting).values.T

        fourier_terms_const = self.create_fourier_terms(dfindex, self.num_terms_level, sSetting)
        mX = self.mX_new if period == 'new' else self.mX_old
        if period == 'old':
            fourier_terms_const.insert(loc=0, column='trend_' + str(iLeaf+1), value=(np.linspace(self.new.index.year[0], self.new.index.year[-1], len(self.new)) - self.new.index.year[0])[:365])
            fourier_terms_const.insert(loc=1, column='nao_index_cdas_winter_' + str(iLeaf+1), value=mX['nao_index_cdas_winter_' + str(iLeaf+1)])
        else: 
            fourier_terms_const.insert(loc=0, column='trend_' + str(iLeaf+1), value=(np.linspace(self.new.index.year[0], self.new.index.year[-1], len(self.new)) - self.new.index.year[0])[-365:])
            fourier_terms_const.insert(loc=1, column='nao_index_cdas_winter_' + str(iLeaf+1), value=mX['nao_index_cdas_winter_' + str(iLeaf+1)])
        std_const = np.sqrt(np.diag(fourier_terms_const.values @ cov_const.values @ fourier_terms_const.values.T))  
        curve_const = vTheta_const @ fourier_terms_const.values.T
        derivs = pd.concat([fourier_terms_const, fourier_terms_pers], axis=1)
        std_full = np.sqrt(np.diag(derivs.values @ cov.values @ derivs.values.T))  
        
        #confidence bounds
        lower_pers, upper_pers = curve_pers - stats.norm.ppf(1 - alpha / 2) * std_pers[-365:], curve_pers + stats.norm.ppf(1 - alpha / 2) * std_pers[-365:]
        lower_const, upper_const = curve_const - stats.norm.ppf(1 - alpha / 2) * std_const[-365:], curve_const + stats.norm.ppf(1 - alpha / 2) * std_const[-365:]
        return_df = pd.DataFrame([curve_pers, lower_pers, upper_pers, curve_const, lower_const, upper_const, std_full]).T
        return_df.columns=['curve_pers_' + sSetting, 'lower_pers_' + sSetting, 'upper_pers_' + sSetting, 'curve_const_' + sSetting, 'lower_const_' + sSetting, 'upper_const_' + sSetting, 'std_full']
        return_df.index = dfindex
        return return_df
    
    def plot_paths_with_nao(self, year, conf_intervals=True, alpha=0.1, plot=True):
        if type(self.old) == type(None):
            self.prepare_data()
        resultnew, resultold = self.results()
        num_params_pers = self.num_terms_pers * 2 + 1
        num_params_const = self.num_terms_level * 2 + 1
        if plot == True:
            #        Create the figure and axes
            if self.iLeafs >= 4:
                n_cols = 2
                n_rows = (self.iLeafs + 1) // 2  # Ensure enough rows for all leaves
                fig, axs = plt.subplots(n_rows, n_cols, figsize=(12, 6), dpi=100)
                axs = axs.flatten()  # Flatten the grid for easy indexing
            else:
                
                fig, axs = plt.subplots(1, self.iLeafs, figsize=(17,4), dpi=100)
                if self.iLeafs == 1:
                    axs = [axs]  # Ensure axs is a list for consistency
                    
        mCurves_old = pd.DataFrame(np.zeros((365, self.iLeafs)))
        mCurves_new = pd.DataFrame(np.zeros((365, self.iLeafs)))
        mCurves_old_conf_low, mCurves_old_conf_up = pd.DataFrame(np.zeros((365, self.iLeafs))), pd.DataFrame(np.zeros((365, self.iLeafs)))
        mCurves_new_conf_low, mCurves_new_conf_up = pd.DataFrame(np.zeros((365, self.iLeafs))), pd.DataFrame(np.zeros((365, self.iLeafs)))
        if conf_intervals:
            for leaf in range(self.iLeafs):
                old = self.gen_curves(self.old, resultold, 'pers_old_' + str(leaf+1), num_params_pers, num_params_const, leaf, alpha=alpha, period='new')
                new = self.gen_curves(self.new, resultnew, 'pers_new_' + str(leaf+1), num_params_pers, num_params_const, leaf, alpha=alpha)
                curve_old, lower_pers_old, upper_pers_old = old.iloc[:, 0], old.iloc[:, 1], old.iloc[:, 2]
                curve_new, lower_pers_new, upper_pers_new= new.iloc[:, 0], new.iloc[:, 1], new.iloc[:, 2]
                mCurves_old.iloc[:, leaf] = curve_old.values
                mCurves_new.iloc[:, leaf] = curve_new.values
                mCurves_old_conf_low.iloc[:,leaf], mCurves_old_conf_up.iloc[:,leaf] = lower_pers_old.values, upper_pers_old.values
                mCurves_new_conf_low.iloc[:,leaf], mCurves_new_conf_up.iloc[:,leaf] = lower_pers_new.values, upper_pers_new.values
                if plot == True:
                    ax = axs[leaf]
                    if self.iLeafs > 2:
                        add = 'Leaf ' + str(leaf+1)
                    else: 
                        add = 'Negative' if leaf==0 else 'Positive'
                    ax.set_title('(' + str(chr(97 + leaf + 0)) + ')' )
                    ax.plot(curve_new, label='New $\\phi(\\tau)$ path', color='red')
                    ax.plot(curve_old, label='Old $\\phi(\\tau)$ path', color='orange')
                    ax.set_ylabel('$\\phi(\\tau)$')
                    ax.legend()
                    ax.grid(True)
                    if conf_intervals:    
                        ax.fill_between(curve_old.index, lower_pers_old, upper_pers_old, color='orange', alpha=.1)
                        ax.fill_between(curve_new.index, lower_pers_new, upper_pers_new, color='red', alpha=.1)
                    ax.xaxis.set_major_locator(MonthLocator())
                    ax.xaxis.set_major_formatter(DateFormatter('%b'))

        self.mCurves_old_conf_low, self.mCurves_old_conf_up = mCurves_old_conf_low, mCurves_old_conf_up 
        self.mCurves_new_conf_low, self.mCurves_new_conf_up = mCurves_new_conf_low, mCurves_new_conf_up
        mCurves_old.index, mCurves_new.index = curve_old.index, curve_new.index
        self.mCurves_old, self.mCurves_new = mCurves_old, mCurves_new
        if plot == True:            
            plt.tight_layout()
            plt.show()







