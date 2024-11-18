#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 10:30:25 2024

@author: admin
"""

import os
import numpy as np
import statsmodels.api as sm
import pandas as pd
import matplotlib.pyplot as plt
from QAR_persistence_precip import QAR_climate
from sklearn.linear_model import LogisticRegression

climate_class = QAR_climate(sCity='DE BILT', use_statsmodels=True, include_nao=True)
climate_class.prepare_data()
df_new = climate_class.new
df_old = climate_class.old

df_new_winter = df_new.loc[df_new.index.month.isin([12,1,2])]
df_old_winter = df_old.loc[df_old.index.month.isin([12,1,2])]

# Add lagged values of y
df_new_winter_lagged = df_new_winter.shift(1).dropna()
df_old_winter_lagged = df_old_winter.shift(1).dropna()

def create_lagged_features(data, n_lags):
    lagged_data = data.copy()
    for lag in range(1, n_lags + 1):
        lagged_data[f'lag_{lag}_prec'] = lagged_data['Temp'].shift(lag)
        lagged_data[f'lag_{lag}_nao'] = lagged_data['nao_index_cdas'].shift(lag)
    return lagged_data.dropna()

def create_prec_quintiles(data):
    quantiles = [0, .5, .75, .9, 1.0]
    print(np.quantile(data.Temp, quantiles[1:-1]))
    data['prec_quintile'] = pd.qcut(data['Temp'], quantiles, labels=False, duplicates='drop')
    dummies = pd.get_dummies(data['prec_quintile'], prefix='prec_quintile', drop_first=True)
    return pd.concat([data, dummies], axis=1)


def create_prec_value_bins(data, bins):
    """
    Categorizes the Temp values based on specified bins and creates dummy variables.

    Parameters:
    data (pd.DataFrame): The input DataFrame containing Temp values.
    bins (list): The cutoff values for categorizing Temp.

    Returns:
    pd.DataFrame: The DataFrame with added dummy variables for Temp bins.
    """
    data['prec_quintile'] = pd.cut(data['Temp'], bins=bins, labels=False, include_lowest=True)
    dummies = pd.get_dummies(data['prec_quintile'], prefix='prec_quintile', drop_first=True)
    return pd.concat([data, dummies], axis=1)

# Prepare the dataset
n_lags = 1
lagged_data = create_lagged_features(df_new_winter, n_lags)
final_data = create_prec_quintiles(lagged_data)
#final_data = create_prec_value_bins(lagged_data, [0,4, 28, 67, 500])

# Define the response variable and predictors
X = final_data[ list(final_data.filter(like='_quintile').columns[1:])]
y = final_data['prec_quintile'].iloc[1:]

X = sm.add_constant(X.iloc[:-1])


model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
result = model.fit(X,y)

result.predict_proba([[1,0,0,0],[1,1,0,0], [1,0,1,0], [1,0,0,1]])
