#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 21:01:45 2024

@author: admin
"""
import os
os.chdir('/Users/admin/Documents/PhD/persistence')
from QAR import QAR_temperature
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns


test = QAR_temperature(sCity='DE BILT', fTau=.95, use_statsmodels=True, include_nao=True, split_nao=False, iLeafs=1)
test.prepare_data()
quantiles = [.05, .1, .25, .75, .9, .95]
annotation_y = 0.7  # Starting y-position for annotations

def calculate_interquantile_ranges(data, quantiles):
    return {q: (np.quantile(data, q), np.quantile(data, 1 - q)) for q in quantiles}


def filtered_data(test, quant, temp, fix_old):
    old_winter, new_winter = test.old.loc[test.old.index.month.isin([12, 1, 2])], test.new.loc[test.new.index.month.isin([12, 1, 2])]
    
    if temp==True:
        if quant > 0.5:
            old_filtered = old_winter.loc[(old_winter.Temp.shift(1) >= np.quantile(old_winter.Temp, quant))]
            new_filtered = new_winter.loc[(new_winter.Temp.shift(1) >= np.quantile(new_winter.Temp, quant))]
        else: 
            old_filtered = old_winter.loc[(old_winter.Temp.shift(1) <= np.quantile(old_winter.Temp, quant))]
            new_filtered = new_winter.loc[(new_winter.Temp.shift(1) <= np.quantile(new_winter.Temp, quant))]
        
    if temp == False: 
        if quant > 0.5:
            old_filtered = old_winter.loc[(old_winter.nao_index_cdas >= np.quantile(old_winter.nao_index_cdas, quant))]
            new_filtered = new_winter.loc[(new_winter.nao_index_cdas >= np.quantile(new_winter.nao_index_cdas, quant))]
        else: 
            old_filtered = old_winter.loc[(old_winter.nao_index_cdas <= np.quantile(old_winter.nao_index_cdas, quant))]
            new_filtered = new_winter.loc[(new_winter.nao_index_cdas <= np.quantile(new_winter.nao_index_cdas, quant))]
    
    if (fix_old == True) & (temp == False): 
        if quant > 0.5:
            old_filtered = old_winter.loc[(old_winter.nao_index_cdas >= np.quantile(old_winter.nao_index_cdas, .925)) & (old_winter.nao_index_cdas <= np.quantile(old_winter.nao_index_cdas, .975))]
            new_filtered = new_winter.loc[(new_winter.nao_index_cdas >= np.quantile(old_winter.nao_index_cdas, .925)) & (new_winter.nao_index_cdas <= np.quantile(old_winter.nao_index_cdas, .975))]
        else: 
            old_filtered = old_winter.loc[(old_winter.nao_index_cdas <= np.quantile(old_winter.nao_index_cdas, quant)) & (old_winter.nao_index_cdas >= np.quantile(old_winter.nao_index_cdas,0))]
            new_filtered = new_winter.loc[(new_winter.nao_index_cdas <= np.quantile(old_winter.nao_index_cdas, quant)) & (new_winter.nao_index_cdas >= np.quantile(old_winter.nao_index_cdas,0))]
    
    if (fix_old == True) & (temp == True): 
        if quant > 0.5:
            old_filtered = old_winter.loc[(old_winter.Temp.shift(1) >= np.quantile(old_winter.Temp, quant)) & (old_winter.Temp.shift(1) <= np.quantile(old_winter.Temp, .99))]
            new_filtered = new_winter.loc[(new_winter.Temp.shift(1) >= np.quantile(old_winter.Temp, quant)) & (new_winter.Temp.shift(1) <= np.quantile(old_winter.Temp, .99))]
        else: 
            old_filtered = old_winter.loc[(old_winter.Temp.shift(1) <= np.quantile(old_winter.Temp, .075)) & (old_winter.Temp.shift(1) >= np.quantile(old_winter.Temp, 0.025))]
            new_filtered = new_winter.loc[(new_winter.Temp.shift(1) <= np.quantile(old_winter.Temp, .075)) & (new_winter.Temp.shift(1) >= np.quantile(old_winter.Temp, 0.025))]
    
    return old_filtered, new_filtered




# Filter for winter months (December, January, February)
old = test.old.loc[test.old.index.month.isin([12, 1, 2])]
new = test.new.loc[test.new.index.month.isin([12, 1, 2])]

# Calculate the 2.5th and 7.5th quantiles for the lagged old temperature data
shifted_temp_old = old.Temp.shift(1)
q_low = shifted_temp_old.quantile(0.025)
q_high = shifted_temp_old.quantile(0.075)

# Determine size and alpha based on quantiles for old data
old_is_fat = (shifted_temp_old >= q_low) & (shifted_temp_old <= q_high)
old_sizes = np.where(old_is_fat, 3, 1)  # Larger size for "fat" points
old_alphas = np.where(old_is_fat, 1, 0.1)  # Full opacity for "fat" points

# Determine size and alpha based on quantiles for new data (same quantiles as old data)
shifted_temp_new = new.Temp.shift(1)
new_is_fat = (shifted_temp_new >= q_low) & (shifted_temp_new <= q_high)
new_sizes = np.where(new_is_fat, 3, 1)  # Larger size for "fat" points
new_alphas = np.where(new_is_fat, 1.0, 0.1)  # Full opacity for "fat" points

# Create the plot
plt.figure(dpi=100)

# Scatter plot for all old data
plt.scatter(shifted_temp_old, old.Temp, c='orange', s=3, alpha=old_alphas)

# Scatter plot for all new data
plt.scatter(shifted_temp_new, new.Temp, c='red', s=3, alpha=new_alphas)

# Highlight old data points within the quantile range
plt.scatter(shifted_temp_old[old_is_fat], old.Temp[old_is_fat], c='orange', s=3, alpha=1.0, label='Old')

# Highlight new data points within the quantile range
plt.scatter(shifted_temp_new[new_is_fat], new.Temp[new_is_fat], c='red', s=3, alpha=1.0, label='New')

# Add vertical dotted lines for the 2.5th and 7.5th percentiles
plt.axvline(x=q_low, color='orange', linestyle='dotted', label='2.5th percentile (old)')
plt.axvline(x=q_high, color='orange', linestyle='dotted', label='7.5th percentile (old)')

# Add labels and title
plt.xlabel('Lagged Temperature')
plt.ylabel('Temperature')
plt.title('Scatter Plot of Temperatures vs Lagged Temperatures (DJF)')
plt.legend()

plt.show()


####################



# Extract temperatures after filtering
old_filtered, new_filtered = filtered_data(test, .05, True, True)
old_data_q5, new_data_q5 = old_filtered['Temp'], new_filtered['Temp']


old_interquantile_ranges = calculate_interquantile_ranges(old_data_q5, quantiles)
new_interquantile_ranges = calculate_interquantile_ranges(new_data_q5, quantiles)

# Plotting the KDE plots with interquantile range annotations
plt.figure(figsize=(16, 8), dpi=100)  # Larger figure size

# KDE plots
sns.kdeplot(old_data_q5, color='orange', label='Old Data')
sns.kdeplot(new_data_q5, color='red', label='New Data')


for idx, q in enumerate(quantiles[:3]):  # Loop over the first three quantiles for visualization
    old_start, old_end = old_interquantile_ranges[q]
    new_start, new_end = new_interquantile_ranges[q]

    # Adding annotations for old data inside the plot
    plt.text(0.05, annotation_y - idx * 0.1, 
             f'Q{q} - Q{1 - q} OLD = {round(old_end - old_start, 2)}', 
             color='orange', fontsize=12, transform=plt.gca().transAxes)

    # Adding annotations for new data inside the plot
    plt.text(0.05, annotation_y - (idx + 3) * 0.1, 
             f'Q{q} - Q{1 - q} NEW = {round(new_end - new_start, 2)}', 
             color='red', fontsize=12, transform=plt.gca().transAxes)

plt.xlabel('Temperature')
plt.ylabel('Density')
#plt.title('KDE Plot of Temperature with Interquantile Ranges (Conditional on $y_{t-1}\leq Q_{y_{t-1}}(' + str(quant) +')$)')
plt.tight_layout()
#plt.axvline(x=np.quantile(new_winter.Temp, quant), color='red', linestyle='--', alpha=0.5, label='New $Q_{y_{t-1}}(' + str(quant) +')=$' + str(np.round(np.quantile(new_winter.Temp, quant), 2)))  
#plt.axvline(x=np.quantile(old_winter.Temp, quant), color='orange', linestyle='--', alpha=0.5, label='Old $Q_{y_{t-1}}(' + str(quant) +')$=' + str(np.round(np.quantile(old_winter.Temp, quant), 2)))
plt.legend(loc='upper right')  # Adjust legend position
plt.show()



####################




# Filter for winter months (December, January, February)
old = test.old.loc[test.old.index.month.isin([12, 1, 2])]
new = test.new.loc[test.new.index.month.isin([12, 1, 2])]

# Calculate the 2.5th and 7.5th quantiles for the lagged old temperature data
shifted_temp_old = old.Temp.shift(1)
q_high_old = shifted_temp_old.quantile(0.05)

shifted_temp_new = new.Temp.shift(1)
q_high_new = shifted_temp_new.quantile(0.05)

# Determine size and alpha based on quantiles for old data
old_is_fat = shifted_temp_old <= q_high_old
old_sizes = np.where(old_is_fat, 3, 1)  # Larger size for "fat" points
old_alphas = np.where(old_is_fat, 1, 0.1)  # Full opacity for "fat" points

# Determine size and alpha based on quantiles for new data (same quantiles as old data)
shifted_temp_new = new.Temp.shift(1)
new_is_fat = shifted_temp_new <= q_high_new
new_sizes = np.where(new_is_fat, 3, 1)  # Larger size for "fat" points
new_alphas = np.where(new_is_fat, 1.0, 0.1)  # Full opacity for "fat" points

# Create the plot
plt.figure(dpi=100)

# Scatter plot for all old data
plt.scatter(shifted_temp_old, old.Temp, c='orange', s=3, alpha=old_alphas)

# Scatter plot for all new data
plt.scatter(shifted_temp_new, new.Temp, c='red', s=3, alpha=new_alphas)

# Highlight old data points within the quantile range
plt.scatter(shifted_temp_old[old_is_fat], old.Temp[old_is_fat], c='orange', s=3, alpha=1.0, label='Old')

# Highlight new data points within the quantile range
plt.scatter(shifted_temp_new[new_is_fat], new.Temp[new_is_fat], c='red', s=3, alpha=1.0, label='New')

# Add vertical dotted lines for the 2.5th and 7.5th percentiles
plt.axvline(x=q_high_old, color='orange', linestyle='dotted', label='5th percentile (old)')
plt.axvline(x=q_high_new, color='red', linestyle='dotted', label='5th percentile (new)')

# Add labels and title
plt.xlabel('Lagged NAO Index Value')
plt.ylabel('Temperature')
#plt.title('Scatter Plot of Temperatures vs Lagged Temperatures (DJF)')
plt.legend()

plt.show()


####################



# Extract temperatures after filtering
old_filtered, new_filtered = filtered_data(test, .05, True, False)
old_data_q5, new_data_q5 = old_filtered['Temp'], new_filtered['Temp']


old_interquantile_ranges = calculate_interquantile_ranges(old_data_q5, quantiles)
new_interquantile_ranges = calculate_interquantile_ranges(new_data_q5, quantiles)

# Plotting the KDE plots with interquantile range annotations
plt.figure(figsize=(16, 8), dpi=100)  # Larger figure size

# KDE plots
sns.kdeplot(old_data_q5, color='orange', label='Old Data')
sns.kdeplot(new_data_q5, color='red', label='New Data')


for idx, q in enumerate(quantiles[:3]):  # Loop over the first three quantiles for visualization
    old_start, old_end = old_interquantile_ranges[q]
    new_start, new_end = new_interquantile_ranges[q]

    # Adding annotations for old data inside the plot
    plt.text(0.05, annotation_y - idx * 0.1, 
             f'Q{q} - Q{1 - q} OLD = {round(old_end - old_start, 2)}', 
             color='orange', fontsize=12, transform=plt.gca().transAxes)

    # Adding annotations for new data inside the plot
    plt.text(0.05, annotation_y - (idx + 3) * 0.1, 
             f'Q{q} - Q{1 - q} NEW = {round(new_end - new_start, 2)}', 
             color='red', fontsize=12, transform=plt.gca().transAxes)

plt.xlabel('Temperature')
plt.ylabel('Density')
#plt.title('KDE Plot of Temperature with Interquantile Ranges (Conditional on $y_{t-1}\leq Q_{y_{t-1}}(' + str(quant) +')$)')
plt.tight_layout()
#plt.axvline(x=np.quantile(new_winter.Temp, quant), color='red', linestyle='--', alpha=0.5, label='New $Q_{y_{t-1}}(' + str(quant) +')=$' + str(np.round(np.quantile(new_winter.Temp, quant), 2)))  
#plt.axvline(x=np.quantile(old_winter.Temp, quant), color='orange', linestyle='--', alpha=0.5, label='Old $Q_{y_{t-1}}(' + str(quant) +')$=' + str(np.round(np.quantile(old_winter.Temp, quant), 2)))
plt.legend(loc='upper right')  # Adjust legend position
plt.show()




####################



# Filter for winter months (December, January, February)
old = test.old.loc[test.old.index.month.isin([12, 1, 2])]
new = test.new.loc[test.new.index.month.isin([12, 1, 2])]

# Calculate the 2.5th and 7.5th quantiles for the lagged old temperature data
shifted_temp_old = old.nao_index_cdas.shift(1)
q_low_old = shifted_temp_old.quantile(0.95)

shifted_temp_new = new.nao_index_cdas.shift(1)
q_low_new = shifted_temp_new.quantile(0.95)

# Determine size and alpha based on quantiles for old data
old_is_fat = shifted_temp_old >= q_low_old
old_sizes = np.where(old_is_fat, 3, 1)  # Larger size for "fat" points
old_alphas = np.where(old_is_fat, 1, 0.1)  # Full opacity for "fat" points

# Determine size and alpha based on quantiles for new data (same quantiles as old data)
shifted_temp_new = new.nao_index_cdas.shift(1)
new_is_fat = shifted_temp_new >= q_low_new
new_sizes = np.where(new_is_fat, 3, 1)  # Larger size for "fat" points
new_alphas = np.where(new_is_fat, 1.0, 0.1)  # Full opacity for "fat" points

# Create the plot
plt.figure(dpi=100)

# Scatter plot for all old data
plt.scatter(shifted_temp_old, old.Temp, c='orange', s=3, alpha=old_alphas)

# Scatter plot for all new data
plt.scatter(shifted_temp_new, new.Temp, c='red', s=3, alpha=new_alphas)

# Highlight old data points within the quantile range
plt.scatter(shifted_temp_old[old_is_fat], old.Temp[old_is_fat], c='orange', s=3, alpha=1.0, label='Old')

# Highlight new data points within the quantile range
plt.scatter(shifted_temp_new[new_is_fat], new.Temp[new_is_fat], c='red', s=3, alpha=1.0, label='New')

# Add vertical dotted lines for the 2.5th and 7.5th percentiles
plt.axvline(x=q_low_old, color='orange', linestyle='dotted', label='95th percentile (old)')
plt.axvline(x=q_low_new, color='red', linestyle='dotted', label='95th percentile (new)')

# Add labels and title
plt.xlabel('Lagged NAO Index Value')
plt.ylabel('Temperature')
#plt.title('Scatter Plot of Temperatures vs Lagged Temperatures (DJF)')
plt.legend()

plt.show()

####################



# Extract temperatures after filtering
old_filtered, new_filtered = filtered_data(test, .95, False, False)
old_data_q5, new_data_q5 = old_filtered['Temp'], new_filtered['Temp']


old_interquantile_ranges = calculate_interquantile_ranges(old_data_q5, quantiles)
new_interquantile_ranges = calculate_interquantile_ranges(new_data_q5, quantiles)

# Plotting the KDE plots with interquantile range annotations
plt.figure(figsize=(16, 8), dpi=100)  # Larger figure size

# KDE plots
sns.kdeplot(old_data_q5, color='orange', label='Old Data')
sns.kdeplot(new_data_q5, color='red', label='New Data')


for idx, q in enumerate(quantiles[:3]):  # Loop over the first three quantiles for visualization
    old_start, old_end = old_interquantile_ranges[q]
    new_start, new_end = new_interquantile_ranges[q]

    # Adding annotations for old data inside the plot
    plt.text(0.05, annotation_y - idx * 0.1, 
             f'Q{q} - Q{1 - q} OLD = {round(old_end - old_start, 2)}', 
             color='orange', fontsize=12, transform=plt.gca().transAxes)

    # Adding annotations for new data inside the plot
    plt.text(0.05, annotation_y - (idx + 3) * 0.1, 
             f'Q{q} - Q{1 - q} NEW = {round(new_end - new_start, 2)}', 
             color='red', fontsize=12, transform=plt.gca().transAxes)

plt.xlabel('Temperature')
plt.ylabel('Density')
#plt.title('KDE Plot of Temperature with Interquantile Ranges (Conditional on $y_{t-1}\leq Q_{y_{t-1}}(' + str(quant) +')$)')
plt.tight_layout()
#plt.axvline(x=np.quantile(new_winter.Temp, quant), color='red', linestyle='--', alpha=0.5, label='New $Q_{y_{t-1}}(' + str(quant) +')=$' + str(np.round(np.quantile(new_winter.Temp, quant), 2)))  
#plt.axvline(x=np.quantile(old_winter.Temp, quant), color='orange', linestyle='--', alpha=0.5, label='Old $Q_{y_{t-1}}(' + str(quant) +')$=' + str(np.round(np.quantile(old_winter.Temp, quant), 2)))
plt.legend(loc='upper right')  # Adjust legend position
plt.show()




####################



# Filter for winter months (December, January, February)
old = test.old.loc[test.old.index.month.isin([12, 1, 2])]
new = test.new.loc[test.new.index.month.isin([12, 1, 2])]

# Calculate the 2.5th and 7.5th quantiles for the lagged old temperature data
shifted_temp_old = old.nao_index_cdas.shift(1)
q_low = shifted_temp_old.quantile(0.925)
q_high = shifted_temp_old.quantile(0.975)

# Determine size and alpha based on quantiles for old data
old_is_fat = (shifted_temp_old >= q_low) & (shifted_temp_old <= q_high)
old_sizes = np.where(old_is_fat, 3, 1)  # Larger size for "fat" points
old_alphas = np.where(old_is_fat, 1, 0.1)  # Full opacity for "fat" points

# Determine size and alpha based on quantiles for new data (same quantiles as old data)
shifted_temp_new = new.nao_index_cdas.shift(1)
new_is_fat = (shifted_temp_new >= q_low) & (shifted_temp_new <= q_high)
new_sizes = np.where(new_is_fat, 3, 1)  # Larger size for "fat" points
new_alphas = np.where(new_is_fat, 1.0, 0.1)  # Full opacity for "fat" points

# Create the plot
plt.figure(dpi=100)

# Scatter plot for all old data
plt.scatter(shifted_temp_old, old.Temp, c='orange', s=3, alpha=old_alphas)

# Scatter plot for all new data
plt.scatter(shifted_temp_new, new.Temp, c='red', s=3, alpha=new_alphas)

# Highlight old data points within the quantile range
plt.scatter(shifted_temp_old[old_is_fat], old.Temp[old_is_fat], c='orange', s=3, alpha=1.0, label='Old')

# Highlight new data points within the quantile range
plt.scatter(shifted_temp_new[new_is_fat], new.Temp[new_is_fat], c='red', s=3, alpha=1.0, label='New')

# Add vertical dotted lines for the 2.5th and 7.5th percentiles
plt.axvline(x=q_low, color='orange', linestyle='dotted', label='2.5th percentile (old)')
plt.axvline(x=q_high, color='orange', linestyle='dotted', label='7.5th percentile (old)')

# Add labels and title
plt.xlabel('Lagged NAO Index Value')
plt.ylabel('Temperature')
#plt.title('Scatter Plot of Temperatures vs Lagged Temperatures (DJF)')
plt.legend()

plt.show()


####################

# Extract temperatures after filtering
old_filtered, new_filtered = filtered_data(test, .95, False, True)
old_data_q5, new_data_q5 = old_filtered['Temp'], new_filtered['Temp']


old_interquantile_ranges = calculate_interquantile_ranges(old_data_q5, quantiles)
new_interquantile_ranges = calculate_interquantile_ranges(new_data_q5, quantiles)

# Plotting the KDE plots with interquantile range annotations
plt.figure(figsize=(16, 8), dpi=100)  # Larger figure size

# KDE plots
sns.kdeplot(old_data_q5, color='orange', label='Old Data')
sns.kdeplot(new_data_q5, color='red', label='New Data')


for idx, q in enumerate(quantiles[:3]):  # Loop over the first three quantiles for visualization
    old_start, old_end = old_interquantile_ranges[q]
    new_start, new_end = new_interquantile_ranges[q]

    # Adding annotations for old data inside the plot
    plt.text(0.05, annotation_y - idx * 0.1, 
             f'Q{q} - Q{1 - q} OLD = {round(old_end - old_start, 2)}', 
             color='orange', fontsize=12, transform=plt.gca().transAxes)

    # Adding annotations for new data inside the plot
    plt.text(0.05, annotation_y - (idx + 3) * 0.1, 
             f'Q{q} - Q{1 - q} NEW = {round(new_end - new_start, 2)}', 
             color='red', fontsize=12, transform=plt.gca().transAxes)

plt.xlabel('Temperature')
plt.ylabel('Density')
#plt.title('KDE Plot of Temperature with Interquantile Ranges (Conditional on $y_{t-1}\leq Q_{y_{t-1}}(' + str(quant) +')$)')
plt.tight_layout()
#plt.axvline(x=np.quantile(new_winter.Temp, quant), color='red', linestyle='--', alpha=0.5, label='New $Q_{y_{t-1}}(' + str(quant) +')=$' + str(np.round(np.quantile(new_winter.Temp, quant), 2)))  
#plt.axvline(x=np.quantile(old_winter.Temp, quant), color='orange', linestyle='--', alpha=0.5, label='Old $Q_{y_{t-1}}(' + str(quant) +')$=' + str(np.round(np.quantile(old_winter.Temp, quant), 2)))
plt.legend(loc='upper right')  # Adjust legend position
plt.show()

############
############
############
############
############



# Assuming the test object and filtering functions are already defined as in your code
test = QAR_temperature(sCity='DE BILT', fTau=.95, use_statsmodels=True, include_nao=True, split_nao=False, iLeafs=1)
test.prepare_data()
quantiles = [.05, .1, .25, .75, .9, .95]
annotation_y = 0.7  # Starting y-position for annotations

# Initialize a 4x2 grid
fig, axs = plt.subplots(4, 2, figsize=(16, 20), dpi=100)
plt.subplots_adjust(hspace=0.25, wspace=0.15)  # Adjust spacing between subplots

# Filter for winter months
old = test.old.loc[test.old.index.month.isin([12, 1, 2])]
new = test.new.loc[test.new.index.month.isin([12, 1, 2])]

# Plot 1: Scatter plot for temperature vs lagged temperature
shifted_temp_old = old.Temp.shift(1)
shifted_temp_new = new.Temp.shift(1)
q_low = shifted_temp_old.quantile(0.025)
q_high = shifted_temp_old.quantile(0.075)

# Determine size and alpha based on quantiles for old and new data
old_is_fat = (shifted_temp_old >= q_low) & (shifted_temp_old <= q_high)
new_is_fat = (shifted_temp_new >= q_low) & (shifted_temp_new <= q_high)

# Plot old and new data, highlighting "fat" points
axs[0, 0].scatter(shifted_temp_old, old.Temp, c='orange', s=1, alpha=0.1)
axs[0, 0].scatter(shifted_temp_old[old_is_fat], old.Temp[old_is_fat], c='orange', s=3, alpha=1, label='Old')
axs[0, 0].scatter(shifted_temp_new, new.Temp, c='red', s=1, alpha=0.1)
axs[0, 0].scatter(shifted_temp_new[new_is_fat], new.Temp[new_is_fat], c='red', s=3, alpha=1, label='New')
axs[0, 0].axvline(x=q_low, color='orange', linestyle='dotted', label='2.5th percentile (old)')
axs[0, 0].axvline(x=q_high, color='orange', linestyle='dotted', label='7.5th percentile (old)')
axs[0, 0].set_xlabel('Lagged Temperature')
axs[0, 0].set_ylabel('Temperature')
axs[0, 0].set_title('(a)')
axs[0, 0].legend()

# Plot 2: KDE Plot with interquantile ranges
old_filtered, new_filtered = filtered_data(test, .05, True, True)
old_data_q5, new_data_q5 = old_filtered['Temp'], new_filtered['Temp']
sns.kdeplot(old_data_q5, color='orange', ax=axs[0, 1], label='Old')
sns.kdeplot(new_data_q5, color='red', ax=axs[0, 1], label='New')
axs[0, 1].set_xlabel('Temperature')
axs[0, 1].set_ylabel('Density')
axs[0, 1].set_title('(b)')
axs[0, 1].legend()

# Plot 3: Scatter plot for temperature vs lagged NAO index (5th percentile)
q_high_old = shifted_temp_old.quantile(0.05)
q_high_new = shifted_temp_new.quantile(0.05)

# Determine size and alpha based on quantiles for old and new data (5th percentile)
old_is_fat = shifted_temp_old <= q_high_old
new_is_fat = shifted_temp_new <= q_high_new

# Plot old and new data, highlighting "fat" points
axs[1, 0].scatter(shifted_temp_old, old.Temp, c='orange', s=1, alpha=0.1)
axs[1, 0].scatter(shifted_temp_old[old_is_fat], old.Temp[old_is_fat], c='orange', s=3, alpha=1, label='Old')
axs[1, 0].scatter(shifted_temp_new, new.Temp, c='red', s=1, alpha=0.1)
axs[1, 0].scatter(shifted_temp_new[new_is_fat], new.Temp[new_is_fat], c='red', s=3, alpha=1, label='New')
axs[1, 0].axvline(x=q_high_old, color='orange', linestyle='dotted', label='5th percentile (old)')
axs[1, 0].axvline(x=q_high_new, color='red', linestyle='dotted', label='5th percentile (new)')
axs[1, 0].set_xlabel('Lagged Temperature')
axs[1, 0].set_ylabel('Temperature')
axs[1, 0].set_title('(c)')
axs[1, 0].legend()

# Plot 4: KDE plot with interquantile ranges (5th percentile)
old_filtered, new_filtered = filtered_data(test, .05, True, False)
old_data_q5, new_data_q5 = old_filtered['Temp'], new_filtered['Temp']
sns.kdeplot(old_data_q5, color='orange', ax=axs[1, 1], label='Old')
sns.kdeplot(new_data_q5, color='red', ax=axs[1, 1], label='New')
axs[1, 1].set_xlabel('Temperature')
axs[1, 1].set_ylabel('Density')
axs[1, 1].set_title('(d)')
axs[1, 1].legend()


# Plot 5: Scatter plot for temperatures (2.5th and 7.5th percentiles of NAO index)
shifted_temp_new = new.nao_index_cdas.shift(1)
shifted_temp_old = old.nao_index_cdas.shift(1)
q_low = shifted_temp_old.quantile(0.925)
q_high = shifted_temp_old.quantile(0.975)

# Determine size and alpha based on quantiles for old and new data (2.5th and 7.5th percentiles)
old_is_fat = (shifted_temp_old >= q_low) & (shifted_temp_old <= q_high)
new_is_fat = (shifted_temp_new >= q_low) & (shifted_temp_new <= q_high)

# Plot old and new data, highlighting "fat" points
axs[2, 0].scatter(shifted_temp_old, old.Temp, c='orange', s=1, alpha=0.1)
axs[2, 0].scatter(shifted_temp_old[old_is_fat], old.Temp[old_is_fat], c='orange', s=3, alpha=1, label='Old')
axs[2, 0].scatter(shifted_temp_new, new.Temp, c='red', s=1, alpha=0.1)
axs[2, 0].scatter(shifted_temp_new[new_is_fat], new.Temp[new_is_fat], c='red', s=3, alpha=1, label='New')
axs[2, 0].axvline(x=q_low, color='orange', linestyle='dotted', label='92.5th percentile (old)')
axs[2, 0].axvline(x=q_high, color='orange', linestyle='dotted', label='97.5th percentile (old)')
axs[2, 0].set_xlabel('Lagged NAO Index Value')
axs[2, 0].set_ylabel('Temperature')
axs[2, 0].set_title('(e)')
axs[2, 0].legend()

# Plot 6: KDE plot with interquantile ranges (2.5th and 7.5th percentiles)
old_filtered, new_filtered = filtered_data(test, .95, False, True)
old_data_q5, new_data_q5 = old_filtered['Temp'], new_filtered['Temp']
sns.kdeplot(old_data_q5, color='orange', ax=axs[2, 1], label='Old')
sns.kdeplot(new_data_q5, color='red', ax=axs[2, 1], label='New')
axs[2, 1].set_xlabel('Temperature')
axs[2, 1].set_ylabel('Density')
axs[2, 1].set_title('(f)')
axs[2, 1].legend()


# Plot 7: Scatter plot for temperature vs lagged NAO index (95th percentile)
shifted_temp_old = old.nao_index_cdas.shift(1)
q_low_old = shifted_temp_old.quantile(0.95)
shifted_temp_new = new.nao_index_cdas.shift(1)
q_low_new = shifted_temp_new.quantile(0.95)

# Determine size and alpha based on quantiles for old and new data (95th percentile)
old_is_fat = shifted_temp_old >= q_low_old
new_is_fat = shifted_temp_new >= q_low_new

# Plot old and new data, highlighting "fat" points
axs[3, 0].scatter(shifted_temp_old, old.Temp, c='orange', s=1, alpha=0.1)
axs[3, 0].scatter(shifted_temp_old[old_is_fat], old.Temp[old_is_fat], c='orange', s=3, alpha=1, label='Old')
axs[3, 0].scatter(shifted_temp_new, new.Temp, c='red', s=1, alpha=0.1)
axs[3, 0].scatter(shifted_temp_new[new_is_fat], new.Temp[new_is_fat], c='red', s=3, alpha=1, label='New')
axs[3, 0].axvline(x=q_low_old, color='orange', linestyle='dotted', label='95th percentile (old)')
axs[3, 0].axvline(x=q_low_new, color='red', linestyle='dotted', label='95th percentile (new)')
axs[3, 0].set_xlabel('Lagged NAO Index Value')
axs[3, 0].set_ylabel('Temperature')
axs[3, 0].set_title('(g)')
axs[3, 0].legend()

# Plot 8: KDE plot with interquantile ranges (95th percentile)
old_filtered, new_filtered = filtered_data(test, .95, False, False)
old_data_q5, new_data_q5 = old_filtered['Temp'], new_filtered['Temp']
sns.kdeplot(old_data_q5, color='orange', ax=axs[3, 1], label='Old')
sns.kdeplot(new_data_q5, color='red', ax=axs[3, 1], label='New')
axs[3, 1].set_xlabel('Temperature')
axs[3, 1].set_ylabel('Density')
axs[3, 1].set_title('(h)')
axs[3, 1].legend()
# Show the combined plot
plt.show()


