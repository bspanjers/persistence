#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 11:01:44 2024

@author: admin
"""


################# plot 1 ##########

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
import matplotlib.patches as mpatches  # For colored patches in the legend
from QAR_persistence_precip import QAR_precipitation
from QAR import QAR_temperature
import seaborn as sns
import pandas as pd
from plots_europe import plot_combined
import stats
from prec_persistence import get_season, fit_ar_logistic_regression, extract_results, plot_coefficients, first_one_index

#### FIGURE 1
# Setup your cities
cities = ['NOTTINGHAM: WATNALL', 'DE BILT', 'ABBEVILLE', 'HETTSTEDT']
start, end = '2023-11-30', '2024-03-01'
# Initialize test objects for each city
test_objects = [QAR_precipitation(sCity=city, fTau=.95, use_statsmodels=True, include_nao=True, oldstart='1990-', oldend='2020-',newend='2025-') for city in cities]

# Prepare data for each city
for test in test_objects:
    test.prepare_data()

# Setup your figure
fig, ax = plt.subplots(dpi=200, figsize=(10, 6))

# Example NAO data for one city (can be used as a reference for x-axis)
test = test_objects[0]
test.new.nao_index_cdas.loc[(test.new.index > start) & (test.new.index < end)].plot(ax=ax, label='NAO index')

# Add lines for y=0 and the 80th percentile
plt.axhline(y=0, color='grey', linestyle='-', linewidth=0.5)
plt.axhline(y=0.95, color='red', linestyle='--', linewidth=1, label='80th percentile NAO')

# Add vertical line for a specific event (e.g., Storm Henk)
plt.axvline(x='2024-01-02', color='black', linestyle='--', linewidth=1, label='Storm Henk')

# Define vertical spacing for each city's rainfall
y_min, y_max = ax.get_ylim()
spacing = 0.1  # Small vertical space between cities
y_ranges = np.linspace(y_min, y_max - len(cities) * spacing, len(cities) + 1)  # Adjust y ranges

# Colors for each city
colors = ['blue', 'green', 'yellow', 'purple']

# Iterate through each city and plot the respective filled area with more confident colors
for i, test in enumerate(test_objects):
    dates = test.new.loc[(test.new.index > start) & (test.new.index < end)].index
    temp_above_5 = test.new.Temp.loc[(test.new.index > start) & (test.new.index < end)] > 5  # Condition when temperature > 5
    mpl_dates = mdates.date2num(dates)

    # Adjust y-ranges to include small vertical spaces
    y_bottom = y_ranges[i] + i * spacing  # Adding space between cities
    y_top = y_ranges[i + 1] + i * spacing

    # Increase the width of the shading by adjusting the start and end of the rain periods
    for j, is_rainy in enumerate(temp_above_5):
        if is_rainy:
            date_start = mpl_dates[j] - 0.5  # Start a little earlier
            date_end = mpl_dates[j] + 0.5  # End a little later
            ax.fill_betweenx([y_bottom, y_top], date_start, date_end, color=colors[i], alpha=0.5, edgecolor=None)

# Create a custom legend with colors
patches = [mpatches.Patch(color=colors[i], alpha=0.5, label=cities[i]) for i in range(len(cities))]

# Add the legend below the plot, including NAO line, 80th percentile line, and Storm Henk
plt.legend(handles=[plt.Line2D([0], [0], color='blue', label='NAO index'),
                    plt.Line2D([0], [0], color='red', linestyle='--', label='80th percentile NAO'),
                    plt.Line2D([0], [0], color='black', linestyle='--', label='Storm Henk')] + patches,
           loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4)

# Add y-axis label
ax.set_ylabel('NAO index value')

# Adjust layout to prevent cutting off labels
plt.tight_layout()

# Show the plot
plt.show()

######### FIGURE 2 ########

test = QAR_temperature(sCity='DE BILT', fTau=0.05, use_statsmodels=True, include_nao=False, split_nao=False, iLeafs=1)
test.plot_fourier_fit_full(vTau=[0.05, 0.5, .95], alpha=0.05)





###### FIGURE 3 #######
#load data
df_05 = pd.read_csv('/Users/admin/Documents/PhD/persistence/data_persistence/results_05_1950.csv')
df_50 = pd.read_csv('/Users/admin/Documents/PhD/persistence/data_persistence/results_.5_1950.csv')
df_50.columns = ['STANAME', 'STAID', 'latitude', 'longitude', 'mean_diff_winter',
       'mean_diff_spring', 'mean_diff_summer', 'mean_diff_autumn',
       'mean_diff_pers_winter', 'mean_diff_pers_spring',
       'mean_diff_pers_summer', 'mean_diff_pers_autumn']
df_95 = pd.read_csv('/Users/admin/Documents/PhD/persistence/data_persistence/results_95_1950.csv')
df_05_plusmin = pd.read_csv('/Users/admin/Documents/PhD/persistence/data_persistence/results_05_19502leafs.csv')
df_50_plusmin = pd.read_csv('/Users/admin/Documents/PhD/persistence/data_persistence/results_.5_19502leafs.csv')
df_95_plusmin = pd.read_csv('/Users/admin/Documents/PhD/persistence/data_persistence/results_95_19502leafs.csv')
df_results_list = [df_05, df_50, df_95, df_05_plusmin, df_50_plusmin, df_95_plusmin]

#plot Fig 3
plot_combined(df_results_list, 'winter')




##### FIGURE 4 #######
from plots_precip import plot_heatmap_precip
df = pd.read_csv('/Users/admin/Documents/PhD/persistence/data_persistence/results_precipitation_1950WithUncProbabilities.csv')
plot_heatmap_precip(df, 'winter', sType='')
plot_heatmap_precip(df, 'winter', sType='_unc')





######## FIGURE 5 (APPENDIX) #########

test = QAR_temperature(sCity='DE BILT', fTau=.05, use_statsmodels=True, include_nao=True, split_nao=True, iLeafs=2)
test.plot_paths_with_nao(2019)
test = QAR_temperature(sCity='DE BILT', fTau=.5, use_statsmodels=True, include_nao=True, split_nao=True, iLeafs=2)
test.plot_paths_with_nao(2019)
test = QAR_temperature(sCity='DE BILT', fTau=.95, use_statsmodels=True, include_nao=True, split_nao=True, iLeafs=2)
test.plot_paths_with_nao(2019)


########## FIGURE 6
import numpy as np
import matplotlib.pyplot as plt

# Initialize lists to store values for both minus (0) and plus (1) versions
l_coefs_005_minus, l_conf_low_005_minus, l_conf_up_005_minus, upper_quintile_values_005_minus = [], [], [], []
l_coefs_005_plus, l_conf_low_005_plus, l_conf_up_005_plus, upper_quintile_values_005_plus = [], [], [], []

l_coefs_05_minus, l_conf_low_05_minus, l_conf_up_05_minus, upper_quintile_values_05_minus = [], [], [], []
l_coefs_05_plus, l_conf_low_05_plus, l_conf_up_05_plus, upper_quintile_values_05_plus = [], [], [], []

l_coefs_095_minus, l_conf_low_095_minus, l_conf_up_095_minus, upper_quintile_values_095_minus = [], [], [], []
l_coefs_095_plus, l_conf_low_095_plus, l_conf_up_095_plus, upper_quintile_values_095_plus = [], [], [], []

x = np.arange(0, 41, 1)  # Create the range for year windows

# Iterate over year windows to calculate the coefficients and confidence bounds for each fTau
for i in x:
    yearstart, yearend = 1950 + i, 1980 + i
    print(f'\rCurrently calculating coefficients for {yearstart}-{yearend}', end='')

    for fTau, l_coefs_minus, l_conf_low_minus, l_conf_up_minus, upper_quintile_values_minus, \
             l_coefs_plus, l_conf_low_plus, l_conf_up_plus, upper_quintile_values_plus in [
            (0.05, l_coefs_005_minus, l_conf_low_005_minus, l_conf_up_005_minus, upper_quintile_values_005_minus, 
             l_coefs_005_plus, l_conf_low_005_plus, l_conf_up_005_plus, upper_quintile_values_005_plus), 
            (0.5, l_coefs_05_minus, l_conf_low_05_minus, l_conf_up_05_minus, upper_quintile_values_05_minus, 
             l_coefs_05_plus, l_conf_low_05_plus, l_conf_up_05_plus, upper_quintile_values_05_plus), 
            (0.95, l_coefs_095_minus, l_conf_low_095_minus, l_conf_up_095_minus, upper_quintile_values_095_minus, 
             l_coefs_095_plus, l_conf_low_095_plus, l_conf_up_095_plus, upper_quintile_values_095_plus)]:

        # Perform the analysis for the minus version
        test = QAR_temperature(sCity='DE BILT', fTau=fTau, use_statsmodels=True, include_nao=True,split_nao=True, oldstart=str(yearstart)+'-', oldend=str(yearend)+'-')
        test.plot_paths_with_nao(2019,plot=False, alpha=0.05)
        l_coefs_minus.append(test.mCurves_old.iloc[0])  # 'minus' version
        l_conf_low_minus.append(test.mCurves_old_conf_low.iloc[0])
        l_conf_up_minus.append(test.mCurves_old_conf_up.iloc[0])

        # Perform the analysis for the plus version
        l_coefs_plus.append(test.mCurves_old.iloc[0])  # 'plus' version
        l_conf_low_plus.append(test.mCurves_old_conf_low.iloc[0])
        l_conf_up_plus.append(test.mCurves_old_conf_up.iloc[0])

        # NAO index data (same for both plus and minus)
        test = QAR_temperature(sCity='DE BILT', fTau=fTau, use_statsmodels=True, include_nao=True, oldstart=str(yearstart)+'-', oldend=str(yearend)+'-')
        test.prepare_data()
        data_old = test.old.nao_index_cdas
        data_winter_old = data_old.loc[data_old.index.month.isin([12, 1, 2])]  # Use winter months (DJF)

        # Calculate the upper quintile of the NAO index in the current 30-year window
        upper_quintile_nao = np.quantile(data_winter_old, 0.8)
        upper_quintile_values_minus.append(upper_quintile_nao)
        upper_quintile_values_plus.append(upper_quintile_nao)

fig, axs = plt.subplots(2, 3, figsize=(18, 12), sharex=True, sharey=False, dpi=200)

# Define subplot labels and titles for each plot
subplot_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
titles = [
    "$\\phi_t(0.05), s=$NAO$-$", "$\\phi_t(0.5), s=$NAO$-$", "$\\phi_t(0.95), s=$NAO$-$",
    "$\\phi_t(0.05), s=$NAO$+$", "$\\phi_t(0.5), s=$NAO$+$", "$\\phi_t(0.95), s=$NAO$+$"
]

# Initialize lists to capture handles and labels for the legend
handles_list = []
labels_list = []

for row, (l_coefs_minus, l_conf_low_minus, l_conf_up_minus, upper_quintile_values_minus, 
          l_coefs_plus, l_conf_low_plus, l_conf_up_plus, upper_quintile_values_plus, fTau) in enumerate([
    (l_coefs_005_minus, l_conf_low_005_minus, l_conf_up_005_minus, upper_quintile_values_005_minus, 
     l_coefs_005_plus, l_conf_low_005_plus, l_conf_up_005_plus, upper_quintile_values_005_plus, 0.05),
    (l_coefs_05_minus, l_conf_low_05_minus, l_conf_up_05_minus, upper_quintile_values_05_minus, 
     l_coefs_05_plus, l_conf_low_05_plus, l_conf_up_05_plus, upper_quintile_values_05_plus, 0.5),
    (l_coefs_095_minus, l_conf_low_095_minus, l_conf_up_095_minus, upper_quintile_values_095_minus, 
     l_coefs_095_plus, l_conf_low_095_plus, l_conf_up_095_plus, upper_quintile_values_095_plus, 0.95)]):
    
    # Plot minus version
    ax = axs[0, row]
    line1, = ax.plot(x, l_coefs_minus, marker='o', linestyle='-', color='blue', label=f'Coefficients (minus, fTau={fTau})')
    ax.fill_between(x, [l_conf_low_minus[i][0] for i in range(len(l_conf_low_plus))], [l_conf_up_minus[i][0] for i in range(len(l_conf_low_plus))], color='blue', alpha=0.3, label='95% Confidence Interval')
    ax.set_xlabel('End year of 30-year window')
    ax.set_ylabel('Coefficient Value', color='black', rotation=90)
    ax.grid(True)
    
    # Add the title and label for the minus version
    ax.set_title(f'{subplot_labels[row]} {titles[row]}')
    
    ax.tick_params(axis='y', colors='blue')
    ax2 = ax.twinx()
    line2, = ax2.plot(x, upper_quintile_values_minus, marker='s', linestyle='--', color='red', label='80$^{th}$ Percentile of the NAO Index')
    ax2.set_ylabel('NAO Index Value', color='black', rotation=90)
    ax2.tick_params(axis='y', colors='red')
    
    # Capture handles for the legend
    handles_list.append(line1)
    handles_list.append(line2)
    labels_list.append(f'$\\phi_t({fTau})$, $s=$NAO$-$')
    labels_list.append('80$^{th}$ Percentile NAO Index')
    
    # Plot plus version
    ax = axs[1, row]
    line1, = ax.plot(x, l_coefs_plus, marker='o', linestyle='-', color='blue', label=f'Coefficients (plus, fTau={fTau})')
    ax.fill_between(x, [l_conf_low_plus[i][0] for i in range(len(l_conf_low_plus))], [l_conf_up_plus[i][0] for i in range(len(l_conf_low_plus))], color='blue', alpha=0.3, label='95% Confidence Interval')
    ax.set_xlabel('End year of 30-year window')
    ax.set_ylabel('Coefficient Value', color='black', rotation=90)
    ax.grid(True)
    
    # Add the title and label for the plus version
    ax.set_title(f'{subplot_labels[row + 3]} {titles[row + 3]}')
    
    ax.tick_params(axis='y', colors='blue')
    ax2 = ax.twinx()
    line2, = ax2.plot(x, upper_quintile_values_plus, marker='s', linestyle='--', color='red', label='80$^{th}$ Percentile of the NAO Index')
    ax2.set_ylabel('NAO Index Value', color='black', rotation=90)
    ax2.tick_params(axis='y', colors='red')
    
    # Capture handles for the legend
    handles_list.append(line1)
    handles_list.append(line2)
    labels_list.append(f'$\\phi_t({fTau})$, $s=$NAO$+$')
    labels_list.append('80$^{th}$ Percentile NAO Index')

# Adjust layout and add legend below the plot
fig.legend(handles=handles_list, labels=labels_list, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.1))
plt.tight_layout()
plt.show()

########### FIGURE 7

import os
import pandas as pd
import matplotlib.pyplot as plt

# Directory where the CSV files are located
directory = "/Users/admin/Documents/PhD/persistence/data_persistence/"

# Initialize lists to store data
data = []
excluded_years = []

# Loop through all the files in the directory
for filename in sorted(os.listdir(directory)):
    # Only process files that start with 'results_' and end with '_0.95_1950.csv'
    if filename.startswith("results_") and filename.endswith("_0.95_1950.csv"):
        # Extract the excluded year from the filename
        excluded_year = filename.split('_')[1]
        excluded_years.append(excluded_year)

        # Read the CSV file
        filepath = os.path.join(directory, filename)
        df = pd.read_csv(filepath)

        # Extract the 'mean_diff_pers_winter' column
        if 'mean_diff_pers_winter' in df.columns:
            mean_diff_values = df['mean_diff_pers_winter'].values

            # Remove the maximum value from the column
            mean_diff_values = mean_diff_values[mean_diff_values != mean_diff_values.max()]

            # Append the remaining values to the data list
            data.append(mean_diff_values)

# Create the boxplot
plt.figure(figsize=(12, 6), dpi=200)
plt.boxplot(data, patch_artist=True)

# Add labels and title
plt.xticks(ticks=range(1, len(excluded_years) + 1), labels=excluded_years, rotation=90)
plt.xlabel("Excluded Year")
plt.ylabel("$\\bar{\\Delta}_{\\phi}(\\tau)$")
#plt.title("Boxplots of $\bar{\Delta}_{\phi}(\tau)$ for 'mean_diff_pers_winter' Across Excluded Years")

# Show the plot
plt.tight_layout()
plt.show()




######## FIGURE 8 KDES APPENDIX ##########


# `magic numbers': change these to make the different plots
quant = .95
temp = False
fix_old = False

test = QAR_temperature(sCity='DE BILT', fTau=.95, use_statsmodels=True, include_nao=True, split_nao=True, iLeafs=2)
test.prepare_data()

# Filter data for the months December, January, and February
old_winter = test.old.loc[test.old.index.month.isin([12, 1, 2])]
new_winter = test.new.loc[test.new.index.month.isin([12, 1, 2])]


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



# Extract temperatures after filtering
old_data_q5 = old_filtered['Temp']
new_data_q5 = new_filtered['Temp']

# Calculate the interquantile ranges for specified quantiles
quantiles = [.05, .1, .25, .75, .9, .95]

def calculate_interquantile_ranges(data, quantiles):
    return {q: (np.quantile(data, q), np.quantile(data, 1 - q)) for q in quantiles}

old_interquantile_ranges = calculate_interquantile_ranges(old_data_q5, quantiles)
new_interquantile_ranges = calculate_interquantile_ranges(new_data_q5, quantiles)

# Plotting the KDE plots with interquantile range annotations
plt.figure(figsize=(16, 8), dpi=100)  # Larger figure size

# KDE plots
sns.kdeplot(old_data_q5, color='orange', label='Old Data')
sns.kdeplot(new_data_q5, color='red', label='New Data')

# Adding annotations for interquantile ranges
annotation_y = 0.7  # Starting y-position for annotations

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
plt.tight_layout()
plt.legend(loc='upper right')  # Adjust legend position
plt.show()




# Filter for winter months (December, January, February)
old = test.old.loc[test.old.index.month.isin([12, 1, 2])]
new = test.new.loc[test.new.index.month.isin([12, 1, 2])]

# Calculate the 2.5th and 7.5th quantiles for the lagged old temperature data
shifted_temp_old = old.nao_index_cdas.shift(1)
q_low = shifted_temp_old.quantile(0.025)
q_high = shifted_temp_old.quantile(0.075)

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
plt.xlabel('Lagged Temperature')
plt.ylabel('Temperature')
plt.title('Scatter Plot of Temperatures vs Lagged Temperatures (DJF)')
plt.legend()

plt.show()


############# FIGURE 9 coef plots precip de bilt

test = QAR_precipitation(sCity='DE BILT', fTau=.95, use_statsmodels=True, include_nao=True)
test.prepare_data()

# Number of lags and months
n_lags = 1
months = [12, 1, 2]
quantiles = [0, 0.2, .4, .6, .8, 1]

predict0, predict1 = [1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 1]
mm_threshold = 5
outcome_threshold = .3


# Generate example binary time series data for test.old
y_prec_old = (test.old.Temp >= mm_threshold) * 1
data_old = pd.DataFrame(y_prec_old, columns=['Temp'])
data_old['nao_index_cdas'] = test.old.nao_index_cdas
y_prec_old_season = y_prec_old.loc[y_prec_old.index.month.isin(months)]

# Generate example binary time series data for test.new
y_prec_new = (test.new.Temp >= mm_threshold) * 1
data_new = pd.DataFrame(y_prec_new, columns=['Temp'])
data_new['nao_index_cdas'] = test.new.nao_index_cdas
y_prec_new_season = y_prec_new.loc[y_prec_new.index.month.isin(months)]


# Assign season to each row
data_old['season'] = data_old.index.month.map(get_season).values
data_new['season'] = data_new.index.month.map(get_season).values

data_winter_new = data_new.loc[data_new.index.month.isin([12,1,2])]
data_winter_old = data_old.loc[data_old.index.month.isin([12,1,2])]

p_rain_cond_nao_new = data_winter_new.loc[data_winter_new.nao_index_cdas.shift(1) > np.quantile(data_winter_new.nao_index_cdas, .8)].mean().Temp
p_rain_cond_nao_old = data_winter_old.loc[data_winter_old.nao_index_cdas.shift(1) > np.quantile(data_winter_old.nao_index_cdas, .8)].mean().Temp




# Define quantiles for quintiles
quantiles = np.linspace(0., 1, 6)  # Define the quintile cutoffs (0.2, 0.4, 0.6, 0.8)

# Initialize lists to hold results
p_rain_new = []
p_rain_old = []
ci_low_new = []
ci_high_new = []
ci_low_old = []
ci_high_old = []

# Confidence level (95%)
confidence = 0.95
z_score = stats.norm.ppf((1 + confidence) / 2)  # Z-score for 95% confidence

# Calculate mean temperatures and confidence intervals for each quintile for both datasets
for i in range(len(quantiles) - 1):
    lower_quantile = quantiles[i]
    upper_quantile = quantiles[i + 1]

    # New dataset
    mask_new = (data_winter_new.nao_index_cdas.shift(1) < np.quantile(data_winter_new.nao_index_cdas, upper_quantile)) & \
               (data_winter_new.nao_index_cdas.shift(1) > np.quantile(data_winter_new.nao_index_cdas, lower_quantile))
    subset_new = data_winter_new.loc[mask_new]
    mean_temp_new = subset_new.mean().Temp
    std_temp_new = subset_new.Temp.std()
    n_new = subset_new.shape[0]
    se_new = std_temp_new / np.sqrt(n_new)
    ci_low_new.append(mean_temp_new - z_score * se_new)
    ci_high_new.append(mean_temp_new + z_score * se_new)
    p_rain_new.append(mean_temp_new)
    
    # Old dataset
    mask_old = (data_winter_old.nao_index_cdas.shift(1) < np.quantile(data_winter_old.nao_index_cdas, upper_quantile)) & \
               (data_winter_old.nao_index_cdas.shift(1) > np.quantile(data_winter_old.nao_index_cdas, lower_quantile))
    subset_old = data_winter_old.loc[mask_old]
    mean_temp_old = subset_old.mean().Temp
    std_temp_old = subset_old.Temp.std()
    n_old = subset_old.shape[0]
    se_old = std_temp_old / np.sqrt(n_old)
    ci_low_old.append(mean_temp_old - z_score * se_old)
    ci_high_old.append(mean_temp_old + z_score * se_old)
    p_rain_old.append(mean_temp_old)

# Create a plot
plt.figure(figsize=(10, 5), dpi=100)

# Define positions for the bars
x = np.arange(len(p_rain_new))  # The x locations for the groups
width = 0.2  # The width of the bars


# Plot old data
plt.errorbar(x - width/2, p_rain_old, yerr=[np.array(p_rain_old) - np.array(ci_low_old), np.array(ci_high_old) - np.array(p_rain_old)], fmt='o', color='orange', label='old')

# Plot new data
plt.errorbar(x + width/2, p_rain_new, yerr=[np.array(p_rain_new) - np.array(ci_low_new), np.array(ci_high_new) - np.array(p_rain_new)], fmt='o', color='red', label='new')

# Labeling
plt.xticks(x, [f'Q{i+1}' for i in range(len(p_rain_new))])
plt.xlabel('NAO Index Quintiles')
plt.ylabel('Probability of rain')
#plt.title('Probability of rain for different Quintiles of NAO Index')
plt.title('(b)')
plt.legend()
plt.grid(True)

# Show plot
plt.show()


# Fit models for both datasets
seasonal_results_old, X_old, y_old = fit_ar_logistic_regression(data_old, n_lags, months, quantiles)
seasonal_results_new, X_new, y_new = fit_ar_logistic_regression(data_new, n_lags, months, quantiles)
sorted_data = np.sort(data_new.nao_index_cdas)
quantiles_fix = np.searchsorted(sorted_data, np.quantile(data_old.nao_index_cdas, [0, .2, .4, .6, .8, 1])) / len(sorted_data)
quantiles_fix[0], quantiles_fix[-1] = 0, 1
seasonal_results_fix, X_fix, y_fix = fit_ar_logistic_regression(data_new, n_lags, months, quantiles_fix)


df_new_ = pd.DataFrame(X_new).drop_duplicates().reset_index(drop=True)
df_new_['first_one_index'] = df_new_.apply(first_one_index, axis=1)
df_sorted_new = df_new_.sort_values(by=[1, 'first_one_index'] + list(df_new_.columns), ascending=[True, True] + [True]*len(df_new_.columns)).drop('first_one_index', axis=1).reset_index(drop=True)


results_old_df = extract_results(seasonal_results_old, 'test.old')
results_new_df = extract_results(seasonal_results_new, 'test.new')

# Combine results into a single DataFrame for comparison
comparison_df = pd.concat([results_old_df, results_new_df])

# Display the comparison DataFrame
#print(comparison_df)

plot_coefficients(comparison_df)



######## FIGURE 10 rolling window precipitation

# Define constants
months = [12, 1, 2]  # Winter months (Dec, Jan, Feb)
mm_threshold = 5  # Temperature threshold for binary rain data
quantile = 0.8  # Upper quintile of NAO index
window_size = 30  # 30-year window
confidence = 0.95  # 95% confidence interval
z_score = stats.norm.ppf((1 + confidence) / 2)  # Z-score for 95% confidence

# Initialize a list to store results for each 30-year window
prob_rain_upper_nao_30_year = []
upper_quintile_values = []  # Store upper quintile of NAO index for each window

# Define the range of years for the 30-year overlapping windows (e.g., 1950–1980, 1951–1981, ...)
start_years = range(1950, 1991)  # 1994 is the last start year that gives 1994-2023

# Perform rolling window analysis for 30-year overlappin∂g windows
for start_year in start_years:
    end_year = start_year + window_size

    # Dynamically adjust the oldstart and oldend for each 30-year window
    oldstart_str = f'{start_year}-'
    oldend_str = f'{end_year}-'

    # Initialize the QAR_climate object for the current 30-year window
    test = QAR_precipitation(sCity='DE BILT', fTau=.95, use_statsmodels=True, include_nao=True, oldstart=oldstart_str, oldend=oldend_str)
    test.prepare_data()

    # Generate binary time series for the old dataset
    y_prec_old = (test.old.Temp >= mm_threshold) * 1
    data_old = pd.DataFrame(y_prec_old, columns=['Temp'])
    data_old['nao_index_cdas'] = test.old.nao_index_cdas

    # Filter for the winter months
    data_winter_old = data_old.loc[data_old.index.month.isin(months)]

    # Calculate the upper quintile of the NAO index in the current 30-year window
    upper_quintile_nao = np.quantile(data_winter_old['nao_index_cdas'], quantile)
    upper_quintile_values.append(upper_quintile_nao)

    # Filter the data for the upper quintile of the NAO index
    subset = data_winter_old[data_winter_old['nao_index_cdas'].shift(1) >= upper_quintile_nao]

    # Calculate the probability of rain in the upper NAO quintile (mean of binary rain data)
    prob_rain = subset['Temp'].mean()

    # Calculate the standard error and confidence interval
    n = len(subset['Temp'])  # Number of observations
    std_temp = subset['Temp'].std()  # Standard deviation
    se_temp = std_temp / np.sqrt(n)  # Standard error of the mean
    ci_low = prob_rain - z_score * se_temp  # Lower bound of the confidence interval
    ci_high = prob_rain + z_score * se_temp  # Upper bound of the confidence interval

    # Store the results
    prob_rain_upper_nao_30_year.append({
        'start_year': start_year,
        'end_year': end_year - 1,  # Inclusive of the last year in the window
        'prob_rain': prob_rain,
        'ci_low': ci_low,
        'ci_high': ci_high
    })

# Convert the results into a DataFrame
prob_rain_df = pd.DataFrame(prob_rain_upper_nao_30_year)

# Plot the probability of rain in the upper NAO quintile over time (30-year windows) with confidence intervals
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot the probability of rain
line1, = ax1.plot(prob_rain_df['end_year'], prob_rain_df['prob_rain'], marker='o', linestyle='-', color='blue', label='Probability of rain (upper NAO quintile)')
ax1.fill_between(prob_rain_df['end_year'], prob_rain_df['ci_low'], prob_rain_df['ci_high'], color='blue', alpha=0.3, label='95% Confidence Interval')
ax1.set_xlabel('End Year of 30-Year Window')
ax1.set_ylabel('Probability of Rain', color='black', rotation=90)
ax1.grid(True)
ax1.set_title(f'Probability of Rain in Upper NAO Quintile in De Bilt in Winter (30-Year Windows)')

# Set the left axis (probability of rain) tick labels to blue, but keep the axis label black
ax1.tick_params(axis='y', colors='blue')

# Create a second y-axis for the upper quintile of the NAO index
ax2 = ax1.twinx()
line2, = ax2.plot(prob_rain_df['end_year'], upper_quintile_values, marker='s', linestyle='--', color='red', label='80$^{th}$ Percentile of the NAO Index')

# Set the right axis (NAO index) label to black and tick labels to red
ax2.set_ylabel('NAO Index Value', color='black', rotation=90)
ax2.tick_params(axis='y', colors='red')

# Combine both axes' legends into one
lines = [line1, line2]
labels = [line1.get_label(), line2.get_label()]
ax1.legend(lines, labels, loc='upper left')

# Show plot
plt.show()


#######FIGURE 11

# Path where the new precipitation CSV files are stored
folder_path = '/Users/admin/Documents/PhD/persistence/data_persistence/'  # Replace with your actual folder path

# List all the files that match the "results_precipitation_DJF_YYYY.csv" pattern
files = sorted([file for file in os.listdir(folder_path) if file.startswith('results_precipitation_DJF_') and file.endswith('.csv')])

# Initialize a dictionary to hold data for each year
mean_diff_winter_data = {}

# Loop through the files to extract the relevant data
for file in files:
    # Extract the year from the filename
    year = int(file.split('_')[3].split('.')[0])

    # Process files only in the range of 1990 to 2019
    if 1990 <= year <= 2019:
        # Read the CSV file
        df = pd.read_csv(os.path.join(folder_path, file))

        # Extract the data for the 'mean_diff_winter' variable if it exists
        if 'mean_diff_winter' in df.columns:
            mean_diff_winter_data[year] = df['mean_diff_winter'].dropna()

# Create a boxplot for 'mean_diff_winter' over the years
fig, ax = plt.subplots(figsize=(14, 7))

# Convert the dictionary to a list for plotting
years = sorted(mean_diff_winter_data.keys())
mean_diff_winter = [mean_diff_winter_data[year] for year in years]

# Boxplot for 'mean_diff_winter'
ax.boxplot(mean_diff_winter, labels=years, patch_artist=True)
ax.set_title('Precipitation - mean_diff_winter')
ax.set_xlabel('Year')
ax.set_ylabel('Value')

plt.tight_layout()
plt.show()


######## FIGURE 12

# Initialize a dictionary to hold data for each year
mean_diff_winter_data = {}

# Loop through the files to extract the relevant data
for file in files:
    # Extract the year from the filename
    year = int(file.split('_')[3].split('.')[0])

    # Process files only in the range of 1990 to 2019
    if 1990 <= year <= 2019:
        # Read the CSV file
        df = pd.read_csv(os.path.join(folder_path, file))

        # Extract the data for the 'mean_diff_winter' variable if it exists
        if 'mean_diff_winter' in df.columns:
            mean_diff_winter_data[year] = df['mean_diff_winter_unc'].dropna()

# Create a boxplot for 'mean_diff_winter' over the years
fig, ax = plt.subplots(figsize=(14, 7))

# Convert the dictionary to a list for plotting
years = sorted(mean_diff_winter_data.keys())
mean_diff_winter = [mean_diff_winter_data[year] for year in years]

# Boxplot for 'mean_diff_winter'
ax.boxplot(mean_diff_winter, labels=years, patch_artist=True)
ax.set_title('Precipitation - mean_diff_winter')
ax.set_xlabel('Year')
ax.set_ylabel('Value')

plt.tight_layout()
plt.show()



