import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from QAR_persistence_precip import QAR_precipitation
from scipy import stats

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
def create_lagged_features(data, n_lags, quantiles):
    lagged_data = data.copy()
    
    # Create lagged features for 'Temp'
    for lag in range(1, n_lags + 1):
        lagged_data[f'lag_{lag}'] = lagged_data['Temp'].shift(lag)
    
    # Create categorical indicators for 'nao_index_cdas'
    lagged_data['nao_index_cdas'] = lagged_data['nao_index_cdas'].shift(1)  # Shift nao_index_cdas
    lagged_data['nao_index_cdas_cat'] = pd.qcut(lagged_data['nao_index_cdas'], quantiles, labels=False)
    
    # Convert categorical indicator into dummy variables
    nao_index_dummies = pd.get_dummies(lagged_data['nao_index_cdas_cat'], prefix='nao_index_cat', drop_first=False)
    lagged_data = pd.concat([lagged_data, nao_index_dummies], axis=1)
    # Drop original categorical indicator column and rows with NaN values due to lagging and shifting

    lagged_data = lagged_data.drop(['nao_index_cdas', 'nao_index_cat_0.0'], axis=1).dropna()
    return lagged_data

# Function to fit AR logistic regression for all winter months
def fit_ar_logistic_regression(data, n_lags, months, quantiles):
    # Filter data for the specified months (e.g., winter)
    winter_data = data[data.index.month.isin(months)].copy()

    # Create lagged features
    seasonal_data = create_lagged_features(winter_data, n_lags, quantiles)

    # Prepare predictors (X) and response variable (y)
    lag_columns = [f'lag_{lag}' for lag in range(1, n_lags + 1)]
    nao_columns = list(seasonal_data.filter(like='nao_index_cat').columns)
    X = seasonal_data[lag_columns + nao_columns].values
    y = seasonal_data['Temp'].values

    # Add a constant term for the intercept
    X = sm.add_constant(X)
    
    # Fit the autoregressive logistic regression model
    model = sm.Logit(y, X)
    result = model.fit(disp=0)

    return result, X, y

# Prepare results for comparison
def extract_results(result, dataset_label):
    rows = []
    for idx, param in enumerate(result.params):
        param_name = '$\\beta_0$' if idx == 0 else f'$\\beta_{idx}$' if idx <= n_lags else f'$\gamma_{idx + 1 - n_lags}$'
        rows.append({
            'Dataset': dataset_label,
            'Parameter': param_name,
            'Coefficient': param,
            'Conf_Lower': result.conf_int(alpha=0.05)[idx][0],  # 90% confidence lower bound
            'Conf_Upper': result.conf_int(alpha=0.05)[idx][1]   # 90% confidence upper bound
        })
    return pd.DataFrame(rows)


# Plotting the coefficients and their confidence bounds comparing old (in orange) vs new (in red)
def plot_coefficients(comparison_df):
    parameters = comparison_df['Parameter'].unique()
    datasets = comparison_df['Dataset'].unique()
    
    fig, ax = plt.subplots(figsize=(10, 5), dpi=100)
    
    # Jitter offset for old and new datasets
    jitter = {'test.old': -0.1, 'test.new': 0.1}
    
    for dataset in datasets:
        subset = comparison_df[comparison_df['Dataset'] == dataset]
        color = 'orange' if dataset == 'test.old' else 'red'
        
        for i, param in enumerate(parameters):
            param_data = subset[subset['Parameter'] == param]
            if not param_data.empty:
                x_pos = i + jitter[dataset]
                ax.errorbar(x_pos, param_data['Coefficient'].values[0], 
                            yerr=[[param_data['Coefficient'].values[0] - param_data['Conf_Lower'].values[0]], 
                                  [param_data['Conf_Upper'].values[0] - param_data['Coefficient'].values[0]]],
                            fmt='o', label=str(dataset)[-3:] if i == 0 else "", color=color)
    
    #plt.xticks(x, ['intercept', '$y_{t-1}$'] + [f'\gamma_{i}' for i in np.arange(2,6)])

    ax.set_xlabel('Parameter')
    ax.set_ylabel('Value')
    ax.set_title('(a)')
    #ax.set_title('Coefficients and 95% Confidence Bounds: Old vs New for ' + test.sCity)
    ax.legend()
    plt.grid()
    plt.xticks(ticks=range(len(parameters)), labels=parameters)
    plt.show()
    

def compare_rows(matrix, row):
    # Convert the row to a numpy array if it's not already
    row = np.array(row)

    # Check which rows in the matrix match the given row
    matches = np.all(matrix == row, axis=1)

    # Find the indices of the matching rows
    matching_indices = np.where(matches)[0]

    return matching_indices


# Helper function to find the first index of 1 in the row starting from the third column
def first_one_index(row):
    return next((i for i, val in enumerate(row[2:], start=2) if val == 1), len(row))




def random_choice_with_probabilities(row):
    return np.random.choice(len(row), p=row/row.sum())

def simulate_prob_rain(model_new, model_old, model_fix=None, iS=2000, iT=21, stay_in_upper_quintile=False, fix_old_quantiles=False):
   df_old_quintiles = pd.qcut(test.old.nao_index_cdas.loc[test.old.index.month.isin([12,1,2])], q=5, labels=False)
   transition_matrix_old = pd.crosstab(df_old_quintiles[:-1].values, df_old_quintiles[1:].values, rownames=['Current Quintile'], colnames=['Next Quintile'], normalize='index')
   
   df_new_quintiles = pd.qcut(test.new.nao_index_cdas.loc[test.new.index.month.isin([12,1,2])], q=5, labels=False)
   transition_matrix_new = pd.crosstab(df_new_quintiles[:-1].values, df_new_quintiles[1:].values, rownames=['Current Quintile'], colnames=['Next Quintile'], normalize='index')
   
   cut = np.quantile(test.old.nao_index_cdas.loc[test.old.index.month.isin([12,1,2])],[0,.2,.4,.6,.8,1])
   cut[0], cut[-1] = -3, 3
   df_fix_quintiles = pd.cut(test.new.nao_index_cdas.loc[test.new.index.month.isin([12,1,2])], cut, labels=False, precision=1)
   transition_matrix_fix = pd.crosstab(df_fix_quintiles[:-1].values, df_fix_quintiles[1:].values, rownames=['Current Quintile'], colnames=['Next Quintile'], normalize='index')
   
   
   mQuintiles_new, mQuintiles_old, mQuintiles_fix, mPrecip_old, mPrecip_new, mPrecip_fix = np.zeros((iS, iT)), np.zeros((iS, iT)), np.zeros((iS, iT)), np.zeros((iS, iT)), np.zeros((iS, iT)), np.zeros((iS, iT))
   mQuintiles_new[:,0], mQuintiles_old[:,0], mQuintiles_fix[:,0] = 4, 4, 4
   mPrecip_new[:,0], mPrecip_old[:,0], mPrecip_fix[:,0] = 1, 1, 1#np.random.binomial(1, 0.65, iS), np.random.binomial(1, 0.55, iS)
   for i in np.arange(1, iT):
       array_new = transition_matrix_new.loc[(mQuintiles_new[:,i-1]).astype(int)].values
       mQuintiles_new[:, i] = 4 if stay_in_upper_quintile == True else np.apply_along_axis(random_choice_with_probabilities, axis=1, arr=array_new)
       array_old = transition_matrix_old.loc[(mQuintiles_old[:,i-1]).astype(int)].values
       mQuintiles_old[:, i] =  4 if stay_in_upper_quintile == True else np.apply_along_axis(random_choice_with_probabilities, axis=1, arr=array_old)   
       
       
       result_rows_new, result_rows_old = np.zeros((array_new.shape[0], array_new.shape[1])), np.zeros((array_new.shape[0], array_new.shape[1]))
       result_rows_new[np.arange(array_new.shape[0]), mQuintiles_new[:, i-1].astype(int)] = 1
       result_rows_new = result_rows_new[:, 1:]
       mPrecip_new[:, i] = np.random.binomial(1, model_new.predict(np.column_stack((np.repeat(1,array_new.shape[0]), mPrecip_new[:, i-1], result_rows_new))))
       
       result_rows_old[np.arange(array_old.shape[0]), mQuintiles_old[:, i-1].astype(int)] = 1
       result_rows_old = result_rows_old[:, 1:]
       mPrecip_old[:, i] = np.random.binomial(1, model_old.predict(np.column_stack((np.repeat(1,array_old.shape[0]), mPrecip_old[:, i-1], result_rows_old))))
       
       if fix_old_quantiles:
           array_fix = transition_matrix_fix.loc[(mQuintiles_new[:,i-1]).astype(int)].values
           mQuintiles_fix[:, i] = 4 if stay_in_upper_quintile == True else np.apply_along_axis(random_choice_with_probabilities, axis=1, arr=array_new)

           result_rows_fix = np.zeros((array_new.shape[0], array_new.shape[1]))
           result_rows_fix[np.arange(array_fix.shape[0]), mQuintiles_fix[:, i-1].astype(int)] = 1
           result_rows_fix = result_rows_fix[:, 1:]
           mPrecip_fix[:, i] = np.random.binomial(1, model_fix.predict(np.column_stack((np.repeat(1,array_fix.shape[0]), mPrecip_fix[:, i-1], result_rows_fix))))
    
   if fix_old_quantiles:
       return mPrecip_old[:, 1:], mPrecip_new[:, 1:], mPrecip_fix[:, 1:]
   else: 
       return mPrecip_old[:, 1:], mPrecip_new[:, 1:], None
   
    

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



# Fit models for both datasets
seasonal_results_old, X_old, y_old = fit_ar_logistic_regression(data_old, n_lags, months, quantiles)
seasonal_results_new, X_new, y_new = fit_ar_logistic_regression(data_new, n_lags, months, quantiles)
sorted_data = np.sort(data_new.nao_index_cdas)
quantiles_fix = np.searchsorted(sorted_data, np.quantile(data_old.nao_index_cdas, [0, .2, .4, .6, .8, 1])) / len(sorted_data)
quantiles_fix[0], quantiles_fix[-1] = 0, 1
seasonal_results_fix, X_fix, y_fix = fit_ar_logistic_regression(data_new, n_lags, months, quantiles_fix)


##### SIMULATE DATA ######
mPrecip_old, mPrecip_new, mPrecip_fix = simulate_prob_rain(seasonal_results_new, seasonal_results_old, model_fix=seasonal_results_fix, iS=20000, iT=31, stay_in_upper_quintile=False, fix_old_quantiles=True)


# Example arrays for mPrecip_new and mPrecip_old where 1 represents rain
# Step 1: Count the number of rainy days (1s) for each 20-day period in both datasets
rain_counts_new = np.sum(mPrecip_new == 1, axis=1)
rain_counts_old = np.sum(mPrecip_old == 1, axis=1)
rain_counts_fix = np.sum(mPrecip_fix == 1, axis=1)

# Step 2: Calculate the relative frequencies (probabilities) for each number of rainy days
total_days = 30
rain_days = np.arange(0, total_days + 1)  # Possible number of rainy days from 0 to 20

# Calculate the probabilities for new and old data
prob_rain_days_new = np.array([np.mean(rain_counts_new == x) for x in rain_days])
prob_rain_days_old = np.array([np.mean(rain_counts_old == x) for x in rain_days])
prob_rain_days_fix = np.array([np.mean(rain_counts_fix == x) for x in rain_days])

# Step 3: Plot the probabilities with specific colors
plt.figure(figsize=(10, 6))

# Plot for mPrecip_new (red)
plt.plot(rain_days, prob_rain_days_new, label='New', marker='o', color='red')

# Plot for mPrecip_old (orange)
plt.plot(rain_days, prob_rain_days_old, label='Old', marker='o', color='orange')

# Plot for mPrecip_old (orange)
plt.plot(rain_days, prob_rain_days_fix, label='New with old transition matrix', marker='o', color='green')


# Step 4: Labeling the plot
plt.xlabel('Number of Rainy Days in 30-Day Period')
plt.ylabel('Probability')

# Ensure x-axis has integer ticks
plt.xticks(rain_days)

# Add grid and legend
plt.grid(True)
plt.legend()

# Show plot
plt.show()

# Example arrays for mPrecip_new and mPrecip_old where 1 represents rain
# Step 1: Count the number of rainy days (1s) for each 20-day period in both datasets
rain_counts_new = np.sum(mPrecip_new == 1, axis=1)
rain_counts_old = np.sum(mPrecip_old == 1, axis=1)
rain_counts_fix = np.sum(mPrecip_fix == 1, axis=1)

# Step 2: Calculate the relative frequencies (probabilities) for each number of rainy days
total_days = 30
rain_days = np.arange(0, total_days + 1)  # Possible number of rainy days from 0 to 20

# Calculate the probabilities for new and old data
prob_rain_days_new = np.array([np.mean(rain_counts_new == x) for x in rain_days])
prob_rain_days_old = np.array([np.mean(rain_counts_old == x) for x in rain_days])
prob_rain_days_fix = np.array([np.mean(rain_counts_fix == x) for x in rain_days])

# Step 3: Calculate the cumulative probabilities (CDF)
cdf_rain_days_new = np.cumsum(prob_rain_days_new)
cdf_rain_days_old = np.cumsum(prob_rain_days_old)
cdf_rain_days_fix = np.cumsum(prob_rain_days_fix)

# Step 4: Plot the cumulative densities (CDF) for both datasets
plt.figure(figsize=(10, 6))

# CDF plot for mPrecip_new (red)
plt.plot(rain_days, cdf_rain_days_new, label='New', linestyle='-', color='red')

# CDF plot for mPrecip_old (orange)
plt.plot(rain_days, cdf_rain_days_old, label='Old', linestyle='-', color='orange')

# CDF plot for mPrecip_old (orange)
plt.plot(rain_days, cdf_rain_days_fix, label='New with old transition matrix', linestyle='-', color='green')


# Step 5: Labeling the plot
plt.xlabel('Number of Rainy Days in 30-Day Period')
plt.ylabel('Cumulative Probability')

# Ensure x-axis has integer ticks
plt.xticks(rain_days)

# Add grid and legend
plt.grid(True)
plt.legend()

# Show plot
plt.show()


# Function to calculate the first streak of rainy days starting from day 1
def first_streak(row):
    if row[0] == 1:  # Start counting only if day 1 is rainy
        streak_length = 1
        for day in row[1:]:
            if day == 1:
                streak_length += 1
            else:
                break  # End streak when encountering the first non-rainy day
        return streak_length
    return 0  # If day 1 is not rainy, return 0

# Step 1: Calculate the first streak lengths for both old and new datasets
first_streaks_new = [first_streak(row) for row in mPrecip_new]
first_streaks_old = [first_streak(row) for row in mPrecip_old]
first_streaks_fix = [first_streak(row) for row in mPrecip_fix]

# Step 2: Calculate the frequency of each streak length for both datasets
unique_streaks_new, counts_new = np.unique(first_streaks_new, return_counts=True)
unique_streaks_old, counts_old = np.unique(first_streaks_old, return_counts=True)
unique_streaks_fix, counts_fix = np.unique(first_streaks_fix, return_counts=True)

# Step 3: Calculate probabilities (relative frequencies) for both datasets
probabilities_new = counts_new / len(first_streaks_new)
probabilities_old = counts_old / len(first_streaks_old)
probabilities_fix = counts_fix / len(first_streaks_fix)

# Step 4: Plot the results for both datasets
plt.figure(figsize=(10, 6))

# Plot for new data (red)
plt.plot(unique_streaks_new, probabilities_new, marker='o', linestyle='-', color='red', label='New Data')

# Plot for old data (orange)
plt.plot(unique_streaks_old, probabilities_old, marker='o', linestyle='-', color='orange', label='Old Data')

# Plot for old data (orange)
plt.plot(unique_streaks_fix, probabilities_fix, marker='o', linestyle='-', color='green', label='fix Data')

# Labeling the plot
plt.xlabel('Length of First Streak of Consecutive Rainy Days')
plt.ylabel('Probability')
plt.title('Probability of Length of First Rainy Streak (Old vs. New)')

# Ensure integers on the x-axis
plt.xticks(np.arange(0, max(unique_streaks_new.max(), unique_streaks_old.max()) + 1, 1))

# Add grid and legend
plt.grid(True)
plt.legend()

# Show plot
plt.show()



# Function to calculate the first streak of rainy days starting from day 1
def first_streak(row):
    if row[0] == 1:  # Start counting only if day 1 is rainy
        streak_length = 1
        for day in row[1:]:
            if day == 1:
                streak_length += 1
            else:
                break  # End streak when encountering the first non-rainy day
        return streak_length
    return 0  # If day 1 is not rainy, return 0

# Step 1: Calculate the first streak lengths for both old and new datasets
first_streaks_new = [first_streak(row) for row in mPrecip_new]
first_streaks_old = [first_streak(row) for row in mPrecip_old]
first_streaks_fix = [first_streak(row) for row in mPrecip_fix]

# Step 2: Calculate the frequency of each streak length for both datasets
unique_streaks_new, counts_new = np.unique(first_streaks_new, return_counts=True)
unique_streaks_old, counts_old = np.unique(first_streaks_old, return_counts=True)
unique_streaks_fix, counts_fix = np.unique(first_streaks_fix, return_counts=True)

# Step 3: Calculate cumulative probabilities (probability of more than x days)
cumulative_probabilities_new = np.cumsum(counts_new[::-1])[::-1] / len(first_streaks_new)
cumulative_probabilities_old = np.cumsum(counts_old[::-1])[::-1] / len(first_streaks_old)
cumulative_probabilities_fix = np.cumsum(counts_fix[::-1])[::-1] / len(first_streaks_fix)

# Step 4: Plot the cumulative probability of having more than x rainy days
plt.figure(figsize=(10, 6))

# Plot for new data (red)
plt.plot(unique_streaks_new, cumulative_probabilities_new, marker='o', linestyle='-', color='red', label='New Data')

# Plot for old data (orange)
plt.plot(unique_streaks_old, cumulative_probabilities_old, marker='o', linestyle='-', color='orange', label='Old Data')

# Plot for old data (orange)
plt.plot(unique_streaks_fix, cumulative_probabilities_fix, marker='o', linestyle='-', color='green', label='fix Data')


# Labeling the plot
plt.xlabel('Length of First Streak of Consecutive Rainy Days')
plt.ylabel('Cumulative Probability of More Than x Days')
plt.title('Cumulative Probability of More Than x Rainy Days (Old vs. New)')

# Ensure integers on the x-axis
plt.xticks(np.arange(0, max(unique_streaks_new.max(), unique_streaks_old.max()) + 1, 1))

# Add grid and legend
plt.grid(True)
plt.legend()

# Show plot
plt.show()

# Step 1: Count the number of rainy days (1s) for each 20-day period in both datasets
rain_counts_new = np.sum(mPrecip_new == 1, axis=1)
rain_counts_old = np.sum(mPrecip_old == 1, axis=1)
rain_counts_fix = np.sum(mPrecip_fix == 1, axis=1)

# Step 2: Calculate the relative frequencies (probabilities) for each number of rainy days
total_days = 20
rain_days = np.arange(0, total_days + 1)  # Possible number of rainy days from 0 to 20

# Calculate the probabilities for new and old data
prob_rain_days_new = np.array([np.mean(rain_counts_new == x) for x in rain_days])
prob_rain_days_old = np.array([np.mean(rain_counts_old == x) for x in rain_days])
prob_rain_days_fix = np.array([np.mean(rain_counts_fix == x) for x in rain_days])

# Step 3: Plot the probabilities with specific colors
plt.figure(figsize=(10, 6))

# Plot for mPrecip_new (red)
plt.plot(rain_days, prob_rain_days_new, label='New Data (p=New)', marker='o', color='red')

# Plot for mPrecip_old (orange)
plt.plot(rain_days, prob_rain_days_old, label='Old Data (p=Old)', marker='o', color='orange')

# Plot for mPrecip_old (orange)
plt.plot(rain_days, prob_rain_days_fix, label='fix Data (p=fix)', marker='o', color='green')

# Step 4: Labeling the plot
plt.xlabel('Number of Rainy Days in 20-Day Period')
plt.ylabel('Probability')
plt.title('Probability of Having x Rainy Days in 20-Day Period')

# Ensure x-axis has integer ticks
plt.xticks(rain_days)

# Add grid and legend
plt.grid(True)
plt.legend()

# Show plot
plt.show()

# Step 1: Count the number of rainy days (1s) for each 20-day period in both datasets
rain_counts_new = np.sum(mPrecip_new == 1, axis=1)
rain_counts_old = np.sum(mPrecip_old == 1, axis=1)

# Step 2: Calculate the relative frequencies (probabilities) for each number of rainy days
total_days = 20
rain_days = np.arange(0, total_days + 1)  # Possible number of rainy days from 0 to 20

# Calculate the probabilities for new and old data
prob_rain_days_new = np.array([np.mean(rain_counts_new == x) for x in rain_days])
prob_rain_days_old = np.array([np.mean(rain_counts_old == x) for x in rain_days])

# Step 3: Calculate the cumulative probabilities (CDF)
cdf_rain_days_new = np.cumsum(prob_rain_days_new)
cdf_rain_days_old = np.cumsum(prob_rain_days_old)

# Step 4: Plot the cumulative densities (CDF) for both datasets
plt.figure(figsize=(10, 6))

# CDF plot for mPrecip_new (red)
plt.plot(rain_days, cdf_rain_days_new, label='New Data (CDF)', linestyle='-', color='red')

# CDF plot for mPrecip_old (orange)
plt.plot(rain_days, cdf_rain_days_old, label='Old Data (CDF)', linestyle='-', color='orange')

# Step 5: Labeling the plot
plt.xlabel('Number of Rainy Days in 20-Day Period')
plt.ylabel('Cumulative Probability')
plt.title('Cumulative Density of Rainy Days in 20-Day Period')

# Ensure x-axis has integer ticks
plt.xticks(rain_days)

# Add grid and legend
plt.grid(True)
plt.legend()

# Show plot
plt.show()

rows_0_old, rows_1_new = compare_rows(X_new, predict0), compare_rows(X_new, predict1)
print(y_prec_new_season.iloc[rows_0_new + n_lags].mean(), y_prec_new_season.iloc[rows_1_new + n_lags].mean())
rows_0_old, rows_1_old = compare_rows(X_old, predict0), compare_rows(X_old, predict1)
print(y_prec_old_season.iloc[rows_0_old + n_lags].mean(), y_prec_old_season.iloc[rows_1_old + n_lags].mean())

y_pred_new = seasonal_results_new.predict(X_new)
y_pred_old = seasonal_results_new.predict(X_old)
#predict outcomes
predicted_choice_new = (y_pred_new > outcome_threshold).astype(int)
predicted_choice_old = (y_pred_old > outcome_threshold).astype(int)




# Parameters
p1 = 0.65  # Probability of success
p2 = 0.55
x_values_limited = np.arange(1, 14)  # Number of consecutive hits, limited to 13

# Calculate P(X > x) for both probabilities
prob_more_than_x_1_limited = (1 - p1) ** x_values_limited
prob_more_than_x_2_limited = (1 - p2) ** x_values_limited

# Plot
plt.figure(figsize=(10, 6))

plt.plot(x_values_limited, prob_more_than_x_1_limited, label=f'p = {p1}', marker='o')
plt.plot(x_values_limited, prob_more_than_x_2_limited, label=f'p = {p2}', marker='o')

plt.xlabel('Number of Consecutive Hits (x)')
plt.ylabel('P(X > x)')
plt.title('Probability of More Than x Consecutive Hits (up to x = 13)')

# Ensure x-axis has integer ticks and stops at 13
plt.xticks(x_values_limited)

plt.grid(True)
plt.legend()
plt.show()



# Parameters
total_days = 30  # Total number of days (trials)
x_days = np.arange(1, total_days + 1)  # Number of consecutive hits (successes)
p1 = 0.65  # Probability of success
p2 = 0.55

# Cumulative probabilities (CDF) for both p1 and p2
cumulative_prob_x_days_1 = stats.binom.cdf(x_days, total_days, p1)
cumulative_prob_x_days_2 = stats.binom.cdf(x_days, total_days, p2)

# Plotting cumulative density functions
plt.figure(figsize=(10,6), dpi=200)

plt.plot(x_days, cumulative_prob_x_days_1, label=f'Cumulative p = {p1}', marker='o', color='red')
plt.plot(x_days, cumulative_prob_x_days_2, label=f'Cumulative p = {p2}', marker='o', color='orange')

plt.xlabel('Rainy days')
plt.ylabel('Cumulative Probability')

# Ensure x-axis has integer ticks
plt.xticks(x_days)

plt.grid(True)
plt.legend()
plt.show()


# Parameters
total_days = 30  # Total number of days (trials)
x_days = np.arange(1, total_days + 1)  # Number of consecutive hits (successes)
p1 = 0.65  # Probability of success
p2 = 0.55

# Binomial probabilities of exactly x hits in 20 trials
prob_x_days_1 = stats.binom.pmf(x_days, total_days, p1)
prob_x_days_2 = stats.binom.pmf(x_days, total_days, p2)

# Plotting with specific colors for each probability
plt.figure(figsize=(10,6), dpi=200)

plt.plot(x_days, prob_x_days_1, label=f'p = {p1}', marker='o', color='red')
plt.plot(x_days, prob_x_days_2, label=f'p = {p2}', marker='o', color='orange')

plt.xlabel('Rainy days')
plt.ylabel('Density')

# Ensure x-axis has integer ticks
plt.xticks(x_days)

plt.grid(True)
plt.legend()
plt.show()


