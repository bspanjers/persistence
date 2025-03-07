import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
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

# Create lag features for each season, including shifting nao_index_cdas
def create_lagged_features2(data, n_lags):
    lagged_data = data.copy()
    for lag in range(1, n_lags + 1):
        lagged_data[f'lag_{lag}'] = lagged_data['Temp'].shift(lag)
    lagged_data['nao_index_cdas'] = lagged_data['nao_index_cdas'].shift(1)  # Shift nao_index_cdas
    return lagged_data.dropna()

# Create lag features for each season, including shifting nao_index_cdas and creating dummies
def create_lagged_features(data, n_lags):
    lagged_data = data.copy()
    
    # Create lagged features for 'Temp'
    for lag in range(1, n_lags + 1):
        lagged_data[f'lag_{lag}'] = lagged_data['Temp'].shift(lag)
    
    # Create categorical indicators for 'nao_index_cdas'
    lagged_data['nao_index_cdas'] = lagged_data['nao_index_cdas'].shift(1)  # Shift nao_index_cdas
    quantiles = [0, 0.4, 0.8, 1.0]  # Define quantiles for categorical conversion
    lagged_data['nao_index_cdas_cat'] = pd.qcut(lagged_data['nao_index_cdas'], quantiles, labels=False)
    
    # Convert categorical indicator into dummy variables
    nao_index_dummies = pd.get_dummies(lagged_data['nao_index_cdas_cat'], prefix='nao_index_cat', drop_first=True)
    lagged_data = pd.concat([lagged_data, nao_index_dummies], axis=1)
    
    # Drop original categorical indicator column and rows with NaN values due to lagging and shifting
    lagged_data = lagged_data.drop(['nao_index_cdas', 'nao_index_cdas_cat'], axis=1).dropna()
    
    return lagged_data

# Function to fit AR logistic regression for a given datteaset
def fit_ar_logistic_regression(data, n_lags):
    # Initialize dictionary to store results
    seasonal_results = {}

    # Iterate over each winter month (December, January, February)
    winter_months = [12, 1, 2]
    for month in winter_months:
        seasonal_data = data[data.index.month == month].copy()
        seasonal_data = create_lagged_features(seasonal_data, n_lags)

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
        model = sm.Logit(y, X_scaled)
        result = model.fit(disp=0)

        # Store the result in the dictionary
        seasonal_results[f'DJF_{month}'] = result

    return seasonal_results


# Generate example binary time series data for test.old
y_prec_old = (test.old.Temp >= 1) * 1
data_old = pd.DataFrame(y_prec_old, columns=['Temp'])
data_old['nao_index_cdas'] = test.old.nao_index_cdas

# Generate example binary time series data for test.new
y_prec_new = (test.new.Temp >= 1) * 1
data_new = pd.DataFrame(y_prec_new, columns=['Temp'])
data_new['nao_index_cdas'] = test.new.nao_index_cdas

# Assign season to each row
data_old['season'] = data_old.index.month.map(get_season).values
data_new['season'] = data_new.index.month.map(get_season).values

# Number of lags
n_lags = 2

# Fit models for both datasets
seasonal_results_old = fit_ar_logistic_regression(data_old, n_lags)
seasonal_results_new = fit_ar_logistic_regression(data_new, n_lags)

# Prepare results for comparison
def extract_results(results, dataset_label):
    rows = []
    for season, result in results.items():
        for idx, param in enumerate(result.params):
            param_name = 'Intercept' if idx == 0 else (f'lag_{idx}' if idx <= n_lags else 'nao_index_cdas')
            rows.append({
                'Dataset': dataset_label,
                'Season': season,
                'Parameter': param_name,
                'Coefficient': param,
                'Conf_Lower': result.conf_int(alpha=0.05)[idx][0],  # 90% confidence lower bound
                'Conf_Upper': result.conf_int(alpha=0.05)[idx][1]   # 90% confidence upper bound
            })
    return pd.DataFrame(rows)

results_old_df = extract_results(seasonal_results_old, 'test.old')
results_new_df = extract_results(seasonal_results_new, 'test.new')

# Combine results into a single DataFrame for comparison
comparison_df = pd.concat([results_old_df, results_new_df])

# Display the comparison DataFrame
print(comparison_df)
