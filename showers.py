import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from QAR_persistence_precip import QAR_climate
import datetime

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
def create_lagged_features(data, n_lags, quantiles, include_NAO):
    lagged_data = data.copy()
    
    # Create lagged features for 'Temp'
    for lag in range(1, n_lags + 1):
        lagged_data[f'lag_{lag}'] = lagged_data['Temp'].shift(lag)
    if include_NAO ==True:
    # Create categorical indicators for 'nao_index_cdas'
        lagged_data['nao_index_cdas'] = lagged_data['nao_index_cdas'].shift(1)  # Shift nao_index_cdas
        lagged_data['nao_index_cdas_cat'] = pd.qcut(lagged_data['nao_index_cdas'], quantiles, labels=False)
    
        # Convert categorical indicator into dummy variables
        nao_index_dummies = pd.get_dummies(lagged_data['nao_index_cdas_cat'], prefix='nao_index_cat', drop_first=True)
        lagged_data = pd.concat([lagged_data, nao_index_dummies], axis=1)
    
        # Drop original categorical indicator column and rows with NaN values due to lagging and shifting
        lagged_data = lagged_data.drop(['nao_index_cdas', 'nao_index_cdas_cat'], axis=1).dropna()
    
    return lagged_data

def within_15_days(data_index, month, day):
    specific_date = datetime.datetime(year=2000, month=month, day=day)  # Year doesn't matter
    start_date = specific_date - datetime.timedelta(days=15)
    end_date = specific_date + datetime.timedelta(days=15)

    # Create a mask
    mask = ((data_index.month == start_date.month) & (data_index.day >= start_date.day)) | \
           ((data_index.month > start_date.month) & (data_index.month < end_date.month)) | \
           ((data_index.month == end_date.month) & (data_index.day <= end_date.day)) | \
           ((start_date.month > end_date.month) & \
           (((data_index.month == start_date.month) & (data_index.day >= start_date.day)) | \
           ((data_index.month == end_date.month) & (data_index.day <= end_date.day))))

    return mask
    
def fit_ar_logistic_regression(data, n_lags, months, quantiles, include_NAO, month=8, day=15):
    # Filter data for the specified months (e.g., winter)
    #winter_data = data[data.index.month.isin(months)].copy()

    mask = within_15_days(data.index, month, day)
    data = data[mask].copy()    
    # Create lagged features
    seasonal_data = create_lagged_features(data, n_lags, quantiles, include_NAO).dropna()

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
        param_name = 'Intercept' if idx == 0 else f'lag_{idx}' if idx <= n_lags else f'nao_index_cat_{idx  - n_lags}'
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
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
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
    
    ax.set_xlabel('Parameters')
    ax.set_ylabel('Coefficients')
    ax.set_title('Coefficients and 95% Confidence Bounds: Old vs New for ' + test.sCity)
    ax.legend()
    plt.xticks(ticks=range(len(parameters)), labels=parameters, rotation=45)
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

def simulate_paths(df_sorted, seasonal_results, y_prec_season, iS=1000):
    iT = len(y_prec_season.loc[y_prec_season.index.year == y_prec_season.index.year[-1]])
    predictions = seasonal_results.predict(df_sorted)
    mB = pd.DataFrame(0, index=range(iS), columns=range(iT))
    mB.iloc[:, 0] = np.random.binomial(1, y_prec_new_season.mean(), iS)
    for s in np.arange(1, iT):
        mB.iloc[:, s] = np.random.binomial(1, predictions[mB.iloc[:, s-1]])
    return mB

def backtest_model(mB, y_test, month, mm_threshold, random_days = 10, ticket_price=1, replace_days=False):
    festival_days = np.sort(np.random.choice(np.arange(1,31), random_days, replace=replace_days))
    mB_selected = mB.iloc[:, festival_days]
    mB_selected_sum = mB_selected.sum(axis=1)
    payout_sum = np.sum(mB_selected_sum * ticket_price)
    y_test_days = y_test.loc[(y_test.index.day.isin(festival_days)) & (y_test.index.month == month)]
    y_test_to_pay = len(y_test_days.loc[y_test_days.Temp>=mm_threshold])
    print(y_test_to_pay)
    
start_year = str(1990) + '-'
end_year = str(int(start_year[:-1]) + 30) + '-'

test = QAR_climate(sCity='DE BILT', use_statsmodels=True, include_nao=True, newend=end_year, newstart=start_year)
test.prepare_data()
y_test = test.predict
# Number of lags and months
n_lags = 1
months = [8] 
quantiles = [0, 1]
month = 8
day = 15
predict0, predict1 = [1,0], [1,1]
mm_threshold = 45
outcome_threshold = .5

# Generate example binary time series data for test.old
y_prec_old = (test.old.Temp >= mm_threshold) * 1
data_old = pd.DataFrame(y_prec_old, columns=['Temp'])
data_old['nao_index_cdas'] = test.old.nao_index_cdas
mask_old = within_15_days(data_old.index, month, day)
y_prec_old_season = y_prec_old.loc[mask_old]

# Generate example binary time series data for test.new
y_prec_new = (test.new.Temp >= mm_threshold) * 1
data_new = pd.DataFrame(y_prec_new, columns=['Temp'])
data_new['nao_index_cdas'] = test.new.nao_index_cdas
mask_new = within_15_days(data_new.index, month, day)
y_prec_new_season = y_prec_new.loc[mask_new]

# Assign season to each row
data_old['season'] = data_old.index.month.map(get_season).values
data_new['season'] = data_new.index.month.map(get_season).values



# Fit models for both datasets
seasonal_results_old, X_old, y_old = fit_ar_logistic_regression(data_old, n_lags, months, quantiles, test.include_nao, month, day)
seasonal_results_new, X_new, y_new = fit_ar_logistic_regression(data_new, n_lags, months, quantiles, test.include_nao, month, day)

df_new_ = pd.DataFrame(X_new).drop_duplicates().reset_index(drop=True)
df_new_['first_one_index'] = df_new_.apply(first_one_index, axis=1)
df_sorted_new = df_new_.sort_values(by=[1, 'first_one_index'] + list(df_new_.columns), ascending=[True, True] + [True]*len(df_new_.columns)).drop('first_one_index', axis=1).reset_index(drop=True)




#print predictions
print(seasonal_results_new.predict([predict0, predict1]))
print(seasonal_results_old.predict([predict0, predict1]))

results_old_df = extract_results(seasonal_results_old, 'test.old')
results_new_df = extract_results(seasonal_results_new, 'test.new')

# Combine results into a single DataFrame for comparison
comparison_df = pd.concat([results_old_df, results_new_df])

# Display the comparison DataFrame
print(comparison_df)

plot_coefficients(comparison_df)



rows_0_new, rows_1_new = compare_rows(X_new, predict0), compare_rows(X_new, predict1)
print(y_prec_new_season.iloc[rows_0_new + n_lags].mean(), y_prec_new_season.iloc[rows_1_new + n_lags].mean())
rows_0_old, rows_1_old = compare_rows(X_old, predict0), compare_rows(X_old, predict1)
print(y_prec_old_season.iloc[rows_0_old + n_lags].mean(), y_prec_old_season.iloc[rows_1_old + n_lags].mean())

y_pred_new = seasonal_results_new.predict(X_new)
y_pred_old = seasonal_results_new.predict(X_old)
#predict outcomes
predicted_choice_new = (y_pred_new > outcome_threshold).astype(int)
predicted_choice_old = (y_pred_old > outcome_threshold).astype(int)


mB = simulate_paths(df_sorted_new, seasonal_results_new, y_prec_new_season, iS=100000)






