# Libraries

import datetime
from itertools import chain, combinations
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import pickle
import statsmodels.api as sm
from scipy.spatial.distance import cdist
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_predict
import warnings

warnings.filterwarnings('ignore')


# Error Metrics

def r_squared(y_true, y_pred):
    return r2_score(y_true, y_pred)


def mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)


def mape(y_true, y_pred):
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred, squared=False)


def calculate_aic(y_true, y_pred, num_features):
    resid = y_true - y_pred
    sse = np.sum(np.square(resid))
    return len(y_true) * np.log(sse / len(y_true)) + 2 * num_features


def calculate_bic(y_true, y_pred, num_features):
    resid = y_true - y_pred
    sse = np.sum(np.square(resid))
    n = len(y_true)
    return n * np.log(sse / n) + num_features * np.log(n)


def calculate_distortion(x_normalized, cluster_centers):
    min_distances = np.min(cdist(x_normalized, cluster_centers, 'euclidean'), axis=1)
    return np.sum(min_distances) ** 2


# Bidirectional Stepwise Selection

def calculate_criterion(x_train, y_train, features, criterion_func):
    model = sm.OLS(y_train, sm.add_constant(x_train[features])).fit()
    predicted_values = model.predict(sm.add_constant(x_train[features]))
    criterion = criterion_func(y_train, predicted_values, len(model.params))
    return criterion


def forward_step(x_train, y_train, selected_features, remaining_features, criterion_func):
    best_criterion = np.inf
    best_feature = None

    for feature in remaining_features:
        temp_features = selected_features + [feature]
        temp_criterion = calculate_criterion(x_train, y_train, temp_features, criterion_func)

        if temp_criterion < best_criterion:
            best_criterion = temp_criterion
            best_feature = feature

    return best_criterion, best_feature


def backward_step(x_train, y_train, selected_features, criterion_func):
    best_criterion = np.inf
    best_feature = None

    for feature in selected_features:
        temp_features = selected_features.copy()
        temp_features.remove(feature)
        temp_criterion = calculate_criterion(x_train, y_train, temp_features, criterion_func)

        if temp_criterion < best_criterion:
            best_criterion = temp_criterion
            best_feature = feature

    return best_criterion, best_feature


def stepwise_bidirectional_selection(x_train, y_train, method='aic'):
    criterion_funcs = {'aic': calculate_aic, 'bic': calculate_bic}
    if method not in criterion_funcs:
        raise ValueError("Invalid Criterion. Use 'aic' or 'bic'.")

    criterion_func = criterion_funcs[method]
    features = list(x_train.columns)
    selected_features = []
    best_criterion = np.inf

    while features:
        forward_criterion, forward_feature = forward_step(x_train, y_train, selected_features, features, criterion_func)
        backward_criterion, backward_feature = backward_step(x_train, y_train, selected_features, criterion_func)

        if forward_criterion < backward_criterion and forward_criterion < best_criterion:
            selected_features.append(forward_feature)
            best_criterion = forward_criterion
        elif backward_criterion < best_criterion:
            selected_features.remove(backward_feature)
            best_criterion = backward_criterion
        else:
            break

    return selected_features


# Pre-Processing Rovere

def compute_residuals(data, df, sensor):
    subset = data[data['sensor_id'] == sensor].copy()
    x = subset.drop(['sensor_id', 'date', 'avg_tens', 'avg_tens_lag1', 'avg_tens_lag2', 'avg_tens_lag3'], axis=1)
    y = subset['avg_tens']

    selected_features = stepwise_bidirectional_selection(x, y, method='aic')
    model = sm.OLS(y, sm.add_constant(x[selected_features])).fit()

    intercept = model.params[0]
    intercept_vector = np.full(3, intercept)

    pred_sensor = model.predict(sm.add_constant(x[selected_features]))
    combined_vector = np.concatenate((intercept_vector, pred_sensor))

    selected_values = df.loc[df['sensor_id'] == sensor, 'avg_tens']
    values = selected_values - combined_vector
    res = list(zip(values))

    return res


def pre_processing(data):
    # Columns Selection and Formatting

    df_rovere = data[['reading_id', 'timestamp', 'sensor_id', 'value', 'description', 'group_id']].copy()
    df_rovere[['reading_id', 'sensor_id', 'description', 'group_id']] = df_rovere[
        ['reading_id', 'sensor_id', 'description', 'group_id']].astype(str)
    df_rovere['timestamp'] = pd.to_datetime(df_rovere['timestamp']).dt.floor('D').dt.date
    df_rovere['value'] = df_rovere['value'].astype(float)

    tens_30 = ['72', '76', '73', '74', '61', '63', '67', '65']
    tens_60 = ['71', '69', '75', '70', '62', '64', '68', '66']
    tens_all = tens_30 + tens_60

    df_rovere.loc[df_rovere['description'] == 'tensiometer', 'description'] = 'Tensiometer'
    df_rovere.loc[df_rovere['description'] == 'irrigation', 'description'] = 'Irrigation'

    # Duplication

    df_dup = df_rovere[~df_rovere['sensor_id'].isin(tens_30)]
    df_dup['group_id'] = df_dup['group_id'] + '_dup'

    df_rovere = df_rovere[~df_rovere['sensor_id'].isin(tens_60)]
    df_rovere = pd.concat([df_rovere, df_dup], ignore_index=True)
    df_rovere.sort_values(by=['group_id', 'timestamp'], inplace=True)

    # Grouping and Creation of Summary Values

    df_group = df_rovere.groupby(['timestamp', 'description', 'sensor_id', 'group_id']).agg(
        {'value': ['min', 'max', 'mean', 'median', 'sum']}).reset_index()
    df_group.columns = ['timestamp', 'description', 'sensor_id', 'group_id', 'val_min', 'val_max', 'val_avg', 'val_med',
                        'val_sum']

    # Pivoting

    df_pivot = df_group.pivot(index=['timestamp', 'group_id'], columns='description',
                              values=['val_min', 'val_max', 'val_avg', 'val_med', 'val_sum']).reset_index()
    df_pivot.columns = ['date', 'group_id'] + [f"{agg}_{feature}" for agg in ['min', 'max', 'avg', 'med', 'sum'] for
                                               feature in ['hum', 'temp', 'solar', 'wind', 'irr', 'rain', 'tens']]

    df_pivot.reset_index(drop=True, inplace=True)

    # Sensor ID Mapping

    group_id_mapping = {str(i): str(j) for i, j in zip(range(1, 9), tens_30)}
    group_id_mapping.update({str(i) + '_dup': str(j) for i, j in zip(range(1, 9), tens_60)})
    df_pivot['group_id'] = df_pivot['group_id'].replace(group_id_mapping)

    df = df_pivot.rename(columns={'group_id': 'sensor_id'})
    columns_to_drop = ['min_irr', 'max_irr', 'avg_irr', 'med_irr', 'min_rain', 'avg_rain', 'sum_hum', 'sum_temp',
                       'sum_solar', 'min_wind', 'max_wind', 'avg_wind', 'sum_wind', 'med_wind']
    df.drop(columns=columns_to_drop, inplace=True)

    df = df[
        ['sensor_id', 'date', 'avg_tens'] + [col for col in df.columns if col not in ['sensor_id', 'date', 'avg_tens']]]
    df = df.sort_values(by=['sensor_id', 'date']).reset_index(drop=True)

    # Imputation of Missing Values

    float_columns = df.select_dtypes(include=['float']).columns
    df[float_columns] = df[float_columns].interpolate(method='linear', limit_direction='both')

    # Shifting Values using the previous 3 Days

    dates = df['date']
    x = df.drop(columns=['date', 'sensor_id'])
    x = x.shift(1).add_suffix('_lag1').join(x.shift(2).add_suffix('_lag2')).join(x.shift(3).add_suffix('_lag3'))

    x['date'] = dates
    dates_to_remove = [datetime.date(2023, 4, 28), datetime.date(2023, 4, 29), datetime.date(2023, 4, 30)]
    x = x[~x['date'].isin(dates_to_remove)].reset_index(drop=True)
    x = x.drop(columns='date')

    y = df[['sensor_id', 'date', 'avg_tens']]
    y = y[~y['date'].isin(dates_to_remove)].reset_index(drop=True)

    df_merged = pd.concat([y, x], axis=1)

    # Computation of the Residuals from a LM

    sensor_ids = tens_all
    sensor_ids.sort()

    results = Parallel(n_jobs=-1)(delayed(compute_residuals)(df_merged, df, sensor) for sensor in sensor_ids)

    residuals = []
    for res in results:
        residuals.extend(res)

    df_residuals = pd.DataFrame(residuals, columns=['residuals'])
    df_residuals = df_residuals.shift(1).add_suffix('_lag1').join(df_residuals.shift(2).add_suffix('_lag2')).join(
        df_residuals.shift(3).add_suffix('_lag3'))
    df_residuals = pd.concat([df['date'], df_residuals], axis=1)
    df_residuals = df_residuals[~df_residuals['date'].isin(dates_to_remove)].reset_index(drop=True)
    df_residuals = df_residuals.drop(columns='date')

    data = pd.concat([df_merged, df_residuals], axis=1)

    # Creation of an Array of Means

    data_lm = data.copy()
    mean_df = []

    for sensor_id in tens_all:
        data_sensor = data_lm[data_lm['sensor_id'] == sensor_id].reset_index(drop=True)
        tens_values = data_sensor['avg_tens']
        sensor_mean = np.mean(tens_values)
        mean_df.append({'sensor_id': sensor_id, 'sensor_mean': sensor_mean})

    mean_df = pd.DataFrame(mean_df)

    return data, mean_df


def pre_processing_target(sensor_id, target, sensors_mean):
    df_rovere = target.copy()
    sensor_id = str(sensor_id)

    # Formatting

    df_rovere['timestamp'] = pd.to_datetime(df_rovere['timestamp']).dt.floor('D').dt.date
    df_rovere['sensor_id'] = df_rovere['sensor_id'].astype(str)
    df_rovere['value'] = df_rovere['value'].astype(float)
    df_rovere['description'] = df_rovere['description'].astype(str)

    df_rovere = df_rovere.sort_values(by=['timestamp', 'description']).reset_index(drop=True)

    # Grouping and Creation of Summary Values

    df_group = df_rovere.groupby(['timestamp', 'description', 'sensor_id']).agg(
        {'value': ['min', 'max', 'mean', 'median', 'sum']}).reset_index()
    df_group.columns = ['timestamp', 'description', 'sensor_id', 'val_min', 'val_max', 'val_avg', 'val_med', 'val_sum']

    # Pivoting

    df_pivot = df_group.pivot(index=['timestamp'], columns='description',
                              values=['val_min', 'val_max', 'val_avg', 'val_med', 'val_sum']).reset_index()
    df_pivot.columns.name = None

    if df_pivot.shape[1] == 36:
        df_pivot.columns = ['date'] + [f"{agg}_{feature}" for agg in ['min', 'max', 'avg', 'med', 'sum'] for feature in
                                       ['hum', 'temp', 'solar', 'wind', 'irr', 'rain', 'tens']]

    elif df_pivot.shape[1] == 31:
        df_pivot.columns = ['date'] + [f"{agg}_{feature}" for agg in ['min', 'max', 'avg', 'med', 'sum'] for feature in
                                       ['hum', 'temp', 'solar', 'irr', 'rain', 'tens']]

    else:
        print('Error: The Number of Features is not Correct')

    columns_to_drop = ['min_irr', 'max_irr', 'avg_irr', 'med_irr', 'min_rain', 'avg_rain', 'sum_hum', 'sum_temp',
                       'sum_solar', 'min_wind', 'max_wind', 'avg_wind', 'sum_wind', 'med_wind']
    columns_present = [col for col in columns_to_drop if col in df_pivot.columns]

    if columns_present:
        df = df_pivot.drop(columns=columns_present).reset_index(drop=True)

    else:
        df = df_pivot.copy()

    df = df[['date', 'avg_tens'] + [col for col in df.columns if col not in ['sensor_id', 'date', 'avg_tens']]]

    # Creates a Row for the next day's observation

    y = df['avg_tens']
    # y = y[-3:].reset_index(drop=True)
    last_date = df['date'].iloc[-1]
    target_date = last_date + datetime.timedelta(days=1)
    new_obs = [target_date] + [np.nan] * (len(df.columns) - 1)
    df.loc[len(df)] = new_obs

    x = df.drop(columns=['date'])
    x = x.shift(1).add_suffix('_lag1').join(x.shift(2).add_suffix('_lag2')).join(x.shift(3).add_suffix('_lag3'))
    x_target = pd.DataFrame(x.iloc[-1]).transpose().reset_index(drop=True)

    # Computation of Residuals

    y_mean = sensors_mean.loc[sensors_mean['sensor_id'] == sensor_id, 'sensor_mean'].values[0]
    predict_lm = [y_mean] * 3

    # Using more than 3 days
    # selected_features = stepwise_bidirectional_selection(X_train, y_train, method='aic')
    # model = sm.OLS(y_train, sm.add_constant(X_train[selected_features])).fit()
    # predict_lm =  model.predict(sm.add_constant(x[selected_features]))

    residuals = y - predict_lm
    residuals = np.append(residuals, np.nan)
    df_residuals = pd.DataFrame(residuals, columns=['residuals'])
    df_residuals = df_residuals.shift(1).add_suffix('_lag1').join(df_residuals.shift(2).add_suffix('_lag2')).join(
        df_residuals.shift(3).add_suffix('_lag3'))
    last_row_res = df_residuals.iloc[-1].to_frame().transpose().reset_index(drop=True)

    x_target = pd.concat([x_target, last_row_res], axis=1).sort_index(axis=1)

    return x_target


# Forecasting Models (ARIMAX)

def evaluate_subset_pls(subset, data, sensor):
    df_train = data[data['sensor_id'].isin(subset)]
    df_test = data[data['sensor_id'] == sensor]

    x_train = df_train.drop(['sensor_id', 'avg_tens'], axis=1)
    y_train = df_train['avg_tens']

    x_test = df_test.drop(['sensor_id', 'avg_tens'], axis=1)
    y_test = df_test['avg_tens']

    model = sm.OLS(y_train, sm.add_constant(x_train)).fit()
    pred_sensor = model.predict(sm.add_constant(x_test))

    rmse_subset = round(rmse(y_test, pred_sensor), 3)

    return subset, rmse_subset


def optimize_ncomp_pls(x, y, n_comp_range, objective):
    errors = []
    ticks = np.arange(1, n_comp_range + 1)

    for n_comp in ticks:
        pls = PLSRegression(n_components=n_comp)
        y_cv = cross_val_predict(pls, x, y, cv=10)
        error = rmse(y, y_cv)
        errors.append(error)

    if objective == 'min':
        return ticks[np.argmin(errors)]
    else:
        return ticks[np.argmax(errors)]


def save_model(sensor_id, data_train, fs_strategy='aic'):
    sensor_id = str(sensor_id)

    # Dictionary for the Cluster associated to each Tensiometer

    subsets_df = pd.DataFrame({
        'sensor_id': ['72', '71', '76', '69',
                      '73', '75', '74', '70',
                      '61', '62', '63', '64',
                      '67', '68', '65', '66'],

        'subset_aic': [['62', '67', '68', '71'], ['61', '64', '70'], ['65', '66', '69', '73'], ['65', '66', '73', '76'],
                       ['65', '66', '75', '76'], ['68', '73', '76'], ['62', '63', '70'], ['64', '65', '68', '69', '71'],
                       ['62', '70', '71', '75'], ['61', '64'], ['64', '67'], ['65', '66', '69', '74'],
                       ['68'], ['67'], ['66', '69', '73', '76'], ['65', '76']],

        'subset_pls': [['62', '63', '67', '68', '71', '74'], ['61', '64', '70', '72', '75'], ['65', '66', '69', '73'],
                       ['65', '66', '73', '76'],
                       ['65', '66', '69', '75', '76'], ['62', '64', '67', '68', '73'],
                       ['62', '63', '67', '68', '70', '72'], ['61', '64', '71', '74', '75'],
                       ['62', '63', '64', '70', '71', '72', '75'], ['61', '64', '67', '68', '75'],
                       ['61', '62', '64', '67', '68', '72', '74'], ['62', '63', '67', '68', '75'],
                       ['62', '63', '64', '68', '72', '74', '75'], ['62', '63', '64', '67', '72', '74', '75'],
                       ['66', '69', '73', '76'], ['65', '69', '73', '76']]
    })

    # AIC Criterion

    if fs_strategy == 'aic':

        best_subset = subsets_df.loc[subsets_df['sensor_id'] == sensor_id, 'subset_aic'].values[0]

        df_train = data_train[data_train['sensor_id'].isin(list(best_subset))]
        x_train = df_train.drop(['sensor_id', 'avg_tens'], axis=1).sort_index(axis=1)
        y_train = df_train['avg_tens']

        selected_features = stepwise_bidirectional_selection(x_train, y_train, method='aic')
        model = sm.OLS(y_train, sm.add_constant(x_train[selected_features])).fit()

        return model, selected_features


    # PLS Method

    elif fs_strategy == 'pls':

        all_sensors = subsets_df.loc[subsets_df['sensor_id'] == sensor_id, 'subset_pls'].values[0]
        all_subsets = list(chain.from_iterable(combinations(all_sensors, r) for r in range(1, len(all_sensors) + 1)))

        df_train = data_train[data_train['sensor_id'].isin(list(all_sensors))]
        ids_train = df_train[['sensor_id']].reset_index(drop=True)
        x_train = df_train.drop(['sensor_id', 'avg_tens'], axis=1).sort_index(axis=1)
        y_train = df_train[['avg_tens']].reset_index(drop=True)

        df_val = data_train[data_train['sensor_id'] == sensor_id]
        ids_val = df_val[['sensor_id']].reset_index(drop=True)
        x_val = df_val.drop(['sensor_id', 'avg_tens'], axis=1).sort_index(axis=1)
        y_val = df_val[['avg_tens']].reset_index(drop=True)

        best_n_comp = optimize_ncomp_pls(x_train, y_train, 50, 'RMSE')

        pls_labels = [f'PC{i}' for i in range(1, best_n_comp + 1)]
        pls_model = PLSRegression(n_components=best_n_comp)
        pls_model.fit(x_train, y_train)

        x_train_pls = pls_model.transform(x_train)
        x_train_pls = pd.DataFrame(x_train_pls, columns=pls_labels)
        x_val_pls = pls_model.transform(x_val)
        x_val_pls = pd.DataFrame(x_val_pls, columns=pls_labels)

        train_pls = pd.concat([ids_train, y_train, x_train_pls], axis=1)
        val_pls = pd.concat([ids_val, y_val, x_val_pls], axis=1)
        data_pls = pd.concat([train_pls, val_pls], ignore_index=True)

        best_subset = None
        best_rmse = float('inf')

        results = Parallel(n_jobs=-1)(
            delayed(evaluate_subset_pls)(subset, data_pls, sensor_id) for subset in all_subsets)

        for subset, rmse_subset in results:
            if rmse_subset < best_rmse:
                best_rmse = rmse_subset
                best_subset = subset

        df_pls = data_pls[data_pls['sensor_id'].isin(list(best_subset))]
        x_train = df_pls.drop(['sensor_id', 'avg_tens'], axis=1)
        y_train = df_pls['avg_tens']

        model = sm.OLS(y_train, sm.add_constant(x_train)).fit()

        return model, pls_model

    else:
        print('Error: The Features Selection Method is not Correct')


def save_model_with_strategy(sensor_id, clean_data, strategy):
    model, fs = save_model(sensor_id, clean_data, fs_strategy=strategy)
    with open(f"models/{sensor_id}-arimax_{strategy}-model.pkl", 'wb') as f_model:
        pickle.dump(model, f_model)
    with open(f"models/{sensor_id}-arimax_{strategy}-fs.pkl", 'wb') as f_fs:
        pickle.dump(fs, f_fs)


def model_predict(model, features_selection, x_target):
    prediction = None
    x_target_copy = x_target.copy()

    # Methods that Reduce the Number of Features
    if isinstance(features_selection, list):
        x_target_copy['const'] = 1
        prediction = model.predict(x_target_copy[['const'] + features_selection])

    # Methods based on Dimensionality Reduction creating new Features
    elif hasattr(features_selection, 'transform'):
        n_comp = features_selection.n_components
        pls_labels = [f'PC{i}' for i in range(1, n_comp + 1)]
        x_target_transformed = features_selection.transform(x_target_copy)
        x_target_transformed = pd.DataFrame(x_target_transformed, columns=pls_labels)
        x_target_transformed['const'] = 1
        prediction = model.predict(x_target_transformed[['const'] + pls_labels])

    else:
        print('Error: The feature selection method is not supported.')

    return prediction
