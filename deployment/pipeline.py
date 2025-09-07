# Libraries

import sys
import os
import warnings

sys.path.append(os.path.abspath('ml_models'))
warnings.filterwarnings('ignore')

from ml_models.package import ml_package
from joblib import Parallel, delayed
import os
import pandas as pd
import pickle
import shutil

# Data Loading & Pre-Processing

row_data = pd.read_csv('../data/row_data_rovere.csv')

clean_data, sensors_mean = ml_package.pre_processing(row_data)

clean_data.to_csv('../data/clean_data_rovere.csv', index=False)
with open('../data/sensors_mean.pkl', 'wb') as f:
    pickle.dump(sensors_mean, f)

clean_data = clean_data.drop(columns='date')

tens_30 = ['72', '76', '73', '74', '61', '63', '67', '65']
tens_60 = ['71', '69', '75', '70', '62', '64', '68', '66']
tens_all = tens_30 + tens_60

# Model Saving

if os.path.exists('models'):
    shutil.rmtree('models')
os.makedirs('models')

Parallel(n_jobs=-1)(delayed(ml_package.save_model_with_strategy)(sensor_id, clean_data, strategy)
                    for sensor_id in tens_all
                    for strategy in ['aic', 'pls'])

shutil.make_archive('models', 'zip', 'models')
shutil.rmtree('models')
