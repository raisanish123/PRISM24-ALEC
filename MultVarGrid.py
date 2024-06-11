import numpy as np
import pandas as pd
import os
import datetime
from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from scikeras.wrappers import KerasRegressor
from keras.metrics import MeanSquaredError

# Load dataset
dataset = read_csv('combined_Room1.csv', header=0, index_col=0)
dataset = dataset.dropna(how='any', axis=0)
dataset = dataset[dataset['occupant_count [number]'] > 0]
X = dataset.drop(['occupant_presence [binary]', 'occupant_count [number]'], axis=1)
y = dataset['occupant_count [number]']

# Normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_X = scaler.fit_transform(X.values)

def create_dataset(data,labels,seq_length,overlap,future_step):
  X,y = [],[]
  for i in range(0, len(data)-seq_length +1, overlap):
    if(i+seq_length>=len(data) or i+seq_length+future_step>=len(labels)):
      continue
    X.append(data[i:i+seq_length])
    y.append(labels[i+seq_length+future_step])
  X = np.array(X)
  # ensure all data is float
  X = X.astype('float32')
  print("X shape",X.shape)
  print(X[0][0])
  y = np.array(y)
  y = y.astype('long')
  ysize = y.shape[0]
  print(y[ysize-1])
  print("Y shape", y.shape)
  return X,y

# Define function to create model
def create_model(lstm_units, dropout_rate):
    model = Sequential()
    model.add(LSTM(lstm_units, input_shape = (None, X.shape[1]), dropout=dropout_rate, recurrent_dropout=dropout_rate))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam', metrics=['accuracy', MeanSquaredError()])
    return model

# Define parameter grid
param_grid = {
    'lstm_units': [50, 100, 200],
    'dropout_rate': [0.2, 0.4, 0.6],
    'batch_size': [16, 32, 64],
    'epochs': [10, 20]
}

# Wrap the model using KerasRegressor
model = KerasRegressor(dropout_rate = 0.2, lstm_units = 50, model=create_model, verbose=0)

# Set up GridSearchCV
tscv = TimeSeriesSplit(n_splits=5)
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=tscv, scoring='neg_mean_squared_error')

# Fit GridSearchCV
Dseq_length = 24
overlap = 5
future_step = 1

X, y = create_dataset(scaled_X, y.values, Dseq_length, overlap, future_step)

grid_result = grid_search.fit(X, y)

# Print best parameters and best score
print(f"Best parameters: {grid_result.best_params_}")
print(f"Best score: {grid_result.best_score_}")

# You can then use the best parameters to train your final model or evaluate further
best_model = grid_result.best_estimator_.model

# Proceed with your evaluation as needed
# e.g., use best_model to predict and evaluate
