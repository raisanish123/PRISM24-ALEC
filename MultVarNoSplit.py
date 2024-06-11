#edited by S. Rai to fix the inverse scaling and add accuracy
#Alec Judd Vidanes: Multivariate Time-Series Code.
#This code was procured from an article by Jason Brownlee.
#With help from ChatGPT, I modified the code to accept inputs of environmental factors to predict Occupancy Estimation.
import numpy as np
import pandas as pd
import seaborn as sns
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
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.metrics import MeanSquaredError

#Dropped for now, it doesn't seem like it does much for us at least.
# load dataset
#Self explanatory stuff
#OG
dataset = read_csv('combined_H1_edited.csv', header=0, index_col=0)
#dataset = dataset.dropna(how='any', axis=0)
# Fill missing values
dataset.fillna(method='ffill', inplace=True)
dataset.fillna(method='bfill', inplace=True)
#dataset = dataset[dataset['number'] > 0]
#dataset = dataset.sample(frac = 1)
#sample = 7
#dataset = dataset.iloc[::sample]
X = dataset.drop(['occupied','number'], axis=1)
y = dataset['number']
'''
#Robod
dataset = read_csv('combined_Room1.csv')
#dataset = dataset.dropna(how='any', axis=0)
# Fill missing values
dataset.fillna(method='ffill', inplace=True)
dataset.fillna(method='bfill', inplace=True)
dataset = dataset[dataset['occupant_count [number]'] > 0]
X = dataset.drop(['occupant_presence [binary]','occupant_count [number]'], axis=1)
y = dataset['occupant_count [number]']
'''



X.to_csv('preframed.csv')
# normalize features
#A scaler to change a range of values to a certain smaller value
scaler = MinMaxScaler(feature_range=(0, 1))
#scaler = StandardScaler()
#takes said values, and converts them all to a range of feature_range
scaled = scaler.fit_transform(X.values)

#data_scaled = np.hstack((features_scaled, np.array(y).reshape(-1, 1)))


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

#OG data gampling
seq_length = 20
overlap = 1
future_step = 0
'''
#Robod sampling
seq_length = 20
overlap = 1
future_step = 0
'''
X,y = create_dataset(scaled, y,seq_length,overlap,future_step)
#Returns a numpy.ndarray of stuff.

# frame as supervised learning
#returns the dimensions of an ndarray, 1 dimension returns the size of the array.
#0 indicates row size, 1 indicates sequence size, and 2 indicates features
n_features = X.shape[1]

#values = reframed.values
xr, xc, xw= X.shape
yc = y.shape

print(np.unique(y))
print("(" + str(xr) + "," + str(xc) + "," + str(xw) + "),(" + str(yc) + ")")


# number of objects to use for predictions, the "1" indicates timesteps for some other forecasting type.
n_obs = (n_features) * 1  # number of features  excluding the label* timesteps
print("No of features",n_obs)

directory = str(datetime.datetime.now())

path = os.path.join(".",directory)

os.mkdir(path)

os.chdir(path)

#Training values, will set the inputs and outputs to the values that are being trained into the machine
train_X, test_X, train_y, test_y = train_test_split(X,y,test_size=0.2, shuffle = True, random_state = 42)

# Define the LSTM model
input_size = X.shape[2]
hidden_size = 50
num_layers = 2
output_size = 5  # Number of classes

model = Sequential()
model.add(LSTM(hidden_size, return_sequences=True, input_shape=(seq_length, input_size)))
for _ in range(num_layers - 1):
    model.add(LSTM(hidden_size, return_sequences=False))
model.add(Dense(output_size, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
num_epochs = 30
batch_size = 64

history = model.fit(train_X, train_y, epochs=num_epochs, batch_size=batch_size, validation_data=(test_X, test_y), shuffle=True)
filename = 'loss' + '.png'
# plot history
pyplot.figure()
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.savefig(filename)

filename = 'accuracy' + '.png'
# plot history
pyplot.figure()
pyplot.plot(history.history['accuracy'], label='train_accuracy')
pyplot.plot(history.history['val_accuracy'], label='test_accuracy')
pyplot.legend()
pyplot.savefig(filename)

# make a prediction
print("Predicting here:")
yhat = model.predict(test_X)
yhat_classes = np.argmax(yhat, axis=1)
print(yhat.shape)
inv_yhat = yhat_classes

# invert scaling for actual
inv_y = test_y
filename = 'prediction_vs_actual' + '.png'
pyplot.figure()
pyplot.plot(yhat, label='Predicted')
pyplot.plot(test_y, label='Actual')
pyplot.legend()
pyplot.savefig(filename)

cm = confusion_matrix(test_y,yhat_classes)
np.set_printoptions(threshold = np.inf)
confusion = pd.DataFrame(cm)
filename = 'confusion_matrix' + '.png'
pyplot.figure(figsize=(10,7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
pyplot.xlabel('Predicted Labels')
pyplot.ylabel('True Labels')
pyplot.title('Confusion Matrix')
pyplot.savefig(filename)

# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)

# Calculate average accuracy
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

avg_train_acc = sum(train_acc) / len(train_acc)
avg_val_acc = sum(val_acc) / len(val_acc)

print("Average Training Accuracy:", avg_train_acc)
print("Average Validation Accuracy:", avg_val_acc)

