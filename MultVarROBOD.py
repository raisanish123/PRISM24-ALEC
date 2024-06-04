#edited by S. Rai to fix the inverse scaling and add accuracy
#Alec Judd Vidanes: Multivariate Time-Series Code.
#This code was procured from an article by Jason Brownlee.
#With help from ChatGPT, I modified the code to accept inputs of environmental factors to predict Occupancy Estimation.
import numpy as np
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
from sklearn.model_selection import StratifiedKFold
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.metrics import MeanSquaredError

#Dropped for now, it doesn't seem like it does much for us at least.
#Anything in triple quotes is dropped code, but maybe will be used if I can understand it better?
'''
# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values (This was causing an index issue when spliting with KFold
    if dropnan:
        agg.dropna(inplace=True)
    return agg
'''
# load dataset
#Self explanatory stuff
dataset = read_csv('combined_H1_edited.csv', header=0, index_col=0)
dataset = dataset.dropna(how='any', axis=0)
sample = 7
dataset = dataset.iloc[::sample]
X = dataset.drop(['occupied','number'], axis=1)
y = dataset['number']

'''
dataset = read_csv('combined_Room1.csv', header=0, index_col=0)
dataset = dataset.dropna(how='any', axis=0)
X = dataset.drop(['occupant_presence [binary]','occupant_count [number]'], axis=1)
y = dataset['occupant_count [number]']
X.to_csv('preframed.csv')
'''
# normalize features
#A scaler to change a range of values to a certain smaller value
scaler = MinMaxScaler(feature_range=(0, 1))
#scaler = StandardScaler()
#takes said values, and converts them all to a range of feature_range
scaled = scaler.fit_transform(X.values)

def create_dataset(data,labels,seq_length,overlap,future_step):
  X,y = [],[]
  for i in range(0, len(data)-seq_length +1, overlap):
    X.append(data[i:i+seq_length])
    y.append(labels[i+seq_length+future_step])
  X = np.array(X)
  X = X.astype('float32')
  print("X shape",X.shape)
  print(X[0][0])
  y = np.array(y)
  y = y.astype('long')
  ysize = y.shape[0]
  print(y[ysize-1])
  print("Y shape", y.shape)
  return X,y
seq_length = 500
overlap = 100
future_step = 0
X,y = create_dataset(X, y,seq_length,overlap,future_step)
#Returns a numpy.ndarray of stuff.
values = X
#values = values.drop(['date','occupied'], axis=1)

# ensure all data is float
#values = values.astype('float32')


# frame as supervised learning
#returns the dimensions of an ndarray, 1 dimension returns the size of the array.
#0 indicates row size, 1 indicates column size
n_features = values.shape[2]


#reframed = series_to_supervised(scaled, 1, 1)
#reframed.to_csv('reframed.csv')
#y = y[:-1]
# drop columns we don't want to predict
# Assuming the last column is the occupancy column we want to predict


#reframed.drop(reframed.columns[[n_features]], axis=1, inplace=True)
#print(reframed.head())
#print("Reframed original columns:",reframed.shape[1])


# split into train and test sets
#values = reframed.values
xr, xc ,xw= values.shape
yc = y.shape
print("(" + str(xr) + "," + str(xc) + "," + str(xw) + "),(" + str(yc) + ")")
'''
n_train_hours = int(len(values) * 0.7)  # using 70% of data for training
train = values[:n_train_hours, :]
test = values[n_train_hours:, :]
'''


'''train = train.delete(['var1(t)','var36(t)','var37(t)'], axis=1)
test = test.delete(['var1(t)','var36(t)','var37(t)'], axis=1)
'''
# split into input and outputs
# number of objects to use for predictions, the "1" indicates timesteps for some other forecasting type.
n_obs = (n_features-1) * 1  # number of features  excluding the label* timesteps
print("No of features",n_obs)

#SKFOLD
skf = StratifiedKFold(n_splits=5)
seq_length = 1000
overlap = 50

for fold, (train_index, test_index) in enumerate(skf.split(values[:, :, :n_obs], y)):
  print(f"Fold {fold + 1}")
  print("Training_index: " + str(train_index.shape) + "\nTesting_index: " + str(test_index.shape))
  #Training values, will set the inputs and outputs to the values that are being trained into the machine
  train_X, train_y = values[train_index,:, :n_obs], y[train_index]

  # print("Training data")
  # print(train_X.shape[1])
  # for i in range(10):
  #   print("Train x:",train_X[i][-1])
  #   print("Train y:",train_y[i])

  #Testing values, will be used to compare for accuracy.
  print(train_X.shape)
  test_X, test_y = values[test_index, : ,:n_obs], y[test_index]

  # reshape input to be 3D [samples, timesteps, features]
  #train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
  #test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
  print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

  # design networks
  #Creates the sequential model
  model = Sequential()
  #Determines this to be an LSTM model, which utilizes the training values 1 is timestep and 2 is the features
  model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))

  model.add(Dense(1))
  model.compile(loss='mae', optimizer='adam',metrics=['accuracy', MeanSquaredError()])

  # fit network
  #This is where the predictions are made, for each ep
  history = model.fit(train_X, train_y, epochs=50, batch_size=72, validation_data=(test_X, test_y), verbose=2, shuffle=False)

  # plot history
  pyplot.plot(history.history['loss'], label='train')
  pyplot.plot(history.history['val_loss'], label='test')
  pyplot.legend()
  pyplot.savefig('loss.png')
  #pyplot.show()

  # plot history
  pyplot.plot(history.history['accuracy'], label='train_accuracy')
  pyplot.plot(history.history['val_accuracy'], label='test_accuracy')
  pyplot.legend()
  pyplot.savefig('accuracy.png')
  #pyplot.show()

  # make a prediction
  print("Predicting here:")
  yhat = model.predict(test_X)
  print(yhat.shape)
  #debug
  inv_yhat = yhat
  #yhat = yhat.reshape(yhat.shape[0],yhat.shape[1],1)
  #test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))

  # invert scaling for forecast
  #inv_yhat = concatenate((yhat, test_X[:, :,:n_obs]), axis=1)
  #inv_yhat = scaler.inverse_transform(inv_yhat)
  #inv_yhat = inv_yhat[:,0]

  # invert scaling for actual
  test_y = test_y.reshape(len(test_y), 1)
  inv_y = test_y
  #inv_y = concatenate((test_y, test_X[:, :, :n_obs]), axis=1)
  #inv_y = scaler.inverse_transform(inv_y)
  #inv_y = inv_y[:,0]

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
