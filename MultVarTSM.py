#edited by S. Rai to fix the inverse scaling and add accuracy
#Alec Judd Vidanes: Multivariate Time-Series Code.
#This code was procured from an article by Jason Brownlee.
#With help from ChatGPT, I modified the code to accept inputs of environmental factors to predict Occupancy Estimation.

from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
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
    #if dropnan:
    #    agg.dropna(inplace=True)
    return agg
'''
# load dataset
dataset = read_csv('combined_H1_edited.csv', header=0, index_col=0)
dataset = dataset.dropna(how='any', axis=0)
X = dataset.drop(['occupied','number'], axis=1)
y = dataset['number']
X.to_csv('preframed.csv')
values = X.values
#values = values.drop(['date','occupied'], axis=1)

# ensure all data is float
values = values.astype('float32')

# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)

# frame as supervised learning
n_features = values.shape[1]

'''
reframed = series_to_supervised(scaled, 1, 1)
reframed.to_csv('reframed.csv')
'''

# drop columns we don't want to predict
# Assuming the last column is the occupancy column we want to predict

'''
reframed.drop(reframed.columns[[n_features]], axis=1, inplace=True)
#print(reframed.head())
print("Reframed original columns:",reframed.shape[1])
'''

# split into train and test sets
values = scaled

xr, xc = values.shape
yc = y.shape
print("(" + str(xr) + "," + str(xc) + "),(" + str(yc) + ")")
'''
n_train_hours = int(len(values) * 0.7)  # using 70% of data for training
train = values[:n_train_hours, :]
test = values[n_train_hours:, :]
'''


'''train = train.delete(['var1(t)','var36(t)','var37(t)'], axis=1)
test = test.delete(['var1(t)','var36(t)','var37(t)'], axis=1)
'''
# split into input and outputs
n_obs = (n_features-1) * 1  # number of features  excluding the label* timesteps
print("No of features",n_obs)

#SKFOLD
skf = StratifiedKFold(n_splits=5)
seq_length = 100
overlap = 10

for fold, (train_index, test_index) in enumerate(skf.split(values[:, :n_obs], y)):
  print(f"Fold {fold + 1}")
  train_X, train_y = values[train_index, :n_obs], values[train_index, -1]
  # print("Training data")
  # print(train_X.shape[1])
  # for i in range(10):
  #   print("Train x:",train_X[i][-1])
  #   print("Train y:",train_y[i])

  print(train_X.shape)
  test_X, test_y = values[test_index, :n_obs], values[test_index, -1]

  # reshape input to be 3D [samples, timesteps, features]
  train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
  test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
  print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

  # design network
  model = Sequential()
  model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
  model.add(Dense(1))
  model.compile(loss='mae', optimizer='adam',metrics=['accuracy', MeanSquaredError()])

  # fit network
  history = model.fit(train_X, train_y, epochs=200, batch_size=72, validation_data=(test_X, test_y), verbose=2, shuffle=False)

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
  test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))

  # invert scaling for forecast
  inv_yhat = concatenate((yhat, test_X[:, :n_obs]), axis=1)
  inv_yhat = scaler.inverse_transform(inv_yhat)
  inv_yhat = inv_yhat[:,0]

  # invert scaling for actual
  test_y = test_y.reshape((len(test_y), 1))
  inv_y = concatenate((test_y, test_X[:, :n_obs]), axis=1)
  inv_y = scaler.inverse_transform(inv_y)
  inv_y = inv_y[:,0]

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
