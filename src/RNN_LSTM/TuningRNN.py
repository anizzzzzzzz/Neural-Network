import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset_train = pd.read_csv('/home/aniz/PycharmProjects/NeuralNetwork/src/RNN_LSTM/data/Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
training_set_scaled = sc.fit_transform(training_set)

# Creating data structure with 60 timesteps and 1 output
X_train = []
y_train = []
for i in range(60, training_set.shape[0]):
    X_train.append(training_set_scaled[i-60 : i, 0])
    y_train.append(training_set_scaled[i])

# Need to convert into numpy array
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping .. RNN needs 3D input
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Tuning ANN
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM

def build_regressor(optimizer):
    regressor = Sequential()

    regressor.add(LSTM(units=50, return_sequences=True, input_shape=(X_train[1], 1)))
    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units=50, return_sequences=True))
    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units=50, return_sequences=True))
    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units=50))
    regressor.add(Dropout(0.2))

    regressor.add(Dense(units=1))

    regressor.compile(optimizer=optimizer, loss='mean_squared_error')
    return regressor

from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV

regressor = KerasRegressor(build_fn=build_regressor)

parameters = {
    'batch_size' : [10, 25, 32],
    'nb_epoch' : [50, 100],
    'optimizer' : ['adam', 'rmsprop']
}

grid_search = GridSearchCV(estimator=regressor, param_grid=parameters, scoring='neg_mean_squared_error', cv=None)
grid_search.fit(X_train[:,:,-1], y_train)

best_param = grid_search.best_params_
best_accuracy = grid_search.best_score_