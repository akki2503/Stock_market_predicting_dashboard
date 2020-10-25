import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential
from keras.layers import Dense,RepeatVector
from keras.layers import Flatten
from keras.layers import TimeDistributed
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import LSTM, Flatten
from keras.layers import Dropout
from keras.layers import RepeatVector, TimeDistributed

import tensorflow as tf
print(tf.__version__)
import time, sys

def prepare_train_data(raw_data, train_fraction=0.8):
    n_train = int(len(raw_data)*0.8)
    ## prepare lstm data unsaceld
    x_train_l = raw_data.values[:,6:11]
    x_train_sub = x_train_l[0:n_train]
    x_train_lstm = x_train_sub.reshape(x_train_sub.shape[0],1,x_train_sub.shape[1])

    x_test_sub = x_train_l[n_train:]
    x_test_lstm = x_test_sub.reshape(x_test_sub.shape[0],1,x_test_sub.shape[1])

    ## y train common
    y_train_lstm = raw_data.values[:,11].reshape(-1,1)[0:n_train]
    y_test_lstm = raw_data.values[:,11].reshape(-1,1)[n_train:]

    print(x_train_lstm.shape, y_train_lstm.shape, y_test_lstm.shape)
    
    x_train_lstm = np.asarray(x_train_lstm).astype(np.float32)
    y_train_lstm = np.asarray(y_train_lstm).astype(np.float32)
    x_test_lstm = np.asarray(x_test_lstm).astype(np.float32)
    y_test_lstm = np.asarray(y_test_lstm).astype(np.float32)

    return (x_train_lstm, y_train_lstm, x_test_lstm, y_test_lstm)

def lstm_model():
    lstm_model = Sequential()
    lstm_model.add(LSTM(50, batch_input_shape=(1, 1, 5),
                   return_sequences=True, activation='relu', recurrent_activation='hard_sigmoid'))
    lstm_model.add(LSTM(100, return_sequences=True, activation='relu', 
                   recurrent_activation='hard_sigmoid'))
    lstm_model.add(LSTM(50, activation='relu', recurrent_activation='hard_sigmoid'))
    lstm_model.add(Dense(16))
    lstm_model.add(Dense(1))
    lstm_model.compile(loss='mean_squared_error', optimizer='adam')
    lstm_model.compile(loss='mse', optimizer='adam')
    print(lstm_model.summary())
    return lstm_model

def train_lstm(lstm_model, n_epochs=10, n_nested_epochs=10):
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss',patience=5, mode='min')
    for i in range(n_nested_epochs):
        lstm_model.fit(x_train_lstm, y_train_lstm, 
                       epochs=n_epochs, batch_size=1)
        lstm_model.reset_states()
    lstm_model.save(model_name)
    return lstm_model

def forecast(model, x, n_batch):
    fore = list()
    x = x.reshape(1, 1, x.shape[1])
    forecast = model.predict(x, batch_size=n_batch)
    x_new = np.array([x for x in forecast[0,:]])
    x_prev = x.reshape(5)
    x_next = np.append(x_prev[1:5],x_new)
    fore =np.append(fore, x_new)
    
    for i in range(4):
        x = x_next.reshape(1, 1, 5)
        forecast = model.predict(x, batch_size=n_batch)
        x_new = [x for x in forecast[0,:]]
        x_prev = x.reshape(5)
        x_next = np.append(x_prev[1:5],x_new)
        fore = np.append(fore, x_new)
    
    return fore

def forecast_lstm(x_test_lstm):
    forecasts = list()
    for i in range(len(x_test_lstm)):
        x= x_test_lstm[i,:]
        fore = forecast(lstm_model, x , 1)
        forecasts = np.append(forecasts, fore)
    return forecasts

def predictions(raw_data):
    x_train_lstm, y_train_lstm, x_test_lstm, y_test_lstm = prepare_train_data(raw_data)
    lstm_model = lstm_model()
    lstm_model = train_lstm(lstm_model)
    y_pred = forecast_lstm(x_test_lstm)
    return y_pred

