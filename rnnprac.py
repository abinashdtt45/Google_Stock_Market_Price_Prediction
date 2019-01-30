#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 15:02:32 2019

@author: abinash
"""

import numpy as np
import pandas as pd
import keras
import matplotlib.pyplot as plt

#importing the dataset
google_train_set = pd.read_csv('Google_Stock_Price_Train.csv')
google_train_set.describe()
google_train_set = google_train_set.iloc[:, 1:2].values

#Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
google_train_set = sc.fit_transform(google_train_set)

#Getting the inputs and outputs for thr observations
X_train = google_train_set[0:1257]
Y_train = google_train_set[1:1258]

#Reshaping the dataset for timestamps as [er Keras documentation
X_train = np.reshape(X_train, (1257, 1, 1))



#importing keras libraries
from keras import Sequential
from keras.layers import Dense
from keras.layers import LSTM

#initializing the regressor
regressor = Sequential()

#Building up the layers
regressor.add(LSTM(units=4, activation='tanh', input_shape=(None, 1)))
regressor.add(Dense(units=1))

regressor.compile(optimizer='adam', loss='mean_squared_error')
regressor.fit(X_train, Y_train, batch_size=32, epochs=200 )

#for the test data set
test_set = pd.read_csv('Google_Stock_Price_Test.csv')
test_set = test_set.iloc[:, 1:2].values
real_prices = test_set

test_set = sc.fit_transform(test_set)
test_set = np.reshape(test_set,(20,1,1))

predicted_stock_price = regressor.predict(test_set)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

#visualizing the results
plt.plot(real_prices, color='red', label='Real_Stock_Prices')
plt.plot(predicted_stock_price, color='blue', label='Predicted_Stock_Prices')
plt.xlabel('Time')
plt.ylabel('Stock_Prices')
plt.title('Google_Stock_Price_Prediction')

