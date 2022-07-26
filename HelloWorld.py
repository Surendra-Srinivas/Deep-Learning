import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense
from keras import Sequential
import numpy as np

x = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype = float)
y = np.array([7.0, 9.0, 11.0, 13.0, 15.0], dtype  = float)

model = Sequential()
model.add(Dense(1, input_shape=[1]))
model.compile(optimizer='sgd', loss='mse')
model.fit(x, y, epochs=1000)
print(model.predict([10.0,20.0]))
