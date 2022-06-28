import keras
from keras import Sequential
import tensorflow as tf
import ssl
import numpy as np
ssl._create_default_https_context = ssl._create_unverified_context

fashion_mnist = tf.keras.datasets.fashion_mnist
(X_train_full,Y_train_full),(X_test,Y_test) = fashion_mnist.load_data()
print(X_train_full.shape)
print(X_train_full.dtype)

X_valid, X_train = X_train_full[:5000]/255.0, X_train_full[5000:]/255.0
Y_valid, Y_train = Y_train_full[:5000], Y_train_full[5000:]
model = Sequential()
model.add(keras.layers.Flatten(input_shape=[28,28]))
model.add(keras.layers.Dense(300,activation='relu'))
model.add(keras.layers.Dense(100,activation='relu'))
model.add(keras.layers.Dense(10,activation='softmax'))
model.summary()

#model.layers # get a list of layers
#layer = model.layers[1] # Accessing Layers
#layer.name
#model.get_layer('dense_30') is layer

model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd', metrics=['accuracy'] )

history = model.fit(X_train,Y_train,epochs = 30, validation_data=(X_valid,Y_valid))
model.evaluate(X_test, Y_test)

classnames = ['T-shirrt/top', "Trouser"," Pullover", " Dress", "Coat", "Sandal", "Shirt", " Sneaker", "Bag", "Ankle boot"]
X_new = X_test[:3]
y_prob = model.predict(X_new)
print(y_prob)
# y_proba = model.predict_classes(X_new) , This feature is removed from Tensorflow 2.6
y_proba = np.argmax(y_prob,axis=1)
print(y_proba)
np.array(classnames)[y_proba]
print(history.history)