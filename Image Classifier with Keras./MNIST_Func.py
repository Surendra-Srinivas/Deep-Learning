# An End to End Functional API Implementation :
from keras.layers import *
from keras.datasets import mnist
import numpy as np
import keras

classnames = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
def preprocessing():
    # Loading the dataset and Splitiing
    (x_train, y_train), (x_test, y_test)  = mnist.load_data()
    # Reshaping the data and converting it into 32 bit float .
    x_train = x_train.reshape(60000, 784).astype('float32')/255
    x_test = x_test.reshape(10000, 784).astype('float32')/255
    
    y_train = y_train.astype('float32')
    y_test = y_test.astype('float32')
    # Splitting data for validation and training.
    x_val = x_train[-10000:]
    y_val = y_train[-10000:]

    x_train = x_train[:-10000]
    y_train = y_train[:-10000]

    return x_val, y_val, x_test, y_test, x_train, y_train

def uncompiled_model():
    input = keras.Input(shape=(784,), name='digits')
    hidden1 = Dense(64, activation='relu', name='hidden1')(input)
    hidden2 = Dense(64, activation='relu', name='hidden2')(hidden1)
    output = Dense(10, activation='softmax', name='outputlayer')(hidden2)
    model = keras.Model(inputs = input, outputs = output)
    # Model groups layers into an object with training and inference features.
    return model

def compiled_model():
    model = uncompiled_model()
    model.compile(loss='sparse_categorical_crossentropy', optimizer = 'rmsprop', metrics = ['accuracy'])

    return model

def modelfitting():
    x_val, y_val, x_test, y_test, x_train, y_train = preprocessing()
    model = compiled_model()
    model.fit(x_train, y_train, batch_size=64, epochs=3, validation_data=(x_val, y_val))
    results = model.evaluate(x_test, y_test, batch_size=128)
    print("test loss, test acc:", results)
    preds = model.predict(x_test[:3])
    print("predictions shape:", preds.shape)
    print(preds)
    y_proba = np.argmax(preds, axis  =1)
    print(y_proba)
    print(np.array(classnames)[y_proba])
    return     
    
model = modelfitting()