# An End to End Functional API  Implementation :

import keras
from keras.layers import *
from keras.datasets import mnist

def uncompiled_model():
    #Functional API
    input = keras.Input(shape=(784, ), name = 'digits')
    layer1 = Dense(64, activation='relu', name='Dense_1')(input)
    layer2 = Dense(64, activation='relu', name='Dense_2' )(layer1)
    output = Dense(10, activation='softmax', name='outputlayer')(layer2)
    model = keras.Model(inputs = input, outputs = output)
    return model

def compiled_model():
    model = uncompiled_model()
    model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model

def Preproccesing():
    # Data Splitting
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # Data Reshaping
    x_train = x_train.reshape(60000,784).astype("float32") / 255
    x_test = x_test.reshape(10000,784).astype("float32") / 255
    # Converting the Data into float 32 bit.
    y_train = y_train.astype('float32')
    y_test = y_test.astype('float32')
    # Splitting data for validation and training.
    x_val = x_train[-10000:]
    y_val = y_train[-10000:]
    x_train = x_train[:-10000]
    y_train = y_train[:-10000]
        
    return x_val, y_val, x_test, y_test, x_train, y_train

def modelfitting():
    x_val, y_val, x_test, y_test, x_train, y_train = Preproccesing()
    model = compiled_model()
    model.fit(x_train, y_train, batch_size=64, epochs=2, validation_data=(x_val, y_val))
    results = model.evaluate(x_test, y_test, batch_size=128)
    print("test loss, test acc:", results)
    preds = model.predict(x_test[:3])
    print("predictions shape:", preds.shape)
    return 

model = modelfitting()
