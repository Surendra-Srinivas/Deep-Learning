# %%
import json

with open('sarcasm1.json', 'r') as f:
    datastore = json.load(f)

sentences = []
labels = []

for item in datastore:
    sentences.append(item['headline'])
    labels.append(item['is_sarcastic'])

# %%
#import wget

#!wget https://storage.googleapis.com/tensorflow-1-public/course3/sarcasm.json

# %%
training_size = 20000

training_sentences = sentences[0:training_size]
testing_sentences = sentences[training_size:]

training_labels = labels[0:training_size]
testing_labels = labels[training_size:]

# %%
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

vocab_size = 10000
max_length = 120
trunc_type = 'post'
padding_type = 'post'
oov_tok = '<OOV>'

# %%

tok = Tokenizer(num_words = vocab_size, oov_token=oov_tok)

tok.fit_on_texts(training_sentences)
word_index = tok.word_index

training_seq = tok.texts_to_sequences(training_sentences)
training_padded = pad_sequences(training_seq, maxlen=max_length, padding = padding_type)

testing_seq = tok.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_seq, maxlen=max_length, padding=  padding_type)

training_labels = np.array(training_labels)
testing_labels = np.array(testing_labels)

# %%
import tensorflow as tf

embedding_dim = 16
dense_dim = 6
kernel_size = 5
filters= 128

model_conv = tf.keras.Sequential()
model_conv.add(tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length = max_length))
model_conv.add(tf.keras.layers.Conv1D(filters, kernel_size, activation = 'relu'))
model_conv.add(tf.keras.layers.GlobalAveragePooling1D())
model_conv.add(tf.keras.layers.Dense(dense_dim, activation = 'relu'))
model_conv.add(tf.keras.layers.Dense(1, activation = 'sigmoid'))

model_conv.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
model_conv.summary()

# %%
num_epochs = 10
history_conv = model_conv.fit(training_padded, training_labels, epochs = num_epochs, validation_data = (testing_padded, testing_labels))

import matplotlib.pyplot as plt

def plot_graph(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_'+string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_'+string])
    plt.show()

plot_graph(history_conv, 'accuracy')
plot_graph(history_conv, 'loss')


