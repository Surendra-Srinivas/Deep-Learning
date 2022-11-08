# %%
import tensorflow_datasets as tfds
import tensorflow  as tf
import numpy as np

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# %%
imdb, info = tfds.load('imdb_reviews', with_info  =True, as_supervised = True)

# %%
train_data, test_data = imdb['train'], imdb['test']

train_sentences = []
test_sentences = []

train_labels = []
test_labels = []

for s, l in train_data:
    train_sentences.append(s.numpy().decode('utf8'))
    train_labels.append(l.numpy())

for s,l in test_data:
    test_sentences.append(s.numpy().decode('utf8'))
    test_labels.append(l.numpy())

train_labels = np.array(train_labels)
test_labels = np.array(test_labels)

# %%
import matplotlib.pyplot as plt

def plot_graph(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_'+string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val'+string])
    plt.show()

# %%
vocab_size = 10000
max_length = 120
trunc_type = 'post'
oov_token = "<OOV>"

tok = Tokenizer(num_words = vocab_size, oov_token = oov_token)

tok.fit_on_texts(train_sentences)
word_index = tok.word_index

seq = tok.texts_to_sequences(train_sentences)
padded = pad_sequences(seq, maxlen = max_length, truncating=trunc_type)

testing_seq = tok.texts_to_sequences(test_sentences)
test_padded = pad_sequences(testing_seq, maxlen = max_length)

# %% [markdown]
# Model 1: Flatten

# %%
embedding_dim = 16
dense_dim = 6

model_flatten = tf.keras.Sequential()
model_flatten.add(tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length))
model_flatten.add(tf.keras.layers.Flatten())
model_flatten.add(tf.keras.layers.Dense(dense_dim, activation='relu'))
model_flatten.add(tf.keras.layers.Dense(1, activation = 'sigmoid'))

model_flatten.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics=  ['accuracy'])
model_flatten.summary()

# %%
num_epochs = 10
Batch_size = 128

history_flatten = model_flatten.fit(padded, train_labels, batch_size=Batch_size, epochs=num_epochs, validation_data=(test_padded, test_labels))
plot_graph(history_flatten, 'accuracy')
plot_graph(history_flatten, 'loss')

# %% [markdown]
# Model 2: LSTM

# %%
embedding_dim = 16
lstm_dim  = 32
dense_dim = 6

model_lstm = tf.keras.Sequential()
model_lstm.add(tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length))
model_lstm.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_dim)))
model_lstm.add(tf.keras.layers.Dense(dense_dim, activation='relu'))
model_lstm.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model_lstm.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
model_lstm.summary()

# %%
num_epochs = 10
Batch_size = 128

history_lstm = model_lstm.fit(padded, train_labels, batch_size = Batch_size, epochs = num_epochs, validation_data =(test_padded, test_labels) )
plot_graph(history_lstm, 'accuracy')
plot_graph(history_lstm, 'loss')

# %% [markdown]
# MODEL 03: GRU

# %%
import  tensorflow as tf

embedding_dim = 16
gru_dim  =32
dense_dim = 6

model_gru = tf.keras.Sequential()
model_gru.add(tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length))
model_gru.add(tf.keras.layers.Bidirectional(tf.keras.layers.GRU(gru_dim)))
model_gru.add(tf.keras.layers.Dense(dense_dim, activation = 'relu'))
model_gru.add(tf.keras.layers.Dense(1, activation = 'sigmoid'))

model_gru.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
model_gru.summary()

# %%
num_epoch  =10
batch_size = 128

history_gru = model_gru.fit(padded, train_labels, batch_size = Batch_size,epochs = num_epoch, validation_data=(test_padded, test_labels))
plot_graph(history_gru, 'accuracy')
plot_graph(history_gru,'loss')

# %% [markdown]
# MODEL 04 : CONVOLUTION

# %%
import tensorflow as tf

embedding_dim = 16
filters = 16
kernel_size = 5
dense_dim = 6

model_conv = tf.keras.Sequential()
model_conv.add(tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length))
model_conv.add(tf.keras.layers.Conv1D(filters, kernel_size, activation='relu'))
model_conv.add(tf.keras.layers.GlobalAveragePooling1D())
model_conv.add(tf.keras.layers.Dense(dense_dim, activation='relu'))
model_conv.add(tf.keras.layers.Dense(1, activation = 'sigmoid'))

model_conv.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
model_conv.summary()

# %%
num_epochs = 10
Batch_size = 128

history_conv = model_conv.fit(padded, train_labels, batch_size = Batch_size, epochs = num_epochs, validation_data = (test_padded, test_labels))
plot_graph(history_conv, "accuracy")
plot_graph(history_conv, 'loss')


