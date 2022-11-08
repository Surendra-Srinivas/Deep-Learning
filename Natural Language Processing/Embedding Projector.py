# %%
import tensorflow_datasets as tfds

# %%
imdb, info = tfds.load("imdb_reviews", with_info = True, as_supervised=True)
print(info)

# %%
for example in imdb['train'].take(2):
    print(example)

# %%
import numpy as np

train_data, test_data = imdb['train'], imdb['test']

training_sentences = []
training_labels = []

testing_sentences = []
testing_labels = []

# Loop over all training examples and save the sentences and labels
for s,l in train_data:
    training_sentences.append(s.numpy().decode('utf8'))
    training_labels.append(l.numpy())
# Loop over all test examples and save the sentences and labels
for s,l in test_data:
  testing_sentences.append(s.numpy().decode('utf8'))
  testing_labels.append(l.numpy())

# Convert labels lists to numpy arrays :
training_labels_final = np.array(training_labels)
testing_labels_final = np.array(testing_labels)

# %%
# Hyper Parameters : 

vocab_size = 10000
max_length = 120
embedding_dim  = 16
trunc_type = 'post'
oov_tok = "<OOV>"

# %%
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tok = Tokenizer(num_words = vocab_size, oov_token = oov_tok)

tok.fit_on_texts(training_sentences)
word_index = tok.word_index

seq = tok.texts_to_sequences(training_sentences)
padded = pad_sequences(seq, maxlen = max_length, truncating = trunc_type)

testing_sequences = tok.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequences, maxlen = max_length, truncating = trunc_type)

# %%
import tensorflow as tf

model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length = max_length))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(6, activation = 'relu'))
model.add(tf.keras.layers.Dense(1, activation = 'sigmoid'))

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
model.summary()

# %%
num_epochs = 10
model.fit(padded, training_labels_final, epochs = num_epochs, validation_data=(testing_padded, testing_labels_final))


# %% [markdown]
# VISUALIZE THE WORD EMBEDDINGS

# %%
embedding_layer  = model.layers[0]
embedding_weights = embedding_layer.get_weights()[0]
print(embedding_weights.shape) # Expected shape : (vocab_size, embedding_dim)

# %%
reverse_word_index  = tok.index_word

# %%
import io

out_v = io.open('vecs.tsv', 'w', encoding='utf-8')
out_m = io.open('meta.tsv', 'w', encoding = 'utf-8')

for word_num in range(1, vocab_size):
    word_name = reverse_word_index[word_num]
    word_embedding = embedding_weights[word_num]
    out_m.write(word_name + "\n")
    out_v.write("\t".join([str(x) for x in word_embedding]) + "\n")

out_v.close()
out_m.close()


