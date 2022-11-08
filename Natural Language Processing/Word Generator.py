# %%
import tensorflow as tf
import numpy as np

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# %%
path = "tensorflow-1-public-main/C3/W4/misc/Laurences_generated_poetry.txt"
data = open(path).read()
corpus = data.lower().split("\n")
#print(corpus)

# %%
tok = Tokenizer()
tok.fit_on_texts(corpus)
total_words  = len(tok.word_index)+1
#print(f'word index dictionary: {tok.word_index}')
print(f'total words: {total_words}')

# %%
input_seq = []
for line in corpus:
    token_list  = tok.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_seq = token_list[:i+1]
        input_seq.append(n_gram_seq)
max_seq_len = max([len(x) for x in input_seq])
input_seq = np.array(pad_sequences(input_seq, maxlen=max_seq_len, padding = 'pre'))
xs, labels = input_seq[:, :-1], input_seq[:, -1]
ys = tf.keras.utils.to_categorical(labels, num_classes=total_words)

# %%
embedding_dim  =100
lstm_units = 150
lr = 0.01

model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(total_words, embedding_dim, input_length=max_seq_len-1))
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_units)))
model.add(tf.keras.layers.Dense(total_words, activation='softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer = tf.keras.optimizers.Adam(learning_rate=lr), metrics = ['accuracy'])
model.summary()

# %%
history = model.fit(xs, ys, epochs = 100)

# %%
import matplotlib.pyplot as plt

def plot_graph(history, string):
    plt.plot(history.history[string])
    plt.xlabel('Epochs')
    plt.ylabel(string)
    plt.show()

plot_graph(history, 'accuracy')

# %% [markdown]
# GENERATING TEXT

# %%
seed_text = "Surendra went to"
next_words = 100

for _ in range(next_words):
    # Convert the seed text to a token sequence
    token_list = tok.texts_to_sequences([seed_text])[0]
    # Pad the Sequence.
    token_list = pad_sequences([token_list], maxlen=max_seq_len-1, padding='pre')
    # Feed to the model and get the probabilities for each index
    probs = model.predict(token_list)
    # Get the index with the highest Probability.
    predicted = np.argmax(probs, axis=-1)[0]
    if predicted != 0: # Maybe its just a padding zero.
        output_word = tok.index_word[predicted]
        seed_text+= " "+ output_word
print(seed_text)

# %%
seed_text = "In the town of Athy one Jeremy Lanigan"

next_words = 20
for _ in range(next_words):
  token_list = tok.texts_to_sequences([seed_text])[0]
  token_list = pad_sequences([token_list], maxlen=max_seq_len-1, padding = 'pre') # why max_seq_len - 1. ?
  probs = model.predict(token_list)
  choice  =np.random.choice([1, 2, 3])
  predicted = np.argsort(probs)[0][-choice]
  if predicted != 0:
    output_word = tok.index_word[predicted]
    seed_text += " "+output_word

print(seed_text)


