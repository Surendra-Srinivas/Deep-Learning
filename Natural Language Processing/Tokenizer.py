# %%
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer


# %%
sentences = [ 'I love my dog', 'I love my cat', 'You love my dog!', 'Do you think my dog is amazing?']
tokenizer = Tokenizer(num_words=100, oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
print(word_index)
# The problem with ASCII Values is that LISTEN and SILENT has same set of ASCII values ;-)

# %%
sequences = tokenizer.texts_to_sequences(sentences)
print(sequences)
test_data = ['i really love my dog!', 'my dog loves my manager@']
test_seq = tokenizer.texts_to_sequences(test_data)
print(test_seq)


