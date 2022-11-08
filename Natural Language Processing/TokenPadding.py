# %%
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

sentences = [ 'I Love my dog', 'I love my cat', 'You love my dog', 'Do you think my dog is amazing']
tk = Tokenizer(num_words = 100, oov_token = "<OOV>")
tk.fit_on_texts(sentences)
wordindex  = tk.word_index
seq = tk.texts_to_sequences(sentences)

print("\n Word Index : ", wordindex)
print("\n Sequences : ", seq)

# %%
padded = pad_sequences(seq, )
print("\n Padded Sequences :")
print(padded)


