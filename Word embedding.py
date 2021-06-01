from tensorflow.keras.preprocessing.text import one_hot
sent = ['the glass of milk','the glass of juice','the cup of tea','I am a good boy','I am a good developer','understand the meaning of words','your videos are good',]
# print(sent)
voc_size = 1000
one_hot_rep = [one_hot(words,voc_size) for words in sent]
# print(one_hot_rep)

from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
import numpy as np

sent_length = 8
embeded_docs = pad_sequences(one_hot_rep,padding='pre',maxlen=sent_length)
print(embeded_docs)
dim = 10
model = Sequential([
    Embedding(voc_size,10,input_length=sent_length)
])
model.compile(optimizer='adam',loss='mse')
print(model.summary())
print(model.predict(embeded_docs))