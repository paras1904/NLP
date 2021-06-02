import pandas as pd

df = pd.read_csv('fake_news_data.csv')
# print(df.head())

df = df.dropna()
X = df.drop('label',axis=1)
y = df['label']
# print(X.shape, y.shape)

from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import LSTM,Dense

voc_size = 5000
messages = X.copy()
messages.reset_index(inplace = True)

import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()
corpus = []
for i in range(0, len(messages)):
    # print(i)
    review = re.sub('[^a-zA-Z]', ' ', messages['title'][i])
    review = review.lower()
    review = review.split()

    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)
# print(corpus)

one_hot_repr = [one_hot(words,voc_size) for words in corpus]
# print(one_hot_repr)

sent_length = 20
embedded_docs = pad_sequences(one_hot_repr,padding='pre',maxlen=sent_length)

# print(embedded_docs)
# print(embedded_docs[0])
# print(len(embedded_docs))

embedding_vector_features = 40
model = Sequential([
    Embedding(voc_size,embedding_vector_features,input_length=sent_length),
    LSTM(100),
    Dense(1,activation='sigmoid'),
])
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
# print(model.summary())

import numpy as np
X_final = np.array(embedded_docs)
y_final = np.array(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_final,y_final,test_size=0.33,random_state=42)
model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=10,batch_size=100)

y_pred = model.predict(X_test)
print(y_pred)
