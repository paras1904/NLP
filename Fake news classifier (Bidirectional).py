import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from tensorflow.keras.layers import Embedding, Dense, LSTM, Bidirectional
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences

df = pd.read_csv('fake_news_data.csv')
df = df.dropna()

X = df.drop('label',axis=1)
y = df['label']

voc_size = 5000
messages = X.copy()
messages.reset_index(inplace=True)
# print(msg['title'][0])

ps = PorterStemmer()
corpus = []
for i in range(0,len(messages)):
    review = re.sub('[^a-zA-Z]',' ',messages['title'][i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = " ".join(review)
    corpus.append(review)
# print(corpus)

one_hot_rep = [one_hot(word,voc_size) for word in corpus]
print(len(sorted(one_hot_rep)[-1]))

sent_length = 20
embeded_doc = pad_sequences(one_hot_rep,maxlen=sent_length,padding='pre')
print(embeded_doc[0])
embedding_vector_features = 40
model = Sequential([
    Embedding(voc_size,embedding_vector_features,input_length=sent_length),
    Bidirectional(LSTM(100)),
    Dense(1,activation='sigmoid')
])
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
# print(model.summary())

import numpy as np
X_final = np.array(embeded_doc)
y_final = np.array(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_final,y_final,test_size=0.33,random_state=42)
model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=10,batch_size=100)

y_pred = model.predict(X_test)
print(y_pred)