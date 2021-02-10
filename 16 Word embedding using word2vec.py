# vid 22
from nltk import sent_tokenize , word_tokenize
import warnings
import gensim
from gensim.models import Word2Vec
sample = open('storydata.txt','r')
s = sample.read()

import nltk
f = s.replace('\n','')
data = []
for i in sent_tokenize(f):
    temp = []
    for j in word_tokenize(i):
        temp.append(j.lower())
    data.append(temp)
print(data)
model1 = gensim.models.Word2Vec(data,min_count = 1,size = 1000,window = 5)
a1 = model1.similarity('camel','farmer')
print('skip garam')
model2 = gensim.models.Word2Vec(data,min_count=1,size=1000,window=5,sg=1)
a2 = model2.similarity('camel','thieves')
print(a1)
print(a2)