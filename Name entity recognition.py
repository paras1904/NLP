import nltk
import pandas as pd

texts = "apple acquired zoom in china and usa on wednesday 6 may."
words = nltk.word_tokenize(texts)
# print(words)
pos_tag = nltk.pos_tag(words)
# print(pos_tag)
chunks = nltk.ne_chunk(pos_tag, binary=True)
for chunk in chunks:
    if list(chunk)[1]=="NN":
        print(list(chunk)[0])