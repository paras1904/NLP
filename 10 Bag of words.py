# vid 11
from sklearn.feature_extraction.text import CountVectorizer
text = ['the quick brown fox jumped over the lazy dog']
vectoriser = CountVectorizer()
vectoriser.fit(text)
print(vectoriser.vocabulary_)

vector = vectoriser.transform(text)
print(vector.shape)
print(type(vector))
print(vector.toarray())
print()

import numpy as np
import re

def tokenize_sentences(sentences):
    words = []
    for sentence in sentences:
        w = extract_words(sentence)
        words.extend(w)

    words = sorted(list(set(words)))
    return words


def extract_words(sentence):
    ignore_words = ['a']
    words = re.sub("[^w]", " ", sentence).split()  # nltk.word_tokenize(sentence)
    words_cleaned = [w.lower() for w in words if w not in ignore_words]
    return words_cleaned


def bagofwords(sentence, words):
    sentence_words = extract_words(sentence)
    # frequency word count
    bag = np.zeros(len(words))
    for sw in sentence_words:
        for i, word in enumerate(words):
            if word == sw:
                bag[i] += 1

    return np.array(bag)
sentences = ["Machine learning is great", "Natural Language Processing is a complex field",
             "Natural Language Processing is used in machine learning"]

vocabulary = tokenize_sentences(sentences)
print(bagofwords("Machine learning is great", vocabulary))

from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(analyzer="word", tokenizer=None, preprocessor=None, stop_words=None, max_features=5000)

train_data_features = vectorizer.fit_transform(sentences)
print(train_data_features)
c = vectorizer.transform(["Machine learning is great", "Natural Language Processing is a complex field",
                      "Natural Language Processing is used in machine learning"]).toarray()
print(c)