# vid 12
# term frequency inverse document frequency
from sklearn.feature_extraction.text import CountVectorizer
text = ['the car is driven on the road','the truck is driven on the highway']
vectoriser =CountVectorizer()
vectoriser.fit(text)
print(vectoriser.vocabulary_)
new_vect = vectoriser.transform(text)
print(new_vect.toarray())

from sklearn.feature_extraction.text import TfidfVectorizer
vectoriser2 = TfidfVectorizer()
vectoriser2.fit(text)
print(vectoriser2.idf_)
print(vectoriser2.vocabulary_)
