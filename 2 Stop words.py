import spacy
nlp = spacy.load('en_core_web_sm')
print(nlp.Defaults.stop_words)
print(len(nlp.Defaults.stop_words))

#checking that word is stop
print(nlp.vocab['myself'].is_stop)
print(nlp.vocab['mystery'].is_stop)

#adding stop words
nlp.Defaults.stop_words.add('mystery')
nlp.vocab['mystery'].is_stop = True
print(nlp.vocab['mystery'].is_stop)

#removin stop words
nlp.Defaults.stop_words.remove('myself')
nlp.vocab['myself'].is_stop = False
print(nlp.vocab['myself'].is_stop)


import string
import nltk
nltk.download('punkt')
import re
from nltk.corpus import stopwords
from nltk import word_tokenize,sent_tokenize

text = 'The Quick brown 2fox jump over the sky!!!'
#make tokens
tokens = word_tokenize(text)
print(tokens)

#make lowercase
tokens = [w.lower() for w in tokens]
print(tokens)
#types of punctuatuion
re_punc = re.compile('[%s]'%re.escape(string.punctuation))
print(re_punc)

#removing punctuation
stripped = [re_punc.sub('',w) for w in tokens]
print(stripped)

#removeing non alpha words
words = [word for word in stripped if word.isalpha()]
print(words)

# filtering stopwords
stop_words = set(stopwords.words('english'))
words = [w for w in words if not w in stop_words]
print(words)