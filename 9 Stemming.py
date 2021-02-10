# vid 10
import spacy
nlp = spacy.load('en_core_web_sm')
# breaking prefix suffix infix exception and printing them
doc = nlp(u'@(paras)')
for tok in doc:
    print(tok)
# porter stemmer
from nltk.stem.porter import PorterStemmer
port_stemmer = PorterStemmer()
words = ['run','runner','running','ran','runs','easily','fairly']
for w in words:
    print(f'{w} - {port_stemmer.stem(w)}')
#Snowballstemer
print('snowball stemmer')
print()
from nltk.stem.snowball import SnowballStemmer
snowball_stemmer = SnowballStemmer(language='english')
words = ['run','runner','running','ran','runs','easily','fairly']
for w in words:
    print(f'{w} - {snowball_stemmer.stem(w)}')
