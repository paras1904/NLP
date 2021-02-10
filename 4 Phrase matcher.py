import spacy
nlp = spacy.load('en_core_web_sm')
from spacy.matcher import PhraseMatcher
matcher = PhraseMatcher(nlp.vocab,attr = 'LOWER')
terms = ['iphone 11','iphone XS','Google Pixel']
pattern = [nlp(text) for text in terms]
matcher.add("terminologylist",None,*pattern)
texts = nlp(u"glowing review overall and some really interesting Google Pixel 3 and iphone XS")
matche = matcher(texts)
print(matche)