# NER name entity recognition vid 8
import spacy
nlp = spacy.load('en_core_web_sm')

def show_ent(doc):
    for ent in doc.ents:
        print(f"{ent.text} - {ent.start}:{ent.end} - {ent.start_char}:{ent.end_char} - {ent.label_} - {str(spacy.explain(ent.label_))}")
    else:
        print("NO entity")
doc = nlp(u'paras is great and chandler bing too')
show_ent(doc)

from spacy.tokens import Span
new_ent = Span(doc,0,1,label = doc.vocab.strings[u'PERSON'])
doc.ents = list(doc.ents)+[new_ent]
show_ent(doc)
print()
print('matcher')

doc1 = nlp(u'Our company plans to introduce a new vacuumcleaner. '
          u'If successful, the vacuum-cleaner will be our first product.')

from spacy.matcher import PhraseMatcher
matcher = PhraseMatcher(nlp.vocab)

phrase_words = ['vacuumcleaner','vacuum-cleaner']
ph_pattern = [nlp(text) for text in phrase_words]
matcher.add('om',None,*ph_pattern)
matches = matcher(doc1)
print(matches)
print(show_ent(doc1))

prodd = doc1.vocab.strings[u"PRODUCT"]
new_ent1 = [Span(doc1,match[1],match[2],label = prodd) for match in matches]
doc1.ents = list(doc1.ents)+new_ent1
print(show_ent(doc1))
print()
print('counting entities')
doc2 = nlp(u'Originally priced at $29.50\n the sweater was marked down to five dollars.')
show_ent(doc2)

