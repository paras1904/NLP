# vid 8
import spacy
nlp = spacy.load('en_core_web_sm')

doc = nlp(u'Autonomous cars shift insurance liability towards manufacture')

for chunk in doc.noun_chunks:
    print(f"{chunk.text} - {chunk.root.text} - {chunk.root.dep_} - {chunk.root.head.text}")
print('length of noun chunks')
print(len(list(doc.noun_chunks)))

print('visualisation')
from spacy import displacy
