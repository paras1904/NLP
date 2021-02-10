import spacy
nlp = spacy.load('en_core_web_sm')
doc = nlp(u'This is the first sentence. This is another sentence. This is the last sentence.')

for sent in doc.sents:
    print(sent)

print(doc[1])

doc_sents = [sent for sent in doc.sents]

print(doc_sents[1])

print(type(doc_sents[1]))

print(doc_sents[1].start, doc_sents[1].end)

doc2 = nlp(u'This is a sentence. This is a sentence. This is a sentence.')

for token in doc2:
    print(token.is_sent_start, ' '+token.text)

doc3 = nlp(u'"Management is doing things right; leadership is doing the right things." -Peter Drucker')

for sent in doc3.sents:
    print(sent)

# ADD A NEW RULE TO THE PIPELINE
def set_custom_boundaries(doc):
    for token in doc[:-1]:
        if token.text == ';':
            doc[token.i+1].is_sent_start = True
    return doc

nlp.add_pipe(set_custom_boundaries, before='parser')
print(nlp.pipe_names)
doc4 = nlp(u'"Management is doing things right; leadership is doing the right things." -Paras')
for sentt in doc4.sents:
    print(sentt)


mystring = u"This is a sentence. This is another.\n\nThis is a \nthird sentence."

doc = nlp(mystring)

for sent in doc.sents:
    print([token.text for token in sent])
from spacy.pipeline import SentenceSegmenter

def split_on_newlines(doc):
    start = 0
    seen_newline = False
    for word in doc:
        if seen_newline:
            yield doc[start:word.i]
            start = word.i
            seen_newline = False
        elif word.text.startswith('\n'): # handles multiple occurrences
            seen_newline = True
    yield doc[start:]

sbd = SentenceSegmenter(nlp.vocab, strategy=split_on_newlines)
nlp.add_pipe(sbd)

doc = nlp(mystring)
for sent in doc.sents:
    print([token.text for token in sent])
