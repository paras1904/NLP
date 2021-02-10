import spacy
nlp = spacy.load('en_core_web_sm')

string = '"I\'m with you for entire life"'
print(string)

# token
doc = nlp(string)
for token in doc:
    print(token.text,end='|')

doc2 = nlp(u"ram is a good boy")
for t in doc2:
    print(t)

# without spcay it will read $ and 1000 as single string
doc3 = nlp(u"give me $1000")
for t in doc3:
    print(t)

#covert full stop also into another token or string
doc4 = nlp(u'ram is a goood boy.')
for i in doc4:
    print(i)
print(len(doc4))
print(len(doc4.vocab))
# accessing tokens
print(doc4[1])
print(doc4[0:2])
print(doc4[-3:])

# getting information of tokens:
doc5 = nlp(u"apple to build a hong kong factory $6 million")
for t in doc5:
    print(f"{t}|",end="")
print()
print()
for t in doc5.ents:
    print(f"{t.text}-{t.label_}-{spacy.explain(t.label_)}")
print()
print(len(doc5.ents))

# chunks or parts sentence can be divided
for t in doc5.noun_chunks:
    print(t)
    