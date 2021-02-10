import stanza
#stanza.download('hindi')
nlp = stanza.Pipeline('hi')
doc = nlp('हम उसे नहीं जानते हैं।')
#print(doc)
nlp = stanza.Pipeline(lang='hi',processors='tokenize')
doc1 = nlp('हम उसे नहीं जानते हैं।')
for i,sentence in enumerate(doc1.sentences):
    print(f'================{i+1} tokens ====================')
    print(*[f'id: {token.id}\ttext: {token.text}' for token in sentence.tokens],sep='\n')
print([sentence.text for sentence in doc.sentences])