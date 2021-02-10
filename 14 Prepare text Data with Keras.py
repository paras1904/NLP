# vid 15
from keras.preprocessing.text import text_to_word_sequence,one_hot,hashing_trick ,Tokenizer
text = 'the quick brown fox jump over the lazy dog'
result = text_to_word_sequence(text,lower=True)
vocab_size = len(set(result))
print(result)
print(vocab_size)
print()
result2 = one_hot(text,round(vocab_size*1.3))
print(result2)
print()
result3 = hashing_trick(text,round(vocab_size*1.3),hash_function='md5')
print(result3)
print()
doc2 = ['paras','is a good','guy with humour','paras']
tok = Tokenizer()
tok.fit_on_texts(doc2)
print(tok.word_counts)
print(tok.document_count)
print(tok.word_index)
print(tok.word_docs)
one_vec = tok.texts_to_matrix(doc2,mode='count')
print(one_vec)
