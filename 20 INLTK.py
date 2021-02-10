from inltk.inltk import setup
setup('hi')


from inltk.inltk import tokenize
hindi_text = 'वह अपने कमरे में रो रही थी।'
tokenize(hindi_text,'hi')

from inltk.inltk import get_embedding_vectors
vectors = get_embedding_vectors(hindi_text,'hi')

from inltk.inltk import predict_next_words
predict_next_words(hindi_text,2,'hi')

from inltk.inltk import identify_language
identify_language('hi')

from inltk.inltk import get_sentence_encoding
encoding = get_sentence_encoding(hindi_text,'hi')

from inltk.inltk import get_sentence_similarity
get_sentence_similarity('वह अपने कमरे में रो रही थी।','वह अपने कमरे में रो रही थी।','hi')

from inltk.inltk import get_similar_sentences
output = get_similar_sentences(hindi_text, 1, 'hi')
print(output)