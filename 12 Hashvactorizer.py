from sklearn.feature_extraction.text import HashingVectorizer
text = ['thew quick brown fox jumped ove the lazy dog']
vectorizer = HashingVectorizer(n_features=20)
vectorizer.transform(text)
vector = vectorizer.transform(text)
print(vector.shape)
print(vector.toarray())