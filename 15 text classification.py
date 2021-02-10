import pandas as pd
data = pd.read_csv("spam.csv", encoding = "latin-1")
data = data[['v1', 'v2']]
data = data.rename(columns = {'v1': 'label', 'v2': 'text'})

def review_messages(msg):
    msg = msg.lower()
    return msg

from nltk import stem
from nltk.corpus import stopwords
stemmer = stem.SnowballStemmer('english')
stopwords = set(stopwords.words('english'))

def alternative_review_messages(msg):
    msg = msg.lower()
    msg = [word for word in msg.split() if word not in stopwords]
    msg = " ".join([stemmer.stem(word) for word in msg])
    return msg
data['text'] = data['text'].apply(alternative_review_messages)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size = 0.1, random_state = 1)
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train)
from sklearn.linear_model import LogisticRegression
svm = LogisticRegression()
svm.fit(X_train, y_train)

from sklearn.metrics import confusion_matrix
X_test = vectorizer.transform(X_test)
y_pred = svm.predict(X_test)
# print(confusion_matrix(y_test, y_pred))

def pred(msg):
    msg = vectorizer.transform([msg])
    prediction = svm.predict(msg)
    return prediction[0]

msg = input()
print(pred(msg))