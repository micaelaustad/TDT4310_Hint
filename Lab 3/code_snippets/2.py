from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
import csv
from nltk import stem
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
stemmer = stem.PorterStemmer()
content = []
stopwords = set(stopwords.words('english'))

with open('spam.csv', newline='', encoding='latin-1') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',')
    for row in spamreader:
        content.append(row)
texts = [row[1] for row in content]
labels = [str(row[0]) for row in content]


def encode_labels(label):
    return 1 if label == "ham" else 0


def text_cleaning(text):
    text = text.lower()
    text = " ".join([word for word in word_tokenize(text)
                     if not word in stopwords])
    text = re.sub(r"[!?:;\-,.<>]", r"", text)
    text = " ".join([stemmer.stem(word) for word in text.split()])
    return text


# apply functions to dataset
texts = list(map(text_cleaning, texts))
labels = list(map(encode_labels, labels))


X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.1, random_state=420)

# Vectorize :slurp:
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train)
# OMG I AM MACHINE LEARNING!
gnb = MultinomialNB()
gnb.fit(X_train, y_train)
# Evaluate performacnce
X_test = vectorizer.transform(X_test)
y_pred = gnb.predict(X_test)

for pred in gnb.predict_proba(X_test):
    print(
        f"Likelyhood for begin spam {pred[0]} | Likelyhood for begin ham {pred[1]}")

#Percision and recall
print(confusion_matrix(y_test, y_pred))
