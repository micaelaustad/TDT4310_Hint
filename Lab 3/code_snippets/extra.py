import pandas as pd
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


texts = ["I like fish and dogs",
         "Fish is not better than dogs or ducks", "The moon landing is fake"]


def text_cleaning(text):
    text = text.lower()
    text = " ".join([word for word in word_tokenize(text)
                     if not word in stopwords])
    text = re.sub(r"[!?:;\-,.<>]", r"", text)
    text = " ".join([stemmer.stem(word) for word in text.split()])
    return text


# apply functions to dataset
texts = list(map(text_cleaning, texts))

# Vectorize :slurp:
vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(texts)
feature_names = vectorizer.get_feature_names()
dense = vectors.todense()
denselist = dense.tolist()
df = pd.DataFrame(denselist, columns=feature_names)
print(df.head(15))
