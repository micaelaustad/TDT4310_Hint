
import numpy as np
import pandas as pd
import re
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, GRU, Dropout
from sklearn.model_selection import train_test_split


# https://towardsdatascience.com/understanding-lstm-and-its-quick-implementation-in-keras-for-sentiment-analysis-af410fd85b47

def genderize(gender):
    return 1 if gender == "male" else 0


def loadTweets():
    #Load + preprocessing
    data = pd.read_csv('data/crowdflower.csv')
    data['text'] = data['text'].apply(lambda x: x.lower())
    data['gender'] = np.array([genderize(gender)
                               for gender in data['gender']])
    return data


# most popular tokens in tweets
max_words = 1000


def sequences():
    # load data
    data = loadTweets()
    # It discards any tokens that are smaller than the 1k most popular tokens
    token = Tokenizer(num_words=max_words, split=' ')
    # Fit the tokenizer (same as TFIDF) - it is not TF-IDF, but serves the same purpose. Remember, functions from the same package are buddies, they play nice, less complexity.
    # TF-IDF from SKlearn + NLTK Classifier is an example of a combo that is a bit bad.
    token.fit_on_texts(data['text'].values)
    X = token.texts_to_sequences(data['text'].values)

    # DL requires equal length "strings", so we pad them with 0 values to make them equal length (to ensure each DL node has a value)
    X = pad_sequences(X, 100)
    Y = pd.get_dummies(data['gender']).values
    return X, Y, token


# data after preprocessing
X, Y, tokenizer = sequences()


# Embedding layer using the Embedding from Glove ^^ Search for it Online :D
def embedding(tokenizer):
    word_index = tokenizer.word_index
    embeddings_index = {}
    # Adding Glove, a Word embedding built for Twitter
    # Yuck old syntax...
    f = open('data/Glove/glove.twitter.27B.100d.txt')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    print('Found %s word vectors.' % len(embeddings_index))
    embedding_matrix = np.zeros((len(word_index) + 1, 100))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    # Input_length must = pad_sequences(,max_length)
    return Embedding(len(word_index) + 1,
                     100,
                     weights=[embedding_matrix],
                     input_length=100,
                     trainable=False)


def buildModel(embedding_layer):
    Gru_out = 32
    model = Sequential()
    model.add(embedding_layer)
    model.add(GRU(Gru_out))
    model.add(Dropout(0.2))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


model = buildModel(embedding(tokenizer))
print(model.summary())


def processing(model, X, Y):
    # Ah our old friend sklearn <3
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.1, random_state=34+35)
    # Yes it is an Ariana grande reference
    model.fit(X_train, Y_train, epochs=10,
              batch_size=32, verbose=1, callbacks=None,)
    # mm small batch sizes are nice ^^
    scores = model.evaluate(X_test, Y_test, verbose=1)
    print(scores)
    print("Accuracy: %.2f" % (scores[1]))


processing(model, X, Y)
