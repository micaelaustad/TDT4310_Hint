import scikitplot as skplt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.ensemble import AdaBoostClassifier
from sklearn import linear_model
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import numpy as np
from datetime import datetime

from sklearn.feature_selection import chi2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import nltk
from nltk.corpus import stopwords
import re
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import pickle
import os
import sklearn
nltk.download('wordnet')
# Import Data

"""For full discloure, this code is heavily inspired by some projects I found online. The sources wont be cited since it solves the task... """


class YoutubeClassifier:
    def __init__(self, titles, descriptions, categories):
        """
        Starts a classifier with default values
        input: 
            titles: str[]
            descriptions: str[]
            categories: str[]
        tiltes, descriptions and categories for youtube videos
        return: 
            Returns a classifier object, with functions for every step of classification.
        """
        self.titles = titles
        self.descriptions = descriptions
        self.categories = categories
        self.labels, self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None, None
        self.TF_IDF_Title = None
        self.TF_IDF_Desc = None
        self.features = None
        self.labelEncoder = None
        self.NaiveBayes = MultinomialNB()
        self.features_title = None
        self.features_description = None

    def cleaning(self, titles=None, descriptions=None):

        titles = self.titles if titles == None else titles
        descriptions = self.descriptions if descriptions == None else descriptions
        titles = list(map(str, titles))
        descriptions = list(map(str, descriptions))
        """
        TODO
        FILL IN CODE HERE

        Task 1):
        Remove noise
        Task 2) 
        Stop word removal
        Task 3)
        Normalize the text


        Function needs to return two lists of strings."""
        return titles, descriptions

    def encodeLabels(self):
        """ Transforms text labels to numbers, to allow the algortihm to process them better"""
        self.labelEncoder = LabelEncoder()
        self.labelEncoder.fit(self.categories)
        self.categories = self.labelEncoder.transform(self.categories)

    def trainVectorizer(self):
        """ Generate the TF-IDF table """
        print(f"Generating TF-IDF Table | {datetime.now()}")
        self.TF_IDF_Title = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2',
                                            encoding='latin-1', ngram_range=(1, 2), stop_words='english')
        self.TF_IDF_Desc = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2',
                                           encoding='latin-1', ngram_range=(1, 2), stop_words='english')
        self.labels = data.category
        self.features_title = self.TF_IDF_Title.fit_transform(
            self.titles).toarray()
        self.features_description = self.TF_IDF_Desc.fit_transform(
            self.descriptions).toarray()
        print(f"Table Generated | {datetime.now()}")

    def inspectFeatures(self):
        """ Lookup the most informative features.
            This is a usual part of understanding the data you're working with. I.e data exploration
            In this case we look at which words in the TF-IDF is the most indicative of each category.
            i.e. has the highest average score for all documents of the category.
            Checks for the most informative features for both descriptions and titles.
        """
        print('Title Features Shape: ' + str(self.features_title.shape))
        print('Description Features Shape: ' +
              str(self.features_description.shape))
        # Best 5 keywords for each class using Title Feaures
        N = 5
        for current_class in list(self.labelEncoder.classes_):
            current_class_id = self.labelEncoder.transform([current_class])[0]
            features_chi2 = chi2(self.features_title,
                                 labels == current_class_id)
            indices = np.argsort(features_chi2[0])
            feature_names = np.array(
                self.TF_IDF_Title.get_feature_names())[indices]
            unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
            bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
            print("# '{}':".format(current_class))
            print("Most correlated unigrams:")
            print('-' * 30)
            print('. {}'.format('\n. '.join(unigrams[-N:])))
            print("Most correlated bigrams:")
            print('-' * 30)
            print('. {}'.format('\n. '.join(bigrams[-N:])))
            print("\n")

        # Best 5 keywords for each class using Description Features
        N = 5
        for current_class in list(self.labelEncoder.classes_):
            current_class_id = self.labelEncoder.transform([current_class])[0]
            features_chi2 = chi2(self.features_description,
                                 labels == current_class_id)
            indices = np.argsort(features_chi2[0])
            feature_names = np.array(
                self.TF_IDF_Desc.get_feature_names())[indices]
            unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
            bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
            print("# '{}':".format(current_class))
            print("Most correlated unigrams:")
            print('-' * 30)
            print('. {}'.format('\n. '.join(unigrams[-N:])))
            print("Most correlated bigrams:")
            print('-' * 30)
            print('. {}'.format('\n. '.join(bigrams[-N:])))
            print("\n")

    def vectorizeData(self, titles=None, descriptions=None):
        """ Uses the generated TF-IDF table to assign values to all the training data, so it becomes vectors"""
        print(f"Vectorizing data | {datetime.now()}")
        titles = self.titles if titles == None else titles
        descriptions = self.descriptions if descriptions == None else descriptions
        title_features = self.TF_IDF_Title.transform(
            titles).toarray()
        desc_features = self.TF_IDF_Desc.transform(
            descriptions).toarray()
        print(f" Vectorization completed | {datetime.now()}")
        return np.concatenate(
            [title_features, desc_features], axis=1)

    def splitDataset(self, features):
        """ Splits the features and the categories into a balanced dataset"""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            features, self.categories, random_state=42)

    def train_classifier(self, classifier):
        """ Boilerplate code for training classifiers in sklearn"""
        print(f"Training {classifier} | {datetime.now()}")
        self.classifier = classifier.fit(self.X_train, self.y_train)
        print(f"Classifier trained {classifier} | {datetime.now()}")

    def plot_classifer(self, classifier):
        """ boilerplate code for testing the accuracy of classifiers in sklearn"""
        test_features = self.X_test
        y_pred = classifier.predict(test_features)
        y_probas = classifier.predict_proba(test_features)

        print(metrics.classification_report(self.y_test, y_pred,
                                            target_names=list(self.labelEncoder.classes_)))

        conf_mat = confusion_matrix(self.y_test, y_pred)
        fig, ax = plt.subplots(figsize=(10, 10))
        sns.heatmap(conf_mat, annot=True, fmt='d', xticklabels=list(
            self.labelEncoder.classes_), yticklabels=list(self.labelEncoder.classes_))
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.title('Confusion Matrix')
        plt.show()

        skplt.metrics.plot_precision_recall_curve(self.y_test, y_probas)
        plt.title('Precision-Recall Curve')
        plt.show()

    def main(self):
        self.titles, self.descriptions = self.cleaning()
        self.encodeLabels()
        self.trainVectorizer()
        # self.inspectFeatures()
        feat = self.vectorizeData()
        self.splitDataset(feat)
        self.train_classifier(self.NaiveBayes)
        self.plot_classifer(self.NaiveBayes)

    def predict(self, titles, descriptions, classifier=None):
        """ Takes inn fresh data, that you have collected, and you can test and play around with the classifier """
        """ Enjoy, test out with you favorite channels, and generate some insights :) """
        classifier = self.NaiveBayes if classifier == None else classifier
        org_titles = titles
        titles, descriptions = self.cleaning(titles, descriptions)
        feat = self.vectorizeData(titles, descriptions)
        y_pred = self.labelEncoder.inverse_transform(classifier.predict(feat))
        y_probas = classifier.predict_proba(feat)
        for x in range(len(titles)):
            print(f"The title {org_titles[x]} ||| is of the class {y_pred[x]}")


if __name__ == "__main__":
    """ TODO READ HERE: template code for running and using the model """
    data = pd.read_csv('bigFile/data/GB.csv')
    data2 = pd.read_csv('bigFile/data/US.csv')

    # NB you can slice the lists to be smaller for faster training or if you have less RAM available than the algorithm uses (i.e. mem errors).
    titles = data['title'].tolist()[:-100] + data2['title'].tolist()[:-100]
    descriptions = data['description'].tolist(
    )[:-100] + data2['description'].tolist()[:-100]
    categories = data['category'].tolist()[:-100] + \
        data2['category'].tolist()[:-100]

    # Running the model and classifying with it.
    ytC = YoutubeClassifier(titles, descriptions, categories)
    ytC.main()

    # Predicting unseen data, Extra TODO, you can test with your own data here :)
    test_titles = data['title'].tolist()[-100:] + \
        data2['title'].tolist()[:-100]
    test_descriptions = data['description'].tolist(
    )[-100:] + data2['description'].tolist()[:-100]
    ytC.predict(test_titles, test_descriptions)
