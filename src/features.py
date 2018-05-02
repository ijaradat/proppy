#!/usr/local/bin/python
# coding: utf-8
import os, sys
# for Arabic encoding purposes
reload(sys)
sys.setdefaultencoding('utf-8')
from setup import load_datset
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from counts_transformer import counts_vectorizer
from readability import LexicalStyle_vectorizer
#from sklearn.feature_extraction import DictVectorizer

import functional_words

class features:

    def __init__(self, train="../data/xtrain.txt"):
        print ('⎛i⎞⎛n⎞⎛i⎞⎛t⎞⎛i⎞⎛a⎞⎛ℓ⎞⎛i⎞⎛z⎞⎛i⎞⎛n⎞⎛g⎞  ...')
        xtrain = load_datset(train)
        # xtrain = load_datset('../data/xtrain.3.txt')
        self.tfidf_vectorizer = TfidfVectorizer(analyzer="word", ngram_range=(1, 3))
        self.documents = [doc.text for doc in xtrain]
        self.tfidf_vectorizer.fit_transform(self.documents)
        # action_count_vectorizer = CountVectorizer(analyzer='word', vocabulary=load_lexicon('../data/lexicons/act_adverbs_wik.txt'))
        # action_count_vectorizer.fit_transform(documents)

    def extract_baseline_feature(self, ds):
        for doc in ds:
            doc.baseline_feature = self.tfidf_vectorizer.transform([doc.text])
        return self.tfidf_vectorizer


    # def extract_functional_ngrams(self, ds):
    #     global funct_vectorizer
    #     for doc in ds:
    #         # text = functional_words.get_functional_words(doc.text)
    #         doc.baseline_feature = funct_vectorizer.transform([doc.text])
    #     return funct_vectorizer


    def extract_from_lexicon1(self, ds,lexicon):
        vectorizer = CountVectorizer(analyzer='word', vocabulary=lexicon)
        for doc in ds:
            doc.feature = vectorizer.transform([doc.text])

        feature_names = vectorizer.get_feature_names()
        print(feature_names)
        return vectorizer


    def extract_from_lexicon(self, ds,lexicon_file):
        vectorizer = counts_vectorizer(lexicon_file)
        vectorizer.transform([doc.text for doc in ds])
        return vectorizer

    def extract_readability_features(self, ds):
        vectorizer = LexicalStyle_vectorizer()
        vectorizer.transform([doc.text for doc in ds])
        return vectorizer




        # funct_vectorizer = TfidfVectorizer(
        #                         analyzer="word",
        #                         ngram_range=(1, 3),
        #                         preprocessor=functional_words.get_functional_words(""))
        #     # TODO perhaps I don't need to load this twice
        #     documents = [doc.text for doc in xtrain]
        #     funct_vectorizer.fit_transform(documents)
        #
        #


    # funct_vectorizer = TfidfVectorizer(
    #                     analyzer="word",
    #                     ngram_range=(1, 3),
    #                     preprocessor=functional_words.get_functional_words(""))
    # # TODO perhaps I don't need to load this twice
    # documents = [doc.text for doc in xtrain]
    # funct_vectorizer.fit_transform(documents)