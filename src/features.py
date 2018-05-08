#!/usr/local/bin/python
# coding: utf-8
import os, sys
# for Arabic encoding purposes
reload(sys)
sys.setdefaultencoding('utf-8')
from setup import load_datset
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from counts_transformer import counts_vectorizer
#from sklearn.feature_extraction import DictVectorizer

class features:
    def __init__(self, train="../data/xtrain.txt"):
        print ('⎛i⎞⎛n⎞⎛i⎞⎛t⎞⎛i⎞⎛a⎞⎛ℓ⎞⎛i⎞⎛z⎞⎛i⎞⎛n⎞⎛g⎞  ...')
        self.xtrain = load_datset("../data/sample.txt")
        self.tfidf_vectorizer = TfidfVectorizer(analyzer="word", ngram_range=(1, 3))
        self.documents = [doc.text for doc in self.xtrain]
        self.tfidf_vectorizer.fit_transform(self.documents)

        self.lexicons = ['../data/lexicons/act_adverbs_wik.txt', '../data/lexicons/assertives_hooper1975.txt',
                    '../data/lexicons/comparative_forms_wik.txt', '../data/lexicons/firstPers_liwc.txt',
                    '../data/lexicons/hear_liwc.txt', '../data/lexicons/hedges_hyland2005.txt',
                    '../data/lexicons/manner_adverbs_wik.txt', '../data/lexicons/modal_adverbs_wik.txt',
                    '../data/lexicons/money_liwc.txt', '../data/lexicons/negations_liwc.txt',
                    '../data/lexicons/number_liwc.txt', '../data/lexicons/secPers_liwc.txt',
                    '../data/lexicons/see_liwc.txt', '../data/lexicons/sexual_liwc.txt',
                    '../data/lexicons/strong_subj_wilson.txt', '../data/lexicons/superlative_forms_wik.txt',
                    '../data/lexicons/swear_liwc.txt', '../data/lexicons/weak_subj_wilson.txt']

    def extract_baseline_feature(self,ds):
        for doc in ds:
            doc.baseline_feature = self.tfidf_vectorizer.transform([doc.text])
        return self.tfidf_vectorizer


    def extract_from_lexicon1(self,ds,lexicon):
        vectorizer = CountVectorizer(analyzer='word', vocabulary=lexicon)
        for doc in ds:
            doc.feature = vectorizer.transform([doc.text])
        feature_names = vectorizer.get_feature_names()
        print(feature_names)
        return vectorizer

    def extract_from_lexicon(self,ds,lexicon_file):
        vectorizer = counts_vectorizer(lexicon_file)
        vectorizer.transform([doc.text for doc in ds])
        return vectorizer

    def extract_lexical(self,ds):
        vectorizer = counts_vectorizer(self.lexicons)
        vectorizer.transform([doc.text for doc in ds])
        return vectorizer

