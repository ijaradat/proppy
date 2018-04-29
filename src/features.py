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



def extract_baseline_feature(ds):
    global tfidf_vectorizer
    for doc in ds:
        doc.baseline_feature = tfidf_vectorizer.transform([doc.text])
    return tfidf_vectorizer



def extract_from_lexicon1(ds,lexicon):
    vectorizer = CountVectorizer(analyzer='word', vocabulary=lexicon)
    for doc in ds:
        doc.feature = vectorizer.transform([doc.text])

    feature_names = vectorizer.get_feature_names()
    print(feature_names)
    return vectorizer

def extract_from_lexicon(ds,lexicon_file):
    vectorizer = counts_vectorizer(lexicon_file)
    vectorizer.transform([doc.text for doc in ds])
    return vectorizer




print ('⎛i⎞⎛n⎞⎛i⎞⎛t⎞⎛i⎞⎛a⎞⎛ℓ⎞⎛i⎞⎛z⎞⎛i⎞⎛n⎞⎛g⎞  ...')
xtrain = load_datset('../data/xtrain.txt')
tfidf_vectorizer =  TfidfVectorizer(analyzer="word",ngram_range=(1, 3))
documents = [doc.text for doc in xtrain]
tfidf_vectorizer.fit_transform(documents)
#action_count_vectorizer = CountVectorizer(analyzer='word', vocabulary=load_lexicon('../data/lexicons/act_adverbs_wik.txt'))
#action_count_vectorizer.fit_transform(documents)
