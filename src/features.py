#!/usr/local/bin/python
# coding: utf-8
import os, sys
# for Arabic encoding purposes
reload(sys)
sys.setdefaultencoding('utf-8')
from setup import load_datset
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_extraction import DictVectorizer
from nltk.tokenize import word_tokenize
import codecs
from numpy import array





def extract_baseline_feature(ds):
    global tfidf_vectorizer
    for doc in ds:
        doc.baseline_feature = tfidf_vectorizer.transform([doc.text])
    return tfidf_vectorizer


def extract_from_lexicon1(ds,lexicon,feature_name):
    print('Extracting '+feature_name + '  feature ...')
    feature = []
    for doc in ds:
        tokens = word_tokenize(doc.text.lower())
        count = 0
        for vocab in lexicon:
            count += tokens.count(vocab)  # count the number of all vocab in the whole document (document = a list of tokens)
        feature.append(count)
    feature=  array(feature)
    dict_vectorizer = DictVectorizer()
    dict_vectorizer.fit(feature)
    print('done !')
    return feature

def extract_from_lexicon(ds,lexicon):
    vectorizer = CountVectorizer(analyzer='word', vocabulary=lexicon)
    for doc in ds:
        doc.feature = vectorizer.transform([doc.text])

    feature_names = vectorizer.get_feature_names()
    print(feature_names)
    return vectorizer


def load_lexicon(file):
    print('Loading lexicon from ' + file + ' ...')
    lexicon = []
    with codecs.open(file, 'r') as f:
        for line in f:
            line = line.strip()
            lexicon.append(line)
        f.close()
        print('done!')
        return lexicon


print ('⎛i⎞⎛n⎞⎛i⎞⎛t⎞⎛i⎞⎛a⎞⎛ℓ⎞⎛i⎞⎛z⎞⎛i⎞⎛n⎞⎛g⎞  ...')
xtrain = load_datset('../data/xtrain.txt')
tfidf_vectorizer =  TfidfVectorizer(analyzer="word",ngram_range=(1, 3))
documents = [doc.text for doc in xtrain]
tfidf_vectorizer.fit_transform(documents)
#action_count_vectorizer = CountVectorizer(analyzer='word', vocabulary=load_lexicon('../data/lexicons/act_adverbs_wik.txt'))
#action_count_vectorizer.fit_transform(documents)
