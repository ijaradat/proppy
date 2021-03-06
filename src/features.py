#!/usr/local/bin/python
# coding: utf-8
import os, sys
# for Arabic encoding purposes
reload(sys)
sys.setdefaultencoding('utf-8')
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from counts_transformer import counts_vectorizer
from readability import LexicalStyle_vectorizer, Readability_vectorizer
from feature_extraction.nela_transformer import nela_vectorizer
from feature_extraction.morality_transformer import morality_vectorizer
from feature_extraction.bias_transformer import bias_vectorizer

class features:
    def __init__(self, train):
        print ('⎛i⎞⎛n⎞⎛i⎞⎛t⎞⎛i⎞⎛a⎞⎛ℓ⎞⎛i⎞⎛z⎞⎛i⎞⎛n⎞⎛g⎞  ...')
        self.xtrain = train
        self.tfidf_vectorizer = TfidfVectorizer(analyzer="word", ngram_range=(1, 3)) # initializing tf-idf vectorizer class with the training ds to learn the vocab
        self.documents = [doc.text for doc in self.xtrain]
        self.tfidf_vectorizer.fit_transform(self.documents)
        self.tfidf_char_vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3,3), use_idf=False)
        self.tfidf_char_vectorizer.fit_transform(self.documents)
        self.lexicons = ['../data/lexicons/act_adverbs_wik.txt', '../data/lexicons/assertives_hooper1975.txt','../data/lexicons/comparative_forms_wik.txt',
                         '../data/lexicons/firstPers_liwc.txt','../data/lexicons/hear_liwc.txt', '../data/lexicons/hedges_hyland2005.txt',
                         '../data/lexicons/manner_adverbs_wik.txt', '../data/lexicons/modal_adverbs_wik.txt', '../data/lexicons/money_liwc.txt',
                         '../data/lexicons/negations_liwc.txt','../data/lexicons/number_liwc.txt', '../data/lexicons/secPers_liwc.txt',
                         '../data/lexicons/see_liwc.txt', '../data/lexicons/sexual_liwc.txt','../data/lexicons/strong_subj_wilson.txt',
                         '../data/lexicons/superlative_forms_wik.txt','../data/lexicons/swear_liwc.txt', '../data/lexicons/weak_subj_wilson.txt']

    def extract_baseline_feature(self,ds):
        #for doc in ds:
        #    doc.baseline_feature = self.tfidf_vectorizer.transform([doc.text])
        return self.tfidf_vectorizer

    def extract_char_n_grams(self,ds):
        return self.tfidf_char_vectorizer

    def extract_from_lexicon1(self,ds,lexicon): # function not used (use if you want each term in each lexicon to be a single feature)
        vectorizer = CountVectorizer(analyzer='word', vocabulary=lexicon)
        #for doc in ds:
        #    doc.feature = vectorizer.transform([doc.text])
        feature_names = vectorizer.get_feature_names()
        print(feature_names)
        return vectorizer

    def extract_from_lexicon(self,ds,lexicon_file): # function not used anymore, used when eacc lexicon count was a separate feature in the pipeline
        vectorizer = counts_vectorizer(lexicon_file)
        #vectorizer.transform([doc.text for doc in ds])
        return vectorizer

    def extract_lexical(self,ds): # function to collect lexical features at once via the counts_tranformer class
        vectorizer = counts_vectorizer(self.lexicons)
        #vectorizer.transform([doc.text for doc in ds])
        return vectorizer

    def extract_lexicalstyle_features(self, ds):
        vectorizer = LexicalStyle_vectorizer()
        #vectorizer.transform([doc.text for doc in ds])
        return vectorizer

    def extract_readability_features(self, ds):
        vectorizer = Readability_vectorizer()
        return vectorizer

    def extract_nela_features(self,ds):
        vectorizer = nela_vectorizer()
        return vectorizer

    def extract_morality_features(self,ds):
        vectorizer = morality_vectorizer()
        return vectorizer

    def extract_bias_features(self,ds):
        vecrotizer = bias_vectorizer()
        return vecrotizer