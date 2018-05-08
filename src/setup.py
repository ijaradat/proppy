#!/usr/local/bin/python
# coding: utf-8
import os, sys
# for Arabic encoding purposes
reload(sys)
sys.setdefaultencoding('utf-8')
import codecs
#import tensorflow as tf
#import numpy as np
import optparse
import pickle
from collections import OrderedDict
from features import *
from document import document
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.ensemble import ExtraTreesClassifier
import counts_transformer
optparser = optparse.OptionParser()

optparser.add_option(
    "-T", "--xtrain", default="../data/sample.txt", # "../data/xtrain.txt"
    help="xtrain set path"
)
optparser.add_option(
    "-D", "--xdev", default="../data/sample.txt",  #"../data/xdev.txt"
    help="xdev set path"
)
optparser.add_option(
    "-t", "--test", default="../data/sample.txt",  #"../data/test.txtconverted.txt"
    help="test set path"
)


opts = optparser.parse_args()[0]


def parse_parameters():
    """
        parameter list
        ---------------------
        --xtrain -T          --dev -D            --test -t


    Parse the input parameters
    :return: <dictionary>  a dictionary of parameters
    """


    param = OrderedDict()

    param['xdev'] = opts.xdev
    param['test'] = opts.test
    param['xtrain'] = opts.xtrain

    print ("PARAMETER LIST:_______________________________________________________________________")
    print (param)
    # if not os.path.isfile(param['train']):
    #     raise Exception("Train dataset file does not exist.".format(param['train']))
    # if not os.path.isfile(param['train']):
    #     raise Exception("Dev dataset file does not exist.".format(param['dev']))
    # if not os.path.isfile(param['train']):
    #     raise Exception("Test dataset file does not exist.".format(param['test']))
    # if not os.path.isfile(param['xtrain']):
    #     raise Exception("Test dataset file does not exist.".format(param['xtest']))


    return param


def load_datset(dataset_file):
    print ('loading dataset: '+dataset_file+ ' ...')
    dataset =[]
    with codecs.open(dataset_file,'r') as f:
        i=0
        for line in f:
            fields = line.split('\t')
            article = document(fields[1],fields[0],i)
            dataset.append(article)
            i+=1
        f.close()
    print ('done !')
    return dataset

def read_datsets(param):
    print ('reading datasets ...')
    xtrain = load_datset(param['xtrain'])
    xdev = load_datset(param['xdev'])
    test = load_datset(param['test'])
    print ('done reading data !')
    return xtrain,xdev,test


def extract_features(ds, feats):

    print('constructing features pipeline ...')
    tfidf = feats.extract_baseline_feature(ds)  # each one of these is a sklearn object that has a transform method (each one is a transformer)
    lexical = feats.extract_lexical(ds)
    # action_adverbs = extract_from_lexicon(ds,'../data/lexicons/act_adverbs_wik.txt')
    # assertives= extract_from_lexicon(ds,'../data/lexicons/assertives_hooper1975.txt')
    # comparatives = extract_from_lexicon(ds,'../data/lexicons/comparative_forms_wik.txt')
    # first_person = extract_from_lexicon(ds,'../data/lexicons/firstPers_liwc.txt')
    # hear = extract_from_lexicon(ds,'../data/lexicons/hear_liwc.txt')
    # hedges = extract_from_lexicon(ds,'../data/lexicons/hedges_hyland2005.txt')
    # manner_adverbs = extract_from_lexicon(ds,'../data/lexicons/manner_adverbs_wik.txt')
    # modal_adverbs = extract_from_lexicon(ds,'../data/lexicons/modal_adverbs_wik.txt')
    # money = extract_from_lexicon(ds,'../data/lexicons/money_liwc.txt')
    # negations = extract_from_lexicon(ds,'../data/lexicons/negations_liwc.txt')
    # number = extract_from_lexicon(ds,'../data/lexicons/number_liwc.txt')
    # second_person = extract_from_lexicon(ds,'../data/lexicons/secPers_liwc.txt')
    # see = extract_from_lexicon(ds,'../data/lexicons/see_liwc.txt')
    # sexual = extract_from_lexicon(ds,'../data/lexicons/sexual_liwc.txt')
    # strong_subjectives = extract_from_lexicon(ds,'../data/lexicons/strong_subj_wilson.txt')
    # superlatives = extract_from_lexicon(ds,'../data/lexicons/superlative_forms_wik.txt')
    # swear = extract_from_lexicon(ds,'../data/lexicons/swear_liwc.txt')
    # weak_subjectives= extract_from_lexicon(ds,'../data/lexicons/weak_subj_wilson.txt')



    features_pipeline =  FeatureUnion([ ('tf-idf',tfidf),
                                        ('lexical', lexical)
                                        # ('action_adverbs', action_adverbs),
                                        # ('assertives', assertives),
                                        # ('comparatives',comparatives),
                                        # ('first_person',first_person),
                                        # ('hear',hear),
                                        # ('hedges',hedges),
                                        # ('manner_adverbs',manner_adverbs),
                                        # ('modal_adverbs',modal_adverbs),
                                        # ('money',money),
                                        # ('negations',negations),
                                        # ('number',number),
                                        # ('second_person',second_person),
                                        # ('see',see),
                                        # ('sexual',sexual),
                                        # ('strong_subjectives',strong_subjectives),
                                        # ('superlatives',superlatives),
                                        # ('swear',swear),
                                        # ('weak_subjectives',weak_subjectives)
                                        ])  # Pipeline([('vectorizer', vec), ('vectorizer2', vec),....])
    print ('features pipeline ready !')
    return  features_pipeline

def select_features(train, feats):
    print ('████████████████  FEATURE SELECTION  ████████████████')
    vectorizer = counts_vectorizer(feats.lexicons)
    dataset_data = vectorizer.transform([doc.text for doc in train])
    dataset_target = [doc.gold_label for doc in train]
    model = ExtraTreesClassifier()
    model.fit(dataset_data, dataset_target)
    # display the relative importance of each attribute
    print (vectorizer.feature_names)
    print(model.feature_importances_)
    print ()

def train_model(train, feats):
    print ('████████████████  𝕋 ℝ 𝔸 𝕀 ℕ 𝕀 ℕ 𝔾  ████████████████')
    features_pipeline = extract_features(train, feats)
    # dump it (to speed up exp.)
    #pickle.dump(features_pipeline, open("train_features.pickle", "wb"))
    model = LogisticRegression(penalty='l2')
    X = features_pipeline.transform([doc.text for doc in train])
    Y = [doc.gold_label for doc in train]
    print ('fitting the model according to given data ...')
    model.fit(X, Y)
    #pickle the model
    joblib.dump(model, 'basic_features_model.pkl')
    print ('model pickled at : basic_features_model.pkl ')


def test_model(test, feats):
    print ('████████████████   𝕋 𝔼 𝕊 𝕋 𝕀 ℕ 𝔾   ████████████████')
    features_pipeline= extract_features(test, feats)
    #pickle.dump(features_pipeline, open("test_features.pickle", "wb"))
    print('loading pickled model from : basic_features_model.pkl ')
    model = joblib.load('basic_features_model.pkl') #load the pickled model
    X = features_pipeline.transform([doc.text for doc in test])
    print ('predicting Y for each given X in test ...')
    Y_ = model.predict(X)

    for i, doc in enumerate(test):
        doc.prediction  = Y_[i]
    return test


def evaluate_model(ds):
    # F1 score
    y_true = [doc.gold_label for doc in ds]
    y_pred = [doc.prediction for doc in ds]
    score = f1_score(y_true, y_pred, average='macro' ,labels=['1','2','3','4'])
    print ("F1 scores:")
    print (score)


def main ():

    param = parse_parameters()

    xtrain,xdev,test = read_datsets(param)
    feats = features(param['xtrain'])
    select_features(xtrain, feats)

    train_model(xtrain, feats)

    tested_dev = test_model(xdev, feats)
    tested_test = test_model(test, feats)

    print ('evaluating the model using dev ds ...')
    evaluate_model(tested_dev)
    print ('evaluating the model using test ds ...')
    evaluate_model(tested_test)

if __name__ == '__main__':
    main()
