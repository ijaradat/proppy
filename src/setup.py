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
    tfidf = feats.extract_baseline_feature(ds)  # each one of these is an sklearn object that has a transform method (each one is a transformer)
    lexical = feats.extract_lexical(ds)
    readability_features = feats.extract_readability_features(ds)


    features_pipeline =  FeatureUnion([ ('tf-idf',tfidf),
                                        ('lexical', lexical),
                                        ('readability', readability_features)
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
    print (vectorizer.feature_names)
    print(model.feature_importances_)  # display the relative importance of each attribute
    print ()

def train_model(train, feats):
    print ('████████████████  𝕋 ℝ 𝔸 𝕀 ℕ 𝕀 ℕ 𝔾  ████████████████')
    features_pipeline = extract_features(train, feats) # call the methods that extract features to initialize transformers
    # ( this method only initializes transformers, pipeline.transform below when called, it calls all transform methods of all tranformers in the pipeline)
    #pickle.dump(features_pipeline, open("train_features.pickle", "wb"))  # dump it (to speed up exp.)
    model = LogisticRegression(penalty='l2') # creating an object from the max entropy with L2 regulariation
    X = features_pipeline.transform([doc.text for doc in train]) # calling transform method of each transformer in the features pipeline to transform data into vectors of features
    Y = [doc.gold_label for doc in train]
    print ('fitting the model according to given data ...')
    model.fit(X, Y)

    joblib.dump(model, 'basic_features_model.pkl') #pickle the model
    print ('model pickled at : basic_features_model.pkl ')


def test_model(test, feats):
    print ('████████████████   𝕋 𝔼 𝕊 𝕋 𝕀 ℕ 𝔾   ████████████████')
    features_pipeline= extract_features(test, feats)  # call the methods that extract features to initialize transformers
    # ( this method only initializes transformers, pipeline.transform below when called, it calls all transform methods of all tranformers in the pipeline)
    #pickle.dump(features_pipeline, open("test_features.pickle", "wb"))
    print('loading pickled model from : basic_features_model.pkl ')
    model = joblib.load('basic_features_model.pkl') #load the pickled model
    X = features_pipeline.transform([doc.text for doc in test])  # calling transform method of each transformer in the features pipeline to transform data into vectors of features
    print ('predicting Y for each given X in test ...')
    Y_ = model.predict(X)  # predicting the labels in this ds via the trained model loaded in the variable 'model'

    for i, doc in enumerate(test):
        doc.prediction  = Y_[i]
    return test


def evaluate_model(ds):
    # F1 score
    y_true = [doc.gold_label for doc in ds] # getting all gold labels of the ds as one list
    y_pred = [doc.prediction for doc in ds] # getting all model predicted lebels as a list
    score = f1_score(y_true, y_pred, average='macro' ,labels=['1','2','3','4']) # calculating F1 score
    print ("F1 scores:")
    print (score)


def main ():

    param = parse_parameters() # get parameters from command

    xtrain,xdev,test = read_datsets(param) # loading datsets as lists of document objects
    feats = features(param['xtrain'])  # creating an object from the class features to initialize important global variables such as lexicons and training ds
    select_features(xtrain, feats)  # feature selection and importance

    train_model(xtrain, feats)  # training the model

    tested_dev = test_model(xdev, feats)  #testing the model with the dev ds
    tested_test = test_model(test, feats)  #testing the model with the test ds

    print ('evaluating the model using dev ds ...')
    evaluate_model(tested_dev)  # evaluating the model on the dev
    print ('evaluating the model using test ds ...')
    evaluate_model(tested_test)  #evaluating the model on the test

if __name__ == '__main__':
    main()
