#!/usr/local/bin/python
import codecs
import os,sys
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
from sklearn.pipeline import Pipeline


optparser = optparse.OptionParser()

optparser.add_option(
    "-T", "--xtrain", default="../data/xtrain.txt",
    help="xtrain set path"
)
optparser.add_option(
    "-D", "--xdev", default="../data/xdev.txt",
    help="xdev set path"
)
optparser.add_option(
    "-t", "--test", default="../data/test.txtconverted.txt",
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
        for line in f:
            fields = line.split('\t')
            article = document(fields[1],fields[0])
            dataset.append(article)
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


def extract_features(ds):
    print('constructing features pipeline ...')
    tfidf_vec = extract_baseline_feature(ds) # returns a vector of tf-idf weighed doc-term matrix (vectorizer)
    features_pipeline = Pipeline([('tf-idf', tfidf_vec)])   #Pipeline([('vectorizer', vec), ('vectorizer2', vec),....])
    print ('features pipeline ready !')
    return  features_pipeline


def train_model(train):
    print ('Training phase: __________________________________________________________________________________________')
    features_pipeline = extract_features(train)
    # dump it (to speed up exp.)
    #pickle.dump(features_pipeline, open("train_features.pickle", "wb"))
    model = LogisticRegression(penalty='l2')
    X = features_pipeline.transform([doc.text for doc in train])
    Y = [doc.gold_label for doc in train]
    print ('fitting the model according to given data ...')
    model.fit(X, Y)
    #pickle the model
    joblib.dump(model, 'baseline_model.pkl')
    print ('model pickled at : baseline_model.pkl ')


def test_model(test):
    print ('Testing phase: __________________________________________________________________________________________')
    features_pipeline= extract_features(test)
    #pickle.dump(features_pipeline, open("test_features.pickle", "wb"))
    print('loading pickled model from : baseline_model.pkl ')
    model = joblib.load('baseline_model.pkl') #load the pickled model
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

    train_model(xtrain)

    tested_dev = test_model(xdev)
    tested_test = test_model(test)

    print ('evaluating the model using dev ds ...')
    evaluate_model(tested_dev)
    print ('evaluating the model using test ds ...')
    evaluate_model(tested_test)

if __name__ == '__main__':
    main()
