#!/usr/local/bin/python
# coding: utf-8
import os, sys
# for Arabic encoding purposes
reload(sys)
sys.setdefaultencoding('utf-8')
import codecs
import numpy as np
import optparse
import pickle
from collections import OrderedDict
from features import *
from document import document
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
from sklearn.metrics import f1_score, accuracy_score
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import MaxAbsScaler
from sklearn.ensemble import ExtraTreesClassifier
import json


maxabs_scaler = MaxAbsScaler()


def parse_parameters(opts):
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

def load_json_dataset (dataset_file):
    print ('loading dataset: ' + dataset_file + ' ...')
    dataset = []
    ds = json.load(open(dataset_file))
    for i, item in enumerate(ds):
        article = document(item['html_text'], item['propaganda_label'], item['gdlt_id'],item['gdlt_srcURL'] )
        dataset.append(article)
    print ('dataset loaded !')
    return dataset

def load_myds(dataset_file):
    print ('loading dataset: ' + dataset_file + ' ...')
    dataset = []
    with codecs.open(dataset_file, 'r', encoding='utf8') as f:
        i = 0
        for line in f:
            line= line.strip()
            fields = line.split('\t')
            article = document(fields[0], fields[-1], fields[4], fields[5]) # html_text, prop_label, gdelt_id, gdelt_sourceURL
            dataset.append(article)
            i += 1
        f.close()
    print ('dataset loaded !')
    return dataset


def load_dataset(dataset_file):
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
    print ('dataset loaded !')
    return dataset

def read_datsets(param):
    print ('reading datasets ...')
    if param['xtrain'].endswith('.json'):
        xtrain = load_json_dataset(param['xtrain'])
    elif param['xtrain'].endswith('.converted.txt'):
        xtrain = load_myds(param['xtrain'])
    else:
        xtrain = load_dataset(param['xtrain'])

    if param['xdev'].endswith('.json'):
        xdev = load_json_dataset(param['xdev'])
    elif param['xdev'].endswith('.converted.txt'):
        xdev = load_myds(param['xdev'])
    else:
        xdev = load_dataset(param['xdev'])

    if param['test'].endswith('.json'):
        test = load_json_dataset(param['test'])
    elif param['test'].endswith('.converted.txt'):
        test = load_myds(param['test'])
    else:
        test = load_dataset(param['test'])

    print ('done reading data !')
    return xtrain,xdev,test


def extract_features(ds, feats):

    print('constructing features pipeline ...')
    tfidf = feats.extract_baseline_feature(ds)  # each one of these is an sklearn object that has a transform method (each one is a transformer)
    #lexical = feats.extract_lexical(ds)
    #lexicalstyle_features = feats.extract_lexicalstyle_features(ds)
    #readability_features = feats.extract_readability_features(ds)
    #nela_features = feats.extract_nela_features(ds)
    # feature union is used from the sklearn pipeline class to concatenate features
    features_pipeline =  FeatureUnion([ ('tf-idf',tfidf)
                                        #('lexical', lexical),
                                        #('lexicalstyle', lexicalstyle_features),
                                        #('readability', readability_features),
                                        #('nela', nela_features)
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


def train_model(train, feats):
    print ('████████████████  𝕋 ℝ 𝔸 𝕀 ℕ 𝕀 ℕ 𝔾  ████████████████')
    features_pipeline = extract_features(train, feats) # call the methods that extract features to initialize transformers
    # ( this method only initializes transformers, pipeline.transform below when called, it calls all transform methods of all tranformers in the pipeline)

    model = LogisticRegression(penalty='l2') # creating an object from the max entropy with L2 regulariation
    print "Computing features"
    X = features_pipeline.transform([doc.text for doc in train]) # calling transform method of each transformer in the features pipeline to transform data into vectors of features
    X = maxabs_scaler.fit_transform(X)
    #print ('maximum absolute values :')
    max_vals = np.amax(X, axis=0) # get the max absolute value of each feature from all data examples
    #print (max_vals)
    #print (max_vals[np.argsort(max_vals)[-10:]])  # get the 10 max values from the list of max abs value of each feature above
    pickle.dump(X, open("train_features.pickle", "wb"))  # dump it (to speed up exp.)
    #X = pickle.load('train_features.pickle')
    #print "Saving features to file"
    Y = [doc.gold_label for doc in train]
    #pickle.dump(Y, open("train_gold.pickle", "wb"))  # dump it (to speed up exp.)
    print ('fitting the model according to given data ...')
    model.fit(X, Y)

    joblib.dump(model, 'maxentr_model.pkl') #pickle the model
    print ('model pickled at : maxentr_model.pkl ')

    # print ('features importance :')
    # coefs = model.coef_[0]
    # feature_list = sorted([ (coefs[i], feature) for i, feature in enumerate(features_pipeline.get_feature_names()) ])
    # joblib.dump(feature_list, 'basic_features_mvf.pkl')



def test_model(test, feats, ds_name, predictions_file = '../data/predictions-'):
    print ('████████████████   𝕋 𝔼 𝕊 𝕋 𝕀 ℕ 𝔾   ████████████████')
    features_pipeline= extract_features(test, feats)  # call the methods that extract features to initialize transformers
    # ( this method only initializes transformers, pipeline.transform below when called, it calls all transform methods of all tranformers in the pipeline)
    print('loading pickled model from : maxentr_model.pkl ')
    model = joblib.load('maxentr_model.pkl') #load the pickled model
    X = features_pipeline.transform([doc.text for doc in test])  # calling transform method of each transformer in the features pipeline to transform data into vectors of features
    X = maxabs_scaler.transform(X)
    #print ('maximum absolute values :')
    #max_vals = np.amax(X, axis=0)
    #print (max_vals)
    #print (max_vals[np.argsort(max_vals)[-10:]])
    pickle.dump(X, open("test_features.pickle", "wb"))  # dump it (to speed up exp.)
    #X = pickle.load('test_features.pickle')
    print ('predicting Y for each given X in test ...')
    Y_ = model.predict(X)  # predicting the labels in this ds via the trained model loaded in the variable 'model'
    for i, doc in enumerate(test):
        doc.prediction  = Y_[i]

    with codecs.open(predictions_file+ds_name+'.txt', 'w',encoding='utf8') as out:
        out.write('document_id\tsource_URL\tgold_label\tprediction\n')
        for doc in test:
            out.write(doc.id+'\t'+doc.source+'\t'+doc.gold_label+'\t'+doc.prediction+'\n')
    return test


def evaluate_model(ds):
    # F1 score
    y_true = [doc.gold_label for doc in ds] # getting all gold labels of the ds as one list
    y_pred = [doc.prediction for doc in ds] # getting all model predicted lebels as a list
    f_score = f1_score(y_true, y_pred, average='macro') # calculating F1 score
    accuracy = accuracy_score(y_true, y_pred)
    print ("F1 score:")
    print (f_score)
    print ("Accuarcy :")
    print (accuracy)


def main (opts):

    param = parse_parameters(opts) # get parameters from command

    xtrain,xdev,test = read_datsets(param) # loading datsets as lists of document objects
    feats = features(xtrain)  # creating an object from the class features to initialize important global variables such as lexicons and training ds
    #select_features(xtrain, feats)  # feature selection and importance

    train_model(xtrain, feats)  # training the model

    tested_dev = test_model(xdev, feats,'test')  #testing the model with the dev ds
    tested_test = test_model(test, feats,'dev')  #testing the model with the test ds

    print ('evaluating the model using dev ds ...')
    evaluate_model(tested_dev)  # evaluating the model on the dev
    print ('evaluating the model using test ds ...')
    evaluate_model(tested_test)  #evaluating the model on the test

if __name__ == '__main__':
    optparser = optparse.OptionParser()

    # optparser.add_option(
    #     "-i", "--input", default="../data/sample.txt", # "../data/xtrain.txt"
    #     help="xtrain set path"
    # )
    optparser.add_option(
        "-T", "--xtrain", default="../data/train.json.converted.txt",  # "../data/xtrain.txt"
        help="xtrain set path"
    )
    optparser.add_option(
        "-D", "--xdev", default="../data/dev.json.converted.txt",  # "../data/xdev.txt"
        help="xdev set path"
    )
    optparser.add_option(
        "-t", "--test", default="../data/test.json.converted.txt",  # "../data/test.txtconverted.txt"
        help="test set path"
    )

    opts = optparser.parse_args()[0]

    main(opts)
