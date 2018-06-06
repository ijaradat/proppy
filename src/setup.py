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
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import MaxAbsScaler
from sklearn.ensemble import ExtraTreesClassifier
import json
import datetime
import logging

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
    param['classification'] =opts.classification
    param['baseline'] = opts.baseline
    param['char_grams'] = opts.char_grams
    param['lexical'] = opts.lexical
    param['style'] = opts.style
    param['readability'] = opts.readability
    param['nela'] = opts.nela
    logging.info ("PARAMETER LIST:_______________________________________________________________________")
    logging.info (param)

    return param

def load_json_dataset (dataset_file):
    logging.info ('loading dataset: ' + dataset_file + ' ...')
    dataset = []
    ds = json.load(open(dataset_file))
    for i, item in enumerate(ds):
        article = document(item['html_text'], item['propaganda_label'], item['gdlt_id'],item['mbfc_url'] )
        dataset.append(article)
    logging.info ('dataset loaded !')
    return dataset

def load_myds(dataset_file):
    logging.info ('loading dataset: ' + dataset_file + ' ...')
    dataset = []
    with codecs.open(dataset_file, 'r', encoding='utf8') as f:
        i = 0
        for line in f:
            line= line.strip()
            fields = line.split('\t')
            article = document(fields[0], fields[-1], fields[4], fields[-2]) # html_text, prop_label, gdelt_id, gdelt_sourceURL
            dataset.append(article)
            i += 1
        f.close()
    logging.info('dataset loaded !')
    return dataset


def load_dataset(dataset_file, classification="binary"):
    logging.info('loading dataset: ' + dataset_file + ' ...')
    dataset =[]
    with codecs.open(dataset_file,'r') as f:
        i=0
        for line in f:
            fields = line.split('\t')
            if fields[0]=='3':
                prop_gold = '1'
            else:
                prop_gold= '-1'
            if classification =='binary':
                article = document(fields[1],prop_gold,str(i),'')
            else:
                article = document(fields[1], fields[0], str(i), '')
            dataset.append(article)
            i+=1
        f.close()
    logging.info ('dataset loaded !')
    return dataset

def read_datsets(param):
    logging.info ('reading datasets ...')
    if param['xtrain'].endswith('.json'):
        xtrain = load_json_dataset(param['xtrain'])
    elif param['xtrain'].endswith('.converted.txt'):
        xtrain = load_myds(param['xtrain'])
    else:
        xtrain = load_dataset(param['xtrain'], param['classification'])

    if param['xdev'].endswith('.json'):
        xdev = load_json_dataset(param['xdev'])
    elif param['xdev'].endswith('.converted.txt'):
        xdev = load_myds(param['xdev'])
    else:
        xdev = load_dataset(param['xdev'],param['classification'])

    if param['test'].endswith('.json'):
        test = load_json_dataset(param['test'])
    elif param['test'].endswith('.converted.txt'):
        test = load_myds(param['test'])
    else:
        test = load_dataset(param['test'], param['classification'])

    logging.info ('done reading data !')
    return xtrain,xdev,test


def construct_pipeline(ds, feats, param):
    feature_set =[]
    logging.info('constructing features pipeline ...')

    if param['baseline'] == True:
        tfidf = feats.extract_baseline_feature(ds)  # each one of these is an sklearn object that has a transform method (each one is a transformer)
        feature_set.append(('tf-idf',tfidf))
    if param['char_grams'] == True:
        char_n_g = feats.extract_char_n_grams(ds)
        feature_set.append(('char-n-g',char_n_g))
    if param['lexical'] == True:
        lexical = feats.extract_lexical(ds)
        feature_set.append(('lexical', lexical))
    if param['style'] == True:
        lexicalstyle_features = feats.extract_lexicalstyle_features(ds)
        feature_set.append(('lexicalstyle', lexicalstyle_features))
    if param['readability'] == True:
        readability_features = feats.extract_readability_features(ds)
        feature_set.append( ('readability', readability_features))
    if param['nela'] == True:
        nela_features = feats.extract_nela_features(ds)
        feature_set.append(('nela', nela_features))


    # feature union is used from the sklearn pipeline class to concatenate features
    features_pipeline =  FeatureUnion(feature_set)  # Pipeline([('vectorizer', vec), ('vectorizer2', vec),....])
    logging.info ('features pipeline ready !')
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


def train_model(train, features_pipeline):
    logging.info('████████████████  𝕋 ℝ 𝔸 𝕀 ℕ 𝕀 ℕ 𝔾  ████████████████')
    #features_pipeline = construct_pipeline(train, feats, param) # call the methods that extract features to initialize transformers
    # ( this method only initializes transformers, pipeline.transform below when called, it calls all transform methods of all tranformers in the pipeline)

    model = LogisticRegression(penalty='l2', class_weight='balanced') # creating an object from the max entropy with L2 regulariation
    logging.info("Computing features")
    X = features_pipeline.transform([doc.text for doc in train]) # calling transform method of each transformer in the features pipeline to transform data into vectors of features
    X = maxabs_scaler.fit_transform(X)
    #print ('maximum absolute values :')
    max_vals = np.amax(X, axis=0) # get the max absolute value of each feature from all data examples
    print (max_vals)
    #print (max_vals[np.argsort(max_vals)[-10:]])  # get the 10 max values from the list of max abs value of each feature above
    #pickle.dump(X, open("train_features.pickle", "wb"))  # dump it (to speed up exp.)
    #X = pickle.load('train_features.pickle')
    #print "Saving features to file"
    Y = [doc.gold_label for doc in train]
    #pickle.dump(Y, open("train_gold.pickle", "wb"))  # dump it (to speed up exp.)
    logging.info ('fitting the model according to given data ...')
    model.fit(X, Y)
    now= datetime.datetime.now().strftime("%I:%M%S%p-%B-%d-%Y")
    model_file_name= now+'maxentr_model.pkl'
    joblib.dump(model,model_file_name ) #pickle the model
    logging.info ('model pickled at : '+ model_file_name)
    return model_file_name
    # print ('features importance :')
    # coefs = model.coef_[0]
    # feature_list = sorted([ (coefs[i], feature) for i, feature in enumerate(features_pipeline.get_feature_names()) ])
    # joblib.dump(feature_list, 'basic_features_mvf.pkl')



def test_model(ds, ds_name, features_pipeline, model_file):
    logging.info ('████████████████   𝕋 𝔼 𝕊 𝕋 𝕀 ℕ 𝔾   ████████████████')
    #features_pipeline= construct_pipeline(test, feats, param)  # call the methods that extract features to initialize transformers
    # ( this method only initializes transformers, pipeline.transform below when called, it calls all transform methods of all tranformers in the pipeline)
    logging.info('loading pickled model from : '+ model_file)
    model = joblib.load(model_file) #load the pickled model
    X = features_pipeline.transform([doc.text for doc in ds])  # calling transform method of each transformer in the features pipeline to transform data into vectors of features
    X = maxabs_scaler.transform(X)
    #print ('maximum absolute values :')
    #max_vals = np.amax(X, axis=0)
    #print (max_vals)
    #print (max_vals[np.argsort(max_vals)[-10:]])
    #pickle.dump(X, open("test_features.pickle", "wb"))  # dump it (to speed up exp.)
    #X = pickle.load('test_features.pickle')
    logging.info ('predicting Y for each given X in test ...')
    Y_ = model.predict(X)  # predicting the labels in this ds via the trained model loaded in the variable 'model'
    for i, doc in enumerate(ds):
        doc.prediction  = Y_[i]

    with codecs.open(model_file+'-predictions-'+ds_name+'.txt', 'w',encoding='utf8') as out:
        out.write('document_id\tsource_URL\tgold_label\tprediction\n')
        for doc in ds:
            out.write(str(doc.id)+'\t'+str(doc.source)+'\t'+doc.gold_label+'\t'+doc.prediction+'\n')
    return ds


def evaluate_model(ds, classification):
    logging.info('████████████████  E V A L U A T I O N  ████████████████')
    # F1 score
    y_true = [doc.gold_label for doc in ds] # getting all gold labels of the ds as one list
    y_pred = [doc.prediction for doc in ds] # getting all model predicted lebels as a list
    if classification =='binary':
        f_score = f1_score(y_true, y_pred, pos_label='1') # calculating F1 score
        precision = precision_score(y_true, y_pred, pos_label='1')
        recall = recall_score(y_true, y_pred, pos_label='1')
    else:
        f_score = f1_score(y_true, y_pred, average='macro') # calculating F1 score
        precision = precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')

    accuracy = accuracy_score(y_true, y_pred)

    logging.info ("F1 score:")
    logging.info (f_score)
    logging.info ("Accuarcy :")
    logging.info (accuracy)
    logging.info ("Precision :")
    logging.info (precision)
    logging.info ("Recall :")
    logging.info (recall)

def main (opts):

    param = parse_parameters(opts) # get parameters from command

    xtrain,xdev,test = read_datsets(param) # loading datsets as lists of document objects
    feats = features(xtrain)  # creating an object from the class features to initialize important global variables such as lexicons and training ds
    #select_features(xtrain, feats)  # feature selection and importance

    train_pipeline = construct_pipeline(xtrain, feats, param)
    model_file = train_model(xtrain, train_pipeline)  # training the model

    dev_pipeline = construct_pipeline(xdev,feats,param)
    tested_dev = test_model(xdev, 'test', dev_pipeline, model_file)  #testing the model with the dev ds

    test_pipeline = construct_pipeline(test,feats,param)
    tested_test = test_model(test,'dev', test_pipeline, model_file)  #testing the model with the test ds

    logging.info ('evaluating the model using dev ds ...')
    evaluate_model(tested_dev, param['classification'])  # evaluating the model on the dev
    logging.info ('evaluating the model using test ds ...')
    evaluate_model(tested_test, param['classification'])  #evaluating the model on the test

if __name__ == '__main__':
    optparser = optparse.OptionParser()

    optparser.add_option(
        "-c", "--classification", default="binary",
        help="experiment type : 'multilabel' or  'binary' classification. i.e prop vs others or 4 classes"
    )
    optparser.add_option(
        "-T", "--xtrain", default="../data/sample.txt",  # "xtrain.txt.filtered.txt"
        help="xtrain set path"
    )
    optparser.add_option(
        "-D", "--xdev", default="../data/sample.txt",  # "xdev.txt.filtered.txt"
        help="xdev set path"
    )
    optparser.add_option(
        "-t", "--test", default="../data/sample.txt",  # "xtest.txt.filtered.txt"
        help="test set path"
    )
    optparser.add_option("-B", "--baseline", dest='baseline', action="store_true", default =False,
                        help="compute tdidf word-n-grams features")
    optparser.add_option("-C", "--chargrams", dest="char_grams", action="store_true", default= False,
                        help="compute char n-grams features")
    optparser.add_option("-L", "--lexical", action="store_true", default=False,
                        help="compute lexical features")
    optparser.add_option("-S", "--style", action="store_true", default=False,
                        help="compute lexical style features")
    optparser.add_option("-R", "--readability", action="store_true", default=False,
                        help="compute readability features")
    optparser.add_option("-N", "--nela", action="store_true", default=True,
                        help="compute Nela features")

    opts = optparser.parse_args()[0]

    main(opts)
