# coding: utf-8
import sys
# for Arabic encoding purposes
reload(sys)
sys.setdefaultencoding('utf-8')
import argparse
import codecs
import json
import logging
import pickle

from collections import OrderedDict
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import MaxAbsScaler

from document import document
from features import *

# DEFAULT_SUFFIX = "feats.pickle"
FORMAT = "%(asctime)-15s %(message)s"
logging.basicConfig(format=FORMAT, level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')


def load_json_dataset (dataset_file):
    logging.info('loading dataset: %s', dataset_file)
    # logging.info ()
    dataset = []
    ds = json.load(open(dataset_file))
    for i, item in enumerate(ds):
        article = document(item['html_text'], item['propaganda_label'], i)
        dataset.append(article)
    logging.info('dataset loaded !')
    return dataset


def load_myds(dataset_file):
    logging.info('loading dataset: %s', dataset_file)
    dataset = []
    with codecs.open(dataset_file, 'r', encoding='utf8') as f:
        i = 0
        for line in f:
            # line= line.strip()
            fields = line.split('\t')
            article = document(fields[0], fields[-1], i)
            dataset.append(article)
            i += 1
        f.close()
    print ('dataset loaded !')
    return dataset


def load_dataset(dataset_file):
    logging.info('loading dataset: %s', dataset_file)
    dataset = []
    with codecs.open(dataset_file,'r') as f:
        i=0
        for line in f:
            fields = line.split('\t')
            article = document(fields[1],fields[0],i)
            dataset.append(article)
            i+=1
        f.close()
    logging.info('dataset loaded !')
    return dataset


def read_datsets(input_file):
    if input_file.endswith('.json'):
        dataset = load_json_dataset(input_file)
    elif input_file.endswith('.converted.txt'):
        dataset = load_myds(input_file)
    else:
        dataset = load_dataset(input_file)
    return dataset


# def compute_features(feats, ds, required_feature):
def compute_features(ds, features_instance, tfidf=True, char_grams=False, lexical=False, style=True, readability=False, nela=False):
    """
    :param ds: input dataset
    :param tfidf: true of you want to compute it
    :param lexical: true of you want to compute it
    :param style:   true of you want to compute it
    :param readability: true of you want to compute it
    :param nela:    true of you want to compute it
    :return: scaled features
    """

    list_of_pipelines = []
    if tfidf:
        list_of_pipelines.append( ('tfidf',  features_instance.extract_baseline_feature(ds)))
    if char_grams:
        list_of_pipelines.append( ('char_grams', features_instance.extract_char_n_grams(ds)))
    if lexical:
        list_of_pipelines.append((('lexical', features_instance.extract_lexical(ds) )))
    if style:
        list_of_pipelines.append(('style', features_instance.extract_lexicalstyle_features(ds) ))
    if readability:
        list_of_pipelines.append(('readability', features_instance.extract_readability_features(ds) ))
    if nela:
        list_of_pipelines.append(('nela', features_instance.extract_nela_features(ds) ))

    features_pipeline = FeatureUnion(list_of_pipelines)
    X = features_pipeline.transform([doc.text for doc in ds])

    logging.info("Features computed")
    return X


def get_output_file_name(input_file, list_of_features):
    return input_file+"."+".".join(list_of_features)+".features.pickle"


def display_params(param):
    parameters = "\n".join(["\t"+x+": " + str(y) for x, y in param.iteritems()])
    logging.info("Input parameters:\n%s \n", parameters)


def dump_feature_file(X, output_file):
    pickle.dump(X, open(output_file, "wb"))
    logging.info("Features stored in %s", output_file)


def main(arguments):
    # param = parse_parameters() # get parameters from command
    display_params(arguments)

    datasets = [read_datsets(x) for x in arguments['input']] # loading datasets as lists of document objects
    features_list = [x for x in ['tfidf', 'char_grams', 'lexical', 'style', 'readability', 'nela'] if arguments[x]]

    maxabs_scaler = MaxAbsScaler()

    features_instance = features(datasets[0])

    for i in range(len(datasets)):
        X = compute_features(datasets[i], features_instance,
                             tfidf=arguments['tfidf'],
                             char_grams=arguments['char_grams'],
                             lexical=arguments['lexical'],
                             style=arguments['style'],
                             readability=arguments['readability'],
                             nela=arguments['nela']
                             )
        if i == 0:  # It is the first iteration and we assume this is training
            X = maxabs_scaler.fit_transform(X)
        else:
            X = maxabs_scaler.transform(X)

        dump_feature_file(X, get_output_file_name(arguments['input'][i], features_list) )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, nargs='+', # "../data/xtrain.txt"
                        help="input dataset")

    parser.add_argument("-t", "--tfidf", dest='tfidf', action="store_true",
                        help="compute tdidf features")
    parser.add_argument("-c", "--chargrams", dest="char_grams", action="store_true",
                        help="compute char n-grams features")
    parser.add_argument("-l", "--lexical", action="store_true", default=False,
                        help="compute lexical features")
    parser.add_argument("-s", "--style", action="store_true", default=False,
                        help="compute lexical style features")
    parser.add_argument("-r", "--readability", action="store_true", default=False,
                        help="compute readability features")
    parser.add_argument("-n", "--nela", action="store_true", default=False,
                        help="compute Nela features")

    arguments = parser.parse_args()

    param = OrderedDict()
    param['input'] = arguments.input
    param['tfidf'] = arguments.tfidf
    param['char_grams']=arguments.char_grams
    param['lexical'] = arguments.lexical
    param['style'] = arguments.style
    param['readability'] = arguments.readability
    param['nela'] = arguments.nela

    main(param)