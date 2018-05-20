# coding: utf-8
import sys
# for Arabic encoding purposes
reload(sys)
sys.setdefaultencoding('utf-8')
import argparse
import codecs
import json
import pickle
from collections import OrderedDict
from features import *

from document import document


DEFAULT_SUFFIX = "feats.pickle"

def load_json_dataset (dataset_file):
    print ('loading dataset: ' + dataset_file + ' ...')
    dataset = []
    ds = json.load(open(dataset_file))
    for i, item in enumerate(ds):
        article = document(item['html_text'], item['propaganda_label'], i)
        dataset.append(article)
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


def read_datsets(input_file):
    if input_file.endswith('.json'):
        dataset = load_json_dataset(input_file)
    else:
        dataset = load_dataset(input_file)
    return dataset


def compute_features(feats, ds, required_feature):

    if required_feature == 'tfidf':
        features_pipeline = feats.extract_baseline_feature(ds)
    elif required_feature == 'lexical':
        features_pipeline = feats.extract_lexical(ds)
    elif required_feature == 'style':
        features_pipeline = feats.extract_lexicalstyle_features(ds)
    elif required_feature == 'readability':
        features_pipeline = feats.extract_readability_features(ds)
    elif required_feature == 'nela':
        features_pipeline = feats.extract_nela_features(ds)
    else:
        print("ERROR, I cannot compute features", required_feature)
        exit(1)
    X = features_pipeline.transform([doc.text for doc in ds])

    print ("Features computed")
    return X


def get_output_file_name(input_file, current_feat):
    return input_file+"."+current_feat+".features.pickle"


def main(arguments):
    # param = parse_parameters() # get parameters from command

    dataset = read_datsets(arguments['input']) # loading dataset as lists of document objects
    features_instance = features(dataset)
    for current in [x for x in ['tfidf', 'lexical', 'style', 'readability', 'nela'] if arguments[x]]:
        X = compute_features(features_instance , dataset, current)
        pickle.dump(X, open(get_output_file_name(arguments['input'], current), "wb"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True,  # "../data/xtrain.txt"
        help="input dataset")

    parser.add_argument("-t", "--tfidf", dest='tfidf', action="store_true",
                        help="compute tdidf features")
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
    param['lexical'] = arguments.lexical
    param['style'] = arguments.style
    param['readability'] = arguments.readability
    param['nela'] = arguments.nela

    main(param)

    #
    # parser.add_argument('-i', '--input', dest='kelp_file', required=True, help="input kelp file")
    # parser.add_argument('-t', '--thread-remove', dest='thread', action='store_true',
    #                     help="remove the thread features (remove the others otherwise)")
    #
    # arguments = parser.parse_args()
    # main(arguments)
    #
    # main()




#

# opts = opt


# def parse_parameters():
#     """
#         parameter list
#         ---------------------
#         --input -i          --output -o
#
#     Parse the input parameters
#     :return: <dictionary>  a dictionary of parameters
#     """
#     param = OrderedDict()
#
#     param['input'] = opts.input
#     param['tfidf'] = opts.tfidf
#     param['lexical'] = opts.lexical
#     param['style'] = opts.style
#     param['readability'] = opts.readability
#     param['nela'] = opts.nela
#
#     print ("PARAMETER LIST:_______________________________________________________________________")
#     print (param)
#
#     return param