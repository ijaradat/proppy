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
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import MaxAbsScaler

from document import document
from features import *

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


# def compute_features(feats, ds, required_feature):
def compute_features(ds, tfidf=True, lexical=False, style=True, readability=False, nela=False):
    """
    :param ds: input dataset
    :param tfidf: true of you want to compute it
    :param lexical: true of you want to compute it
    :param style:   true of you want to compute it
    :param readability: true of you want to compute it
    :param nela:    true of you want to compute it
    :return: scaled features
    """

    features_instance = features(ds)

    list_of_pipelines = []
    if tfidf:
        list_of_pipelines.append( ('tfidf',  features_instance.extract_baseline_feature(ds)))
    if lexical:
        list_of_pipelines.append((('lexical', features_instance.extract_lexical(ds) )))
    if style:
        list_of_pipelines.append(('style', features_instance.extract_lexicalstyle_features(ds) ))
    if readability:
        list_of_pipelines.append(('readability', features_instance.extract_readability_features(ds) ))
    if nela:
        list_of_pipelines.append(('nela', features_instance.extract_nela_features(ds) ))

    features_pipeline = FeatureUnion(list_of_pipelines)
    print('features pipeline ready!')
    X = features_pipeline.transform([doc.text for doc in ds])

    maxabs_scaler = MaxAbsScaler()
    X = maxabs_scaler.fit_transform(X)
    print ("Features computed")
    return X


# def get_output_file_name(input_file, current_feat):
#     return input_file+"."+current_feat+".features.pickle"

def get_output_file_name(input_file, list_of_features):
    return input_file+"."+".".join(list_of_features)+".features.pickle"



def main(arguments):
    # param = parse_parameters() # get parameters from command

    dataset = read_datsets(arguments['input']) # loading dataset as lists of document objects

    # for current in [x for x in ['tfidf', 'lexical', 'style', 'readability', 'nela'] if arguments[x]]:
    #     X = compute_features(dataset, tfidf=arguments['tfidf'], )
    #     pickle.dump(X, open(get_output_file_name(arguments['input'], current), "wb"))

    # for current in [x for x in ['tfidf', 'lexical', 'style', 'readability', 'nela'] if arguments[x]]:
    X = compute_features(dataset,
                         tfidf=arguments['tfidf'],
                         lexical=arguments['lexical'],
                         style=arguments['style'],
                         readability=arguments['readability'],
                         nela=arguments['nela']
                         )
    output_file = get_output_file_name(arguments['input'],
                            [x for x in ['tfidf', 'lexical', 'style', 'readability', 'nela'] if arguments[x]])
    pickle.dump(X, open(output_file, "wb"))
    print ("Features stored in", output_file)


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


# features_pipeline = FeatureUnion([
#     ('tfidf', load_features_from_pickle("pickle_file"))
#
# ]
#
#
# )

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