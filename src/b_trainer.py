# coding: utf-8

import argparse
import logging
import pickle

from collections import OrderedDict
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression

from a_feature_computer import read_datsets, display_params
# from setup import load_json_dataset, load_myds, load_dataset


FORMAT = "%(asctime)-15s %(message)s"
logging.basicConfig(format=FORMAT, level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')


def get_output_file_name(features_file, perform_multiclass=False):
    if perform_multiclass:
        return features_file.replace(".features.pickle", ".mdl")
    else:
        return features_file.replace(".features.pickle", ".binary.mdl")


def get_gold_labels(input_file, perform_multiclass=False):
    dataset = read_datsets(input_file, perform_multiclass)  # loading dataset as lists of document objects
    Y = [doc.gold_label for doc in dataset]
    logging.info("Labels loaded from %s", input_file)
    return Y


def train_model(X, Y):
    model = LogisticRegression(penalty='l2')
    logging.info('training the model')
    model.fit(X, Y)
    logging.info("Model trained")
    return model


def main(arguments):
    display_params(arguments)
    Y = get_gold_labels(arguments['input'], arguments['multi'])

    X = pickle.load(open(arguments['features'], 'rb'))  # todo not sure if this file is closed, as in with
    logging.info("Features loaded from %s", arguments['features'])

    model = train_model(X, Y)

    output_file = get_output_file_name(arguments['features'], param['multi'])
    joblib.dump(model, output_file)  # dump the model
    logging.info("model dumped to %s", output_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True,  # "../data/xtrain.txt"
                        help="input file (either json ot txt) to load the gold labels from")

    parser.add_argument("-f", "--feats", required=True,
                        help="input file with pre-computed features in pickle format (use a_feature_computer.py if you don't have this file)")

    parser.add_argument("-m", "--multi", action="store_true", default=False,
                        help="perform multi-class classification")

    arguments = parser.parse_args()

    param = OrderedDict()
    param['input'] = arguments.input
    param['features'] = arguments.feats
    param['multi'] = arguments.multi
    main(param)

