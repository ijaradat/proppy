# coding: utf-8
import argparse
import logging
import pickle

from sklearn.externals import joblib
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score    #, confusion_matrix,
from a_feature_computer import *
from b_trainer import get_gold_labels
FORMAT = "%(asctime)-15s %(message)s"
logging.basicConfig(format=FORMAT, level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

def get_predictions_file_name(features_file):
    return features_file.replace(".features.pickle", ".pred")


def load_model(serialized_model):
    model = joblib.load(serialized_model)
    logging.info("Model loaded from %s", serialized_model)
    return model


def evaluate_model(y_true, y_pred, perform_multiclass=False):
    # score = f1_score(y_true, y_pred, average='macro')  # calculating F1 score
    # print ("\t".join([]))
    # print y_true
    # print y_pred
    # print("\t".join(["F", "Acc"]))
    acc = accuracy_score(y_true, y_pred)
    if perform_multiclass:
        f1 = f1_score(y_true, y_pred, average='macro')
        p = precision_score(y_true, y_pred, average='macro')
        r = recall_score(y_true, y_pred, average='macro')
    else:
        f1 = f1_score(y_true, y_pred, pos_label="1")
        p = precision_score(y_true, y_pred, pos_label="1")
        r = recall_score(y_true, y_pred, pos_label="1")

    return ("\t".join([
            str(f1),
            str(acc),
            str(p),
            str(r)# precision_score(y_true, y_pred),
            # recall_score((y_true, y_pred))
            ])
            )

def main(arguments):
    display_params(arguments)
    model = load_model(arguments['model'])
    # Y = get_gold_labels(arguments['input'])

    X = pickle.load(open(arguments['features'], 'rb'))  # todo not sure if this file is closed, as in with
    logging.info("Features loaded from %s", arguments['features'])

    y_pred = model.predict(X)
    y_true = get_gold_labels(arguments['input'], arguments['multi'])

    f_acc = evaluate_model(y_true, y_pred)
    print("\t".join(["F", "Acc", "P", "R"]))
    print(f_acc)
    output_file = get_predictions_file_name(arguments['features'])
    with open(output_file, 'w') as output:
        output.write("true\tpredicted\n")
        for t, p in zip(y_true, y_pred):
            output.write("{!s}\t{!s}\n".format(t, p))

    logging.info("model dumped to %s", output_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True,  # "../data/xtrain.txt"
                        help="pre-trained model")

    parser.add_argument("-i", "--input", required=True,  # "../data/xtrain.txt"
                        help="input file (either json ot txt) to load the gold labels from")

    parser.add_argument("-f", "--features", required=True,
                        help="input file with pre-computed features in pickle format (use a_feature_computer.py if you don't have this file)")

    parser.add_argument("-c", "--class_multi", action="store_true", default=False,
                        help="perform multi-class classification")

    arguments = parser.parse_args()

    param = OrderedDict()
    param['model'] = arguments.model
    param['input'] = arguments.input
    param['features'] = arguments.features
    param['multi'] = arguments.class_multi

    main(param)

