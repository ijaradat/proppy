#!/usr/local/bin/python
# coding: utf-8

import codecs
from setup import *
from features import *
from sklearn.metrics import precision_score, recall_score
def parse_parameters(opts):

    param = OrderedDict()

    param['dev'] = opts.dev
    param['test'] = opts.test
    param['train'] = opts.train
    param['sources'] = opts.sources
    param['new'] = opts.new
    param['pred'] = opts.pred
    param['fix']=opts.fix
    param['baseline'] = opts.baseline
    param['char_grams'] = opts.char_grams
    param['lexical'] = opts.lexical
    param['style'] = opts.style
    param['readability'] = opts.readability
    param['nela'] = opts.nela

    print ("PARAMETER LIST:_______________________________________________________________________")
    print (param)

    return param


def read_new_datsets(param):
    print ('reading datasets ...')
    train = load_myds(param['new'])
    dev = load_myds(param['dev'])
    test = load_myds(param['test'])
    print ('done reading data !')
    return train,dev,test



#this function creates a databse , half of it is from the list of sources provided as the second parameter,
# the other half is from random non propagandistic sources
def create_dataset(ds_file,sources,random_sources,new_ds_file,fix_number=None):
    articles_from_sources=0
    articles_from_random =0
    with codecs.open(new_ds_file, 'w', encoding='utf8') as out:
        with codecs.open(ds_file, 'r', encoding='utf8') as f:
            lines = f.readlines()
            for line in lines:
                line= line.strip()
                fields = line.split('\t')
                if fields[-2] in sources:
                    assert (fields[-1] == '1')
                    out.write(line+'\n')
                    articles_from_sources+=1
                    if fix_number != None and articles_from_sources >= fix_number:
                        break
            for line in lines:
                line =line.strip()
                fields =line.split('\t')
                if fields[-2] in random_sources:
                    assert (fields[-1] == '-1')
                    out.write(line+'\n')
                    articles_from_random+=1
                    if articles_from_random == 9*articles_from_sources:
                        print ("number of articles from selected sources is: "+ str(articles_from_sources))
                        print ("number of articles from random sources is :"+str(articles_from_random))
                        f.close()
                        out.close()
                        break


#this function creates two dicts one for prop sources and the other for non prop sources in a given dataset
# in each dict, the key is the source URL, and the value is the number of articles from that source in the dataset
def list_sources_in_ds(ds_file):
    prop_sources = dict()
    nonprop_sources =dict()
    nonprop_articles =0
    prop_articles =0
    with codecs.open(ds_file, 'r', encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            fields = line.split('\t')
            if fields[-1] == '1':
                prop_articles+=1
                if fields[-2] in prop_sources:
                    prop_sources[fields[-2]] += 1
                else:
                    prop_sources[fields[-2]] = 1
            elif fields[-1] == '-1':
                nonprop_articles+=1
                if fields[-2] in nonprop_sources:
                    nonprop_sources[fields[-2]] += 1
                else:
                    nonprop_sources[fields[-2]] = 1
    print("Propagandistic sources statistics:______________________________________________________")
    print(prop_sources)
    print('TOTAL = '+str(len(prop_sources)))
    print("Non propagandistic sources statistics:__________________________________________________")
    print(nonprop_sources)
    print('TOTAL = ' + str(len(nonprop_sources)))
    print("Non Propagandistic articles statistics:______________________________________________________")
    print('TOTAL = '+str(nonprop_articles))
    print("Propagandistic articles statistics:__________________________________________________")
    print('TOTAL = ' + str(prop_articles))
    return prop_sources,nonprop_sources

def print_scores(f_score, accuracy, precision, recall):
    print ("F1 score:")
    print (f_score)
    print ("Accuarcy :")
    print (accuracy)
    print ("Precision :")
    print (precision)
    print ('Recall :')
    print (recall)

def custom_evaluate(ds,source_list):
    print('████████████████  CUSTOM EVALUATION  ████████████████')
    # F1 score
    y_true = [doc.gold_label for doc in ds]  # getting all gold labels of the ds as one list
    y_pred = [doc.prediction for doc in ds]  # getting all model predicted lebels as a list

    positive_insource_instances = []
    positive_outsource_instances = []
    all_negative_instances=[]
    all_positive_instances=[]
    for doc in ds:
        if doc.gold_label == '1':
            all_positive_instances.append(doc)
            if doc.source in source_list:
                positive_insource_instances.append(doc)
            else:
                positive_outsource_instances.append(doc)
        else:
            all_negative_instances.append(doc)

    insource_pos_pred = [doc.prediction for doc in positive_insource_instances]
    insource_pos_gold = [doc.gold_label for doc in positive_insource_instances]
    outsource_pos_pred = [doc.prediction for doc in positive_outsource_instances]
    outsource_pos_gold = [doc.gold_label for doc in positive_outsource_instances]
    all_pos_pred = [doc.prediction for doc in all_positive_instances]
    all_pos_gold = [doc.gold_label for doc in all_positive_instances]
    all_neg_pred = [doc.prediction for doc in all_negative_instances]
    all_neg_gold = [doc.gold_label for doc in all_negative_instances]
    print ('Evaluation on all instances:')
    f_score = f1_score(y_true, y_pred, pos_label='1')  # calculating F1 score
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, pos_label='1')
    recall = recall_score(y_true, y_pred, pos_label='1')
    print_scores(f_score,accuracy, precision, recall)

    print ('Evaluation on in-source positive instances only: ')
    f_score = f1_score(insource_pos_gold, insource_pos_pred, pos_label='1')  # calculating F1 score
    accuracy = accuracy_score(insource_pos_gold, insource_pos_pred)
    precision = precision_score(insource_pos_gold, insource_pos_pred, pos_label='1')
    recall= recall_score(insource_pos_gold, insource_pos_pred, pos_label='1')
    print_scores(f_score, accuracy, precision, recall)

    print ('Evaluation on out-source positive instances only:')
    f_score = f1_score(outsource_pos_gold, outsource_pos_pred, pos_label='1')  # calculating F1 score
    accuracy = accuracy_score(outsource_pos_gold, outsource_pos_pred)
    precision = precision_score(outsource_pos_gold, outsource_pos_pred, pos_label='1')
    recall = recall_score(outsource_pos_gold, outsource_pos_pred, pos_label='1')
    print_scores(f_score, accuracy, precision, recall)

    print ('Evaluation on all positive instances only:')
    f_score = f1_score(all_pos_gold, all_pos_pred, pos_label='1')  # calculating F1 score
    accuracy = accuracy_score(all_pos_gold, all_pos_pred)
    precision = precision_score(all_pos_gold, all_pos_pred, pos_label='1')
    recall = recall_score(all_pos_gold, all_pos_pred, pos_label='1')
    print_scores(f_score, accuracy, precision, recall)

    print ('Evaluation on all negative instances only:')
    f_score = f1_score(all_neg_gold, all_neg_pred, pos_label='1')  # calculating F1 score
    accuracy = accuracy_score(all_neg_gold, all_neg_pred)
    precision = precision_score(all_neg_gold,all_neg_pred, pos_label='1')
    recall =  recall_score(all_neg_gold, all_neg_pred, pos_label='1')
    print_scores(f_score, accuracy, precision, recall)

def main(opts):
    #list_sources_in_ds('../data/test.dist.converted.txt')
    param = parse_parameters(opts)  # get parameters from command
    selected_sources = param['sources'].split(',')
    prop_sources,nonprop_sources = list_sources_in_ds(param['train'])
    random_sources = nonprop_sources.keys()

    create_dataset(param['train'],selected_sources,random_sources,param['new'],param['fix'])
    print('a new training dataset created at :'+ param['new'])

    new_train, dev, test = read_new_datsets(param)  # loading datsets as lists of document objects
    feats = features(new_train)  # creating an object from the class features to initialize important global variables such as lexicons and training ds

    train_pipeline = construct_pipeline(new_train, feats, param)
    train_model(new_train, train_pipeline)  # training the model

    dev_pipeline = construct_pipeline(dev, feats, param)
    tested_dev = test_model(dev, 'dev', dev_pipeline,param['pred'])  # testing the model with the dev ds

    test_pipeline = construct_pipeline(test, feats, param)
    tested_test = test_model(test, 'test', test_pipeline,param['pred'])

    print ('evaluating the model on dev ds ...')
    custom_evaluate(tested_dev,selected_sources)
    print ('evaluating the model on test ds ...')
    custom_evaluate(tested_test,selected_sources)

if __name__ == '__main__':
    optparser = optparse.OptionParser()

    optparser.add_option(
        "-s", "--sources", default ='http://www.shtfplan.com/,https://www.lewrockwell.com/' ,
        help="list of selected propagandistic sources, type each source URL separated by a comma"
    )
    optparser.add_option(
        "-f", "--fix", default =None,
        help="fix_number: number of propagandistic articles to collect from the given list of sources in parameter -s. NOTE: if the max no. of articles from the list od sources is \n actually less than the given (fix) it will return the max"

    )
    optparser.add_option(
        "-T", "--train", default="../data/train.dist.converted.txt",
        help="train ds path"
    )
    optparser.add_option(
        "-d", "--dev", default="../data/dev.dist.converted.txt",
        help="dev ds path"
    )
    optparser.add_option(
        "-t", "--test", default="../data/test.dist.converted.txt",
        help="test ds path"
    )
    optparser.add_option(
        "-n", "--new", default="../data/new_train.txt",
        help="full path where the new train ds will be saved"
    )
    optparser.add_option(
        "-p", "--pred", default="../data/predictions",
        help="full path where the predictions file will be saved: e.g:../data/predictions "
    )
    optparser.add_option("-B", "--baseline", dest='baseline', action="store_true", default =True,
                        help="compute tdidf word-n-grams features")
    optparser.add_option("-C", "--chargrams", dest="char_grams", action="store_true", default= False,
                        help="compute char n-grams features")
    optparser.add_option("-L", "--lexical", action="store_true", default=False,
                        help="compute lexical features")
    optparser.add_option("-S", "--style", action="store_true", default=False,
                        help="compute lexical style features")
    optparser.add_option("-R", "--readability", action="store_true", default=False,
                        help="compute readability features")
    optparser.add_option("-N", "--nela", action="store_true", default=False,
                        help="compute Nela features")

    opts = optparser.parse_args()[0]
    main(opts)



