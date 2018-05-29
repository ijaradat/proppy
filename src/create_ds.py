#!/usr/local/bin/python
# coding: utf-8

import codecs
from setup import *
from features import *

def parse_parameters(opts):

    param = OrderedDict()

    param['dev'] = opts.dev
    param['test'] = opts.test
    param['train'] = opts.train
    param['sources'] = opts.sources
    param['new'] = opts.new
    param['pred'] = opts.pred

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
def create_dataset(ds_file,sources,random_sources,new_ds_file):
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
    with codecs.open(ds_file, 'r', encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            fields = line.split('\t')
            if fields[-1] == '1':
                if fields[-2] in prop_sources:
                    prop_sources[fields[-2]] += 1
                else:
                    prop_sources[fields[-2]] = 1
            elif fields[-1] == '-1':
                if fields[-2] in nonprop_sources:
                    nonprop_sources[fields[-2]] += 1
                else:
                    nonprop_sources[fields[-2]] = 1
    print("Propagandistic sources statistics:______________________________________________________")
    print(prop_sources)
    print("Non propagandistic sources statistics:__________________________________________________")
    print(nonprop_sources)
    return prop_sources,nonprop_sources

def custom_evaluate(ds,source_list):
    print('████████████████  CUSTOM EVALUATION  ████████████████')
    # F1 score
    y_true = [doc.gold_label for doc in ds]  # getting all gold labels of the ds as one list
    y_pred = [doc.prediction for doc in ds]  # getting all model predicted lebels as a list
    positive_insource_instances = [doc for d in ds if d.gold_label=='1' and d.source in source_list]
    positive_outsource_instances =[doc for d in ds if d.gold_label=='1' and d.source not in source_list]

    insource_pos_y_pred = [doc.prediction for doc in positive_insource_instances]
    insource_pos_y_gold = [doc.gold_label for doc in positive_insource_instances]
    outsource_pos_y_pred = [doc.prediction for doc in positive_outsource_instances]
    outsource_pos_y_gold = [doc.gold_label for doc in positive_outsource_instances]

    print ('Evaluation on all instances:')
    f_score = f1_score(y_true, y_pred, average='macro')  # calculating F1 score
    accuracy = accuracy_score(y_true, y_pred)
    print ("F1 score:")
    print (f_score)
    print ("Accuarcy :")
    print (accuracy)

    print ('Evaluation on in-source positive instances only: ')
    f_score = f1_score(insource_pos_y_gold, insource_pos_y_pred, average='macro')  # calculating F1 score
    accuracy = accuracy_score(insource_pos_y_gold, insource_pos_y_pred)
    print ("F1 score:")
    print (f_score)
    print ("Accuarcy :")
    print (accuracy)

    print ('Evaluation on out-source positive instances only:')
    f_score = f1_score(outsource_pos_y_gold, outsource_pos_y_pred, average='macro')  # calculating F1 score
    accuracy = accuracy_score(outsource_pos_y_gold, outsource_pos_y_pred)
    print ("F1 score:")
    print (f_score)
    print ("Accuarcy :")
    print (accuracy)

def main(opts):
    list_sources_in_ds('../data/test.dist.converted.txt')
    param = parse_parameters(opts)  # get parameters from command
    selected_sources = param['sources'].split(',')
    prop_sources,nonprop_sources = list_sources_in_ds(param['train'])
    random_sources = nonprop_sources.keys()

    create_dataset(param['train'],selected_sources,random_sources,param['new'])

    new_train, dev, test = read_new_datsets(param)  # loading datsets as lists of document objects
    feats = features(new_train)  # creating an object from the class features to initialize important global variables such as lexicons and training ds

    train_model(new_train, feats)  # training the model

    tested_dev = test_model(dev, feats, 'test',param['pred'])  # testing the model with the dev ds
    tested_test = test_model(test, feats, 'dev',param['pred'])  # testing the model with the test ds

    print ('evaluating the model on dev ds ...')
    custom_evaluate(tested_dev,selected_sources)
    print ('evaluating the model on test ds ...')
    custom_evaluate(tested_test,selected_sources)

if __name__ == '__main__':
    optparser = optparse.OptionParser()

    optparser.add_option(
        "-s", "--sources", default ='http://www.shtfplan.com/' ,
        help="list of selected propagandistic sources, type each source URL separated by a comma"
    )
    optparser.add_option(
        "-T", "--train", default="../data/train.json.converted.txt",
        help="train ds path"
    )
    optparser.add_option(
        "-d", "--dev", default="../data/dev.json.converted.txt",
        help="dev ds path"
    )
    optparser.add_option(
        "-t", "--test", default="../data/test.json.converted.txt",
        help="test ds path"
    )
    optparser.add_option(
        "-n", "--new", default="../data/new_train.txt",
        help="path where the new train ds will be saved"
    )
    optparser.add_option(
        "-p", "--pred", default="../data/predictions-",
        help="path where the predictions file will be saved"
    )

    opts = optparser.parse_args()[0]
    main(opts)



