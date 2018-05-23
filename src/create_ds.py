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
                    if articles_from_random == articles_from_sources:
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
        for line in f:
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


def main(opts):
    #list_sources_in_ds('../data/train.json.converted.txt')
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

    print ('evaluating the model using dev ds ...')
    evaluate_model(tested_dev)  # evaluating the model on the dev
    print ('evaluating the model using test ds ...')
    evaluate_model(tested_test)  # evaluating the model on the test



if __name__ == '__main__':
    optparser = optparse.OptionParser()

    optparser.add_option(
        "-s", "--sources", default ='https://remnantnewspaper.com,http://personalliberty.com/,http://www.frontpagemag.com/',
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



