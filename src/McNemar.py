#!/usr/bin/python

import sys
import codecs
  
def McNemar(scoresone, scorestwo, labels, output, binary=True):
    with codecs.open(output, 'w', encoding='utf8') as out:
        muz, mzu = 0, 0
        for (a,b,l) in zip(scoresone, scorestwo, labels):
            #first predictor is correct, second predictor makes a mistake
            if (a == l) and (b != l):
                #print "m10: ", a,b,l,muz
                #out.write("m10: "+'\t'+ str(a)+'\t'+str(b) +'\t'+str(l)+ '\t'+str(muz)+'\n')
                muz += 1
            #second predictor is correct, first predictor makes a mistake
            if (a != l) and (b == l):
                mzu += 1
                #print "m01: ", a,b,l,mzu
                #out.write("m01: "+'\t'+ str(a)+ '\t'+ str(b)+'\t'+ str(l)+'\t'+str(mzu)+'\n')
        print muz+mzu
        out.write(str(muz + mzu) + '\n')
        if mzu+mzu != 0:
            print ((abs(muz-mzu)-1)**2)/(muz+mzu)
            out.write( str(((abs(muz-mzu)-1)**2)/(muz+mzu)))
        else:
            out.write ('cannot divide by zero')
#pag. 227 per confrontare i valori


def read_files (predictions_file1, predictions_file2, direct):
    with codecs.open (predictions_file1, 'r', encoding ='utf8') as f1:
        with codecs.open(predictions_file2, 'r', encoding='utf8') as f2:
            igonre = f1.readline()
            ignore = f2.readline()
            lines1 = f1.readlines()
            lines2 = f2.readlines()
            scoresone = []
            scorestwo = []
            labels =[]
            for line in lines1:
                line = line.strip()
                fields = line.split()
                scoresone.append(float(fields[1]))
                labels.append(float(fields[0]))
            for line in lines2:
                line= line.strip()
                fields = line.split()
                scorestwo.append(float(fields[1]))

            file1_name = predictions_file1.replace(direct,'')
            file2_name = predictions_file2.replace(direct,'')
            out= direct+'McNemar/'+file1_name+'VS'+file2_name

            label_list = [ int(x) for x in list(set(labels)) ]
            if len(label_list)<2:
                sys.exit("ERROR: only one target label found %s"%(label_list[0]))
            if len(label_list)==2:
                binary = True
                if not sum([x-y for x,y in zip(sorted(label_list), [-1, 1])]) == 0:
                    sys.exit("ERROR: in a binary problem the labels are supposed to be -1, 1. Found instead %s"%(",".join(label_list)))
        
                scoresone = [ -1 if x<0 else 1 for x in scoresone ]
                scorestwo = [ -1 if x<0 else 1 for x in scorestwo ]

            else:
                binary = False
                print "multiclass problem with the following labels %s"%(",".join([ str(x) for x in label_list]))
                scoresone = [ int(x) for x in scoresone ]
                scorestwo = [ int(x) for x in scorestwo ]
                if len(set(scoresone)) > len(label_list) or len(set(scorestwo)) > len(label_list):
                    sys.exit("ERROR: found more different labels in one of the prediction files (%d, %d) than in the target label file (%d)"
                             %(len(set(scoresone)), len(set(scorestwo)), len(label_list)))
            
            McNemar(scoresone, scorestwo, labels, out, binary)



char_grams = '../data/predictions.multi/test/test.txtconverted.txt.filtered.txt.char_grams.pred'
lexical = '../data/predictions.multi/test/test.txtconverted.txt.filtered.txt.lexical.pred'
nela = '../data/predictions.multi/test/test.txtconverted.txt.filtered.txt.nela.pred'
readability = '../data/predictions.multi/test/test.txtconverted.txt.filtered.txt.readability.pred'
style = '../data/predictions.multi/test/test.txtconverted.txt.filtered.txt.style.pred'
tfidf = '../data/predictions.multi/test/test.txtconverted.txt.filtered.txt.tfidf.pred'

#style = "/home/gmartino/McNemar/pred-multi/test.txtconverted.txt.filtered.txt.style.pred"
#tfidf = "/home/gmartino/McNemar/pred-multi/test.txtconverted.txt.filtered.txt.tfidf.pred"
#style = "/home/gmartino/McNemar/pred-binary/test.txtconverted.txt.filtered.txt.style.pred"
#tfidf = "/home/gmartino/McNemar/pred-binary/test.txtconverted.txt.filtered.txt.tfidf.pred"

read_files(style, tfidf, '/home/gmartino/McNemar/pred-binary/') 
