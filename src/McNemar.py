#!/usr/bin/python

import sys
import codecs

def read_files2():
    if len(sys.argv) < 4:
        print "Usage: " + sys.argv[0] + " scores1 scores2 labels"
        sys.exit(1)

    fscoresone = open(sys.argv[1],"r")
    scoresone = [ float(x.rstrip()) for x in fscoresone.readlines() ]
    fscoresone.close()

    fscorestwo = open(sys.argv[2],"r")
    scorestwo = [ float(x.rstrip()) for x in fscorestwo.readlines() ]
    fscorestwo.close()

    flabels = open(sys.argv[3],"r")
    labels = [ float(x.rstrip()) for x in flabels.readlines() ]
    flabels.close()


    McNemar(scoresone,scorestwo, labels)




def McNemar(scoresone, scorestwo, labels, output):
    with codecs.open(output, 'w', encoding='utf8') as out:
        muz, mzu = 0, 0
        for (a,b,l) in zip(scoresone, scorestwo, labels):
            if (a*l > 0) and (b*l < 0 or (l<0 and b*l==0)):
                print "m10: ", a,b,l,muz
                out.write("m10: "+'\t'+ str(a)+'\t'+str(b) +'\t'+str(l)+ '\t'+str(muz)+'\n')
                muz += 1
            if (b*l > 0) and (a*l < 0 or (l<0 and a*l==0)):
                mzu += 1
                print "m01: ", a,b,l,mzu
                out.write("m01: "+'\t'+ str(a)+ '\t'+ str(b)+'\t'+ str(l)+'\t'+str(mzu)+'\n')
        print muz+mzu
        out.write(str(muz+mzu)+'\n')
        print ((abs(muz-mzu)-1)**2)/(muz+mzu)
        out.write( str(((abs(muz-mzu)-1)**2)/(muz+mzu)))

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
            McNemar(scoresone, scorestwo, labels, out)



char_grams = '../data/predictions.bin/test/test.txtconverted.txt.filtered.txt.char_grams.pred'
lexical = '../data/predictions.bin/test/test.txtconverted.txt.filtered.txt.lexical.pred'
nela = '../data/predictions.bin/test/test.txtconverted.txt.filtered.txt.nela.pred'
readability = '../data/predictions.bin/test/test.txtconverted.txt.filtered.txt.readability.pred'
style = '../data/predictions.bin/test/test.txtconverted.txt.filtered.txt.style.pred'
tfidf = '../data/predictions.bin/test/test.txtconverted.txt.filtered.txt.tfidf.pred'

read_files(char_grams, tfidf, '../data/predictions.bin/test/')