import codecs

def replace_separator(data_file):
    with codecs.open(data_file, 'r') as f:
        with codecs.open(data_file+'-fixed.txt', 'w') as out:
            for line in f:
                line = line.replace(',','\t',1) # replace the first comma separator with a tab separator (only the first occurence)
                out.write(line)
            out.close()
            f.close()

def separate_liwc_lexicons(file):
    with codecs.open(file,'r') as f:
        swear =[]
        hear = []
        sexual =[]
        see =[]
        negations =[]
        number = []
        money =[]
        first_pers_sing =[]
        sec_pers =[]
        for line in f:
            line=line.strip()
            fields= line.split('\t')
            if '121' in fields:
                swear.append(fields[0])
            if '62' in fields:
                hear.append(fields[0])
            if '73' in fields:
                sexual.append(fields[0])
            if '61' in fields:
                see.append(fields[0])
            if '15' in fields:
                negations.append(fields[0])
            if '24' in fields:
                number.append(fields[0])
            if '113' in fields:
                money.append(fields[0])
            if '4' in fields:
                first_pers_sing.append(fields[0])
            if '6' in fields:
                sec_pers.append(fields[0])
        f.close()
        i=1
        for lex in [swear,hear,sexual,see,negations,number,money,first_pers_sing,sec_pers]:
            with codecs.open('../data/lexicons/'+str(i)+'.txt', 'w') as out:
                for word in lex:
                    word=word.replace('*','')
                    out.write(word+'\n')
                out.close()
                i+=1

def separate_subjectives(file):
    with codecs.open(file, 'r') as f:
        weak_subj=[]
        strong_subj=[]
        for line in f:
            line=line.strip()
            fields = line.split(' ')
            subj = fields[0].split('=')
            word_fields = fields[2].split('=')
            word = word_fields[1]
            if subj[1] == 'weaksubj':
                weak_subj.append(word)
            elif subj[1] == 'strongsubj' :
                strong_subj.append(word)
        f.close()
        with codecs.open ('../data/lexicons/weak_subj_wilson.txt','w') as out:
            for sub in weak_subj:
                out.write(sub+'\n')
            out.close()
        with codecs.open('../data/lexicons/strong_subj_wilson.txt','w') as out:
            for sub in strong_subj:
                out.write(sub+'\n')
            out.close()






#separate_liwc_lexicons('../data/lexicons/LIWC/LIWC2015_English.txt')
separate_subjectives('../data/lexicons/subjectivity_clues_hltemnlp05/subjectivity_clues_hltemnlp05/subjclueslen1-HLTEMNLP05.txt')