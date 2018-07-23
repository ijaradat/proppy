
import codecs
import glob


#opens all files in a specific folder and collects results
def get_results(directory):
    for f in glob.glob(directory):
        with codecs.open(f+'-easy.tsv','w', encoding='utf8') as out:
            with codecs.open(f,'r', encoding='utf8') as log:
                subdir = directory.replace('*.out','')
                expNo = f.replace(subdir,'')
                expNo= expNo.replace('slurm-','')
                expNo =expNo.replace('.out','')

                lines = log.readlines()
                for line in lines:
                    if "('sources', " in line:
                        start_index = line.find("('sources', ")
                        last_index = line.find("'),",start_index)
                        sources_slice = line[start_index+12:last_index]
                        sources_slice = sources_slice.replace("'", "")
                        sources =  sources_slice.split(',')
                        if len(sources) ==2:
                            sources.append('X')
                        elif len(sources) ==1:
                            sources.append('X')
                            sources.append('X')

                baseline=""
                char_grams=""
                lexical =""
                style =""
                readability =""
                nela =""
                for line in lines:
                    if "('baseline', " in line:
                        start_index= line.find("('baseline', ")
                        last_index = line.find("), (", start_index)
                        baseline = line[start_index + 13:last_index]
                        baseline = baseline.replace("'", "")
                        break
                for line in lines:
                    if "('char_grams', " in line:
                        start_index= line.find("('char_grams', ")
                        last_index = line.find("), (", start_index)
                        char_grams = line[start_index + 15:last_index]
                        char_grams = char_grams.replace("'", "")
                        break
                for line in lines:
                    if "('lexical', " in line:
                        start_index= line.find("('lexical', ")
                        last_index = line.find("), (", start_index)
                        lexical = line[start_index + 12:last_index]
                        lexical = lexical.replace("'", "")
                        break
                for line in lines:
                    if "('style', " in line:
                        start_index= line.find("('style', ")
                        last_index = line.find("), (", start_index)
                        style = line[start_index + 10:last_index]
                        style = style.replace("'", "")
                        break
                for line in lines:
                    if "('readability', " in line:
                        start_index= line.find("('readability', ")
                        last_index = line.find("), (", start_index)
                        readability = line[start_index + 16:last_index]
                        readability = readability.replace("'", "")
                        break
                for line in lines:
                    if "('nela', " in line:
                        start_index= line.find("('nela', ")
                        last_index = line.find(")])", start_index)
                        nela = line[start_index + 9:last_index]
                        nela = nela.replace("'", "")
                        break

                scores=[]
                for i in range(len(lines) - 1):
                    if "F1 score:" in lines[i] or "Accuarcy :" in lines[i] or "Recall :" in lines[i] or "Precision :" in lines[i]:
                        fields = lines[i+1].split()
                        scores.append(fields[-1])
                scores_string=""
                for score in scores:
                    scores_string +=score+'\t'
                out.write(expNo+"\tdone\t"+baseline+"\t"+char_grams+"\t"+lexical+"\t"+style+"\t"+readability+"\t"+nela+"\t"+sources[0]+'\t'+sources[1]+'\t'+sources[2]+'\t'+scores_string+'\n')

def collect_from_files(directory):
    dir = directory.replace('*.tsv','')
    with codecs.open(dir+'combned.tsv', 'w', encoding='utf8') as out:
        for f in glob.glob(directory):
            with codecs.open(f , 'r', encoding='utf8') as log:
                lines = log.readlines()
                for line in lines:
                    out.write(line)
                log.close()
        out.close()








directory = '../results/slurms_notfidf_7/*.out'
results = '../results/slurms_notfidf_7/*.tsv'
get_results(directory)
collect_from_files(results)