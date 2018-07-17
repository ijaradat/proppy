
import codecs
import glob


#opens all files in a specific folder and collects results
def get_results(directory = '../results/slurms_notfidf_1/*.out'):
    for f in glob.glob(directory):
        with codecs.open(f+'-easy.tsv','w', encoding='utf8') as out:
            with codecs.open(f,'r', encoding='utf8') as log:
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
                scores=[]
                for i in range(len(lines) - 1):
                    if "F1 score:" in lines[i] or "Accuarcy :" in lines[i] or "Recall :" in lines[i] or "Precision :" in lines[i]:
                        fields = lines[i+1].split()
                        scores.append(fields[-1])
                scores_string=""
                for score in scores:
                    scores_string +=score+'\t'
                out.write("done\tT\tF\tF\tF\tF\tF\t"+sources[0]+'\t'+sources[1]+'\t'+sources[2]+'\t'+scores_string+'\n')

def collect_from_files(directory = '../results/slurms_notfidf_1/*.tsv'):
    with codecs.open('combined.tsv', 'w', encoding='utf8') as out:
        for f in glob.glob(directory):
            with codecs.open(f , 'r', encoding='utf8') as log:
                lines = log.readlines()
                for line in lines:
                    out.write(line)
                log.close()
        out.close()









get_results()
collect_from_files()