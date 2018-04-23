import codecs

def replace_separator(data_file):
    with codecs.open(data_file, 'r') as f:
        with codecs.open(data_file+'-fixed.txt', 'w') as out:
            for line in f:
                line = line.replace(',','\t',1) # replace the first comma separator with a tab separator (only the first occurence)
                out.write(line)
            out.close()
            f.close()

