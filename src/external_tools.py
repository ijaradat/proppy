import itertools
import codecs


def generate_combinations_from_list(mylist):
    with codecs.open('sources_combinaition.txt', 'w', encoding='utf8') as out:
        combs = []
        for i in xrange(1, len(mylist)+1):
            els = [list(x) for x in itertools.combinations(mylist, i)]
            combs.extend(els)
        for comb in combs:
            line=''
            for item in comb:
                line += item+','
            line = line[:-1]
            out.write(line+'\n')
        out.close()



sources=['http://freedomoutpost.com/' ,'http://www.shtfplan.com/', 'http://clashdaily.com/',
        'http://breaking911.com','https://www.lewrockwell.com/','https://remnantnewspaper.com',
        'http://personalliberty.com/', 'http://www.frontpagemag.com/','http://www.vdare.com/',
         'http://thewashingtonstandard.com/']
generate_combinations_from_list(sources)
