from setup import load_datset
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
import codecs


print ('initialization: ')
xtrain = load_datset('../data/xtrain.txt')
vectorizer =  TfidfVectorizer(analyzer="word",ngram_range=(1, 3))
documents = [doc.text for doc in xtrain]
vectorizer.fit_transform(documents)


def extract_baseline_feature(ds):
    global vectorizer
    for doc in ds:
        doc.baseline_feature = vectorizer.transform([doc.text])
    return vectorizer


def extract_from_lexicon(ds,lexicon):
    for doc in ds:
        tokens = word_tokenize(doc.text.lower())
    count = 0
    for word in lexicon:
        count += tokens.count(word)
    return count


def load_lexicon(file):
    lexicon=[]
    with codecs.open(file,'r') as f:
        for line in f:
            line=line.strip()
            lexicon.append(line)
        f.close()
        return lexicon

