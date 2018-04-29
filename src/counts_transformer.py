
from sklearn.base import BaseEstimator, TransformerMixin
from nltk.tokenize import word_tokenize
import numpy as np
import codecs

class counts_vectorizer(TransformerMixin):
    def __init__(self,lexicon_file):
        self.lexicon = self.load_lexicon(lexicon_file)

    def transform(self,X):
        counts = []
        for doc in X:
            counts.append(self.extract_lexical_counts(doc,self.lexicon))
        vect = np.array(counts).reshape(-1,1)
        return vect


    def fit(self):
        return self

    def fit_transform(self,X):
        self.fit()
        return self.transform(X)

    def extract_lexical_counts(self,doc, lexicon):
        tokens = word_tokenize(doc.lower())
        count = 0
        for vocab in lexicon:
            count += tokens.count(vocab)  # count the number of all vocab in the whole document (document = a list of tokens)
        return count

    def load_lexicon(self,file):
        print('Loading lexicon from ' + file + ' ...')
        lexicon = []
        with codecs.open(file, 'r') as f:
            for line in f:
                line = line.strip()
                lexicon.append(line)
            f.close()
            print('done!')
            return lexicon





