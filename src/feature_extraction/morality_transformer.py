
from sklearn.base import TransformerMixin
from feature_functions import Functions
import numpy as np

class morality_vectorizer(TransformerMixin): # any custom transformer needs to inherit sklearn transformMixin or any python class that implements .fit method
    def __init__(self):
        self. Functions = Functions()
        self.seq = ('HarmVirtue, HarmVice, ' +
               'FairnessVirtue, FairnessVice, IngroupVirtue, IngroupVice, ' +
               'AuthorityVirtue, AuthorityVice, PurityVirtue, PurityVice, ' +
               'MoralityGeneral')

        self.num_features = sum([len(s.split(',')) for s in self.seq])
        self.feature_names = self.seq

    def transform(self,X):
        vects = []
        for article in X:

            #article = Functions.fix(' '.join([L for L in article.split('\n') if L.strip() != '']))
            if len(article.strip()) == 0:
                # if text is not available, generate a set of zeros
                seq = ['0'] * self.num_features
            else:
                HarmVirtue, HarmVice, FairnessVirtue, FairnessVice, \
                IngroupVirtue, IngroupVice, AuthorityVirtue, \
                AuthorityVice, PurityVirtue, PurityVice, \
                MoralityGeneral = self.Functions.moral_foundation_feats(article)


                seq = [HarmVirtue, HarmVice,FairnessVirtue, FairnessVice, IngroupVirtue, IngroupVice,
                       AuthorityVirtue, AuthorityVice, PurityVirtue, PurityVice, MoralityGeneral]

            vects.append(seq)
        matrix = np.array(vects).reshape(len(X),len(seq))
        return matrix


    def fit(self):
        return self

    def fit_transform(self,X):
        self.fit()
        return self.transform(X)

    def get_feature_names(self):
        return self.feature_names

    def make_str(seq):
        return [str(s) for s in seq]




