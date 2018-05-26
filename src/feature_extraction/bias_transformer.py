
from sklearn.base import TransformerMixin
from feature_functions import Functions
import numpy as np

class bias_vectorizer(TransformerMixin): # any custom transformer needs to inherit sklearn transformMixin or any python class that implements .fit method
    def __init__(self):
        self. Functions = Functions()
        self.seq = ('bias_count, assertives_count, ' +
               'factives_count, hedges_count, implicatives_count, ' +
               'report_verbs_count, positive_op_count, negative_op_count, ' +
               'wneg_count, wpos_count, wneu_count, sneg_count, ' +
               'spos_count, sneu_count, NB_pobj, NB_psubj')

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

                NB_pobj, NB_psubj = self.Functions.subjectivity(article)
                bias_count, assertives_count, factives_count, hedges_count, \
                implicatives_count, report_verbs_count, positive_op_count, \
                negative_op_count, wneg_count, wpos_count, wneu_count, \
                sneg_count, spos_count, \
                sneu_count = self.Functions.bias_lexicon_feats(article)


                seq = [bias_count, assertives_count,
                       factives_count, hedges_count, implicatives_count,
                       report_verbs_count, positive_op_count, negative_op_count,
                       wneg_count, wpos_count, wneu_count, sneg_count, spos_count,
                       sneu_count,NB_pobj, NB_psubj]

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




