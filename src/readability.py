# coding=utf-8

import math
import numpy as np
import string

from nltk import word_tokenize
from sklearn.base import BaseEstimator, TransformerMixin


class LexicalStyle_vectorizer(TransformerMixin):

    # TODO perhaps we should limit the lengths of the texts here as well
    def __init__(self):
        None
        # self.text = text
        # self._process_text(text)

    def transform(self, X):
        # COMPUTE THE FEATURES
        print('Computing readability features...')
        counts = []
        for doc in X:
            # on row; in my case a matrix [5x16000]
            # first try with a vector and then with features
            self._process_text(doc)
            counts.append([self.ttr(),
                           self.hapax_legomena(),
                           self.hapax_dislegomena(),
                           self.honore_R(),
                           self.yule_K()])
        # return transformations (value for each documen).
        # (16,000 array for training)
        print(counts)
        vect = np.array(counts)#.reshape(-1,1)
        print('done')
        return vect

    def fit(self):
        return self

    def fit_transform(self, X):
        self.fit()
        return self.transform(X)


    def _process_text(self, text):
        """
        Produces counters for tokens, types, and legomena.
        In this implementation punctuation marks are discarded.
        :param text:
        :return:
        """
        tokens = [w for w in word_tokenize(text) if w not in string.punctuation]
        types_count = 0
        hapax_legomena_count = 0
        type_freqs = {}
        for token in tokens:
            try:
                type_freqs[token] += 1
            except KeyError:
                type_freqs[token] = 1

        # Counting the number of types appearing i times for all i
        counter_all = {}
        for i in range(1, max(type_freqs.values())+1):
            counter_all[i] = 0
            counter_all[i] += sum([1 for x in type_freqs if type_freqs[x] == i])

        # hapax_legomena_count = sum([1 for x in type_freqs if type_freqs[x] == 1])
        # hapax_dislegomena_count = sum([1 for x in type_freqs if type_freqs[x] == 2])

        # count_tokens = len(tokens)
        # count_types = len(set)
        self.counter = {
            'tokens': len(tokens),
            'types': len(type_freqs),
            'hapax_legomena': counter_all[1],   #    hapax_legomena_count,
            'hapax_dislegomena': counter_all[2], #hapax_dislegomena_count
            'all': counter_all
        }


    def _get_tokens(self, text):
        return [w for w in word_tokenize(text) if w not in string.punctuation]

    def ttr(self):
        """
        Type-token ratio (types/tokens). The number of types id divided by the number of
        tokens. Described in Stamatatos' 2.1
        :return:
        """
        try:
            return float(self.counter['types']) / self.counter['tokens']
        except ZeroDivisionError:
            return 0

    def hapax_legomena(self):
        """
        We compute the number of hapax_legomena and normalize by the
        number of types.

        :return: number of types appearing only once in the text (normalized)
        """
        try:
            return float(self.counter['hapax_legomena']) / self.counter['types']
        except ZeroDivisionError:
            return 0

    def hapax_dislegomena(self):
        """
        We compute the number of hapax_dislegomena and normalize by the
        number of types.

        :return: number of types appearing twice in the text (normalized)
        """

        return float(self.counter['hapax_dislegomena']) / self.counter['types']

    def honore_R(self):
        """
        Computes Honore's Function R:
              100 * log(|tokens\)
           ------------------------
            1 - \hapax_leg\/\types\
        The higher the value of R, the richer the vocabulary. According
        to [1] Honoré considers that |tokens| should be 1300 for the
        computation to stabilize.

        [1] https://www.physics.smu.edu/pseudo/ScienceReligion/MormonStylometric.pdf
        :return:
        """
        return 100 * math.log(self.counter['tokens']) / (1 - (self.counter['hapax_legomena'] / self.counter['types']))

    def yule_K(self):
        """
        Yule's K, defined as
        K = 10^4 (\sum i^2V_i - N) / N^2 for i=1,2,...
        According to [1]
        [1] https://www.physics.smu.edu/pseudo/ScienceReligion/MormonStylometric.pdf
        :return:
        """

        summ = sum([i**2 * self.counter['all'][i] for i in self.counter['all']])
        k = 10**4 * (summ - self.counter['tokens']) / self.counter['tokens']**2
        return k



#
# document = \
# """Subfossil lemurs are primates from Madagascar, especially the extinct giant lemurs,
# represented by subfossils (partially fossilized remains) dating from nearly 26,000 to around
# 560 years ago. Almost all of these species, including the sloth lemurs, koala lemurs and monkey
# lemurs, were living around 2,000 years ago, when humans first arrived on the island. The extinct
# species are estimated to have ranged in size from slightly over 10 kg (22 lb) to roughly 160 kg
# (350 lb). The subfossil sites found around most of the island demonstrate that most giant lemurs
# had wide distributions. Like living lemurs, they had poor day vision and relatively small brains,
# and developed rapidly, but they relied less on leaping, and more on terrestrial locomotion, slow
# climbing, and suspension. Although no recent remains of giant lemurs have been found, oral
# traditions and reported recent sightings by Malagasy villagers suggest that there may be lingering
# populations or very recent extinctions. """
#
# lexstyle = LexicalStyle_vectorizer(document)
# print "Type/token ratio:", lexstyle.ttr(document)
# print "Hapax legomena:", lexstyle.hapax_legomena()
# print "Hapax dislegomena:", lexstyle.hapax_dislegomena()
# print "Honore's R:", lexstyle.honore_R()
# print "Yule's K:", lexstyle.yule_K()