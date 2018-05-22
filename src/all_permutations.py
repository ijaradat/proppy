# coding: utf-8
from itertools import combinations


a=["t", "c", "l", "s", "r", "n"]


# print [list(combinations(a, i)) for i in range(1, len(a)+1)]
print '", "'.join('" "'.join(' '.join("-"+i for i in c) for c in combinations(a, i)) for i in range(1, len(a)+1))