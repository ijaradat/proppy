from setup import load_datset
from sklearn.feature_extraction.text import TfidfVectorizer

print ('initialization: ')
xtrain = load_datset('../data/xtrain.txt')
vectorizer =  TfidfVectorizer(analyzer="word",tokenizer=None,ngram_range=(1, 3))
documents = [doc.text for doc in xtrain]
vectorizer.fit_transform(documents)


def extract_baseline_feature(ds):
    global vectorizer
    for doc in ds:
        doc.baseline_feature = vectorizer.transform([doc.text])
    return vectorizer