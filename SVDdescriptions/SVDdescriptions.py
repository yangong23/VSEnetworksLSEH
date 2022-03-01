# ----------------------------------------------------------------
# Created by Yan Gong
# Last revised: Feb 2022
# Reference: Efficient Learning: Semantically Enhanced Hard Negatives for Visual Semantic Embeddings.
# -----------------------------------------------------------------

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from sklearn.decomposition import TruncatedSVD
import numpy as np
from nltk.stem.porter import *
from nltk.corpus import stopwords

# loading data and preprocessing
def loading_data_and_preprocessing(Source_dir='.'):

    print('loading data')
    news_df = []
    with open(Source_dir, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            news_df.append(line)
    news_df = pd.Series(news_df)
    print('documents number: ', len(news_df))

    news_df = news_df.str.replace("[^a-zA-Z#]", " ")
    # make all text lowercase
    news_df = news_df.apply(lambda x: x.lower())

    # delete “it”、“they”、“am”、“been”、“about”、“because”、“while”
    nltk.download('stopwords')
    stop_words = stopwords.words('english')

    # tokenization
    tokenized_doc = news_df.apply(lambda x: x.split())
    # remove stop-words
    tokenized_doc = tokenized_doc.apply(lambda x: [item for item in x if item not in stop_words])

    # stemmer
    stemmer = PorterStemmer()
    # stemmer = SnowballStemmer("english")
    print('preprocessing data: e.g. porter stemmer')
    tokenized_doc_ste = tokenized_doc
    for i in range(len(tokenized_doc)):
        singles = [stemmer.stem(ste) for ste in tokenized_doc[i]]
        tokenized_doc_ste[i] = singles

    # detokenized
    detokenized_doc_ste = []
    for i in range(len(news_df)):
        t = ' '.join(tokenized_doc_ste[i])
        detokenized_doc_ste.append(t)
    dataset_ste = detokenized_doc_ste

    # TF-IDF vector
    vectorizer = TfidfVectorizer(stop_words='english', smooth_idf=True)
    X = vectorizer.fit_transform(dataset_ste)

    return X

def main():
    source_dir = '/home/lunet/coyg4/data/coco/coco_precomp/train_caps.txt'
    output_dir = './output/'
    NumSV = 400 #number of singular value

    X = loading_data_and_preprocessing(Source_dir=source_dir)
    print('SVD processing')
    svd_model = TruncatedSVD(n_components=NumSV, algorithm='randomized', n_iter=100, random_state=122)
    lsa = svd_model.fit_transform(X)

    # save description vectors
    np.savetxt(output_dir + 'train_svd.txt', lsa, fmt='%s', delimiter=',')

if __name__ == '__main__':

    main()