from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

from nltk.stem import SnowballStemmer

from .text_processing import (
    preprocess_text,
    StemmerTokenizer
)


class TfidfSvdKmeans(BaseEstimator):
    """
    Fits a k-means clustering alogrithm using reduced dimensionality
    TF-IDF vectors as features

    Args:
        k (int): Number of clusters

    Attributes:
        pipeline: sklearn pipeline incorpororating TfidfVectorizer and KMeans
        clustering. Note that pipeline parameters may be modified from the
        default.
    """

    def __init__(self, k=3):

        self.k = k
        self.tokenizer = StemmerTokenizer(SnowballStemmer('english'))
        self.pipeline = Pipeline([
            (
                'tfidf',
                TfidfVectorizer(
                    max_df=0.9,
                    min_df=5,
                    ngram_range=(1, 3),
                    tokenizer=self.tokenizer.stem
                )
            ),
            (
                'svd',
                TruncatedSVD(n_components=200)
            ),
            (
                'kmeans',
                KMeans(n_clusters=self.k, verbose=1)
            )
        ])

    def fit(self, texts):
        """
        Preprocesses text and fits an sklearn pipeline incorporating
        TfidfVectorizer and KMeans clusterer

        Args:
            texts: list of strings
        """
        print('Processing text and fitting k-means...')
        texts = preprocess_text(texts)
        self.pipeline.fit(texts)

        print('Done.')
        return self

    def predict(self, texts):
        """
        Applies preprocessing and predicts labels

        Args:
            texts: list of strings

        Returns:
            array of labels
        """

        texts = preprocess_text(texts)
        labels = self.pipeline.predict(texts)

        return labels
