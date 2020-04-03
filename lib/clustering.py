import numpy as np

from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

from nltk.stem import SnowballStemmer

from gensim.models.ldamodel import LdaModel
from gensim.corpora.dictionary import Dictionary

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
        tokenizer: instance of StemmerTokenizer
    """

    def __init__(self, k=10):

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


class LdaClusterer(BaseEstimator):
    """
    Performs Latent Dirichlet Allocation (LDA) on a set of documents, and
    uses the most probable topic as the cluster id.

    Args:
        n_topics: Number of topics

    Attributes:
        lda_dictionary: Dictionary object corresponding to tokens in corpus
        lda = Gensim LdaModel
        tokenizer: instance of StemmerTokenizer
    """

    def __init__(self, n_topics=10):
        self.n_topics = n_topics
        self.tokenizer = StemmerTokenizer(SnowballStemmer('english'))
        self.lda_dictionary = None
        self.lda_corpus = None
        self.lda = None

    def fit(self, texts):
        """
        Preprocess text and fits a LDA model.

        Args:
            texts: list of strings
        """
        print('Processing text and fitting LDA...')

        texts = preprocess_text(texts)
        stemmed_texts = [
            list(set(self.tokenizer.stem(text))) for text in texts]
        self.lda_dictionary = Dictionary(stemmed_texts)
        lda_corpus = [
            self.lda_dictionary.doc2bow(text) for text in stemmed_texts]
        self.lda = LdaModel(lda_corpus, num_topics=self.n_topics)
        print('Done.')

        return self

    def predict(self, texts):
        """
        Returns a list of the most probable topic for each text

        Args:
            texts: list of strings

        Returns:
            array of labels
        """
        topic_max = []
        for text in texts:
            topic_probs = self.lda[
                self.lda_dictionary.doc2bow(self.tokenizer.stem(text))]
            prob = 0.0
            for topic_prob in topic_probs:
                if topic_prob[1] > prob:
                    topic = topic_prob[0]
                prob = topic_prob[1]
            topic_max.append(topic)

        return np.array(topic_max)
