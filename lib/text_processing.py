import re
import string

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import emoji

URL_MATCH = r'(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})'


def replace_emojis(text):
    """
    Replaces emojis with text
    """
    return emoji.demojize(text)

def regex_substitutions(text):
    """
    Replaces URLs, numbers, etc.

    Args:
        text (string): Raw text

    Returns:
        text with substitutions applied
    """

    text = re.sub(URL_MATCH, 'URL', text)
    text = re.sub(r'\([0-9]+\)', 'BRACKETNUM', text)
    text = re.sub(r'[0-9]+\.', 'STOPNUM', text)
    text = re.sub(r'[0-9]', 'd', text)
    text = re.sub(r'\s+', ' ', text)

    return text

class StemmerTokenizer():
    """
    Class to stem (or lemmatize) text.

    stem method returns list of stemmed or lemmatized tokens.

    Args:
        stemmer: Expects an instance of an nltk stemmer or lemmatizer.
            Must implement a stem method that returns a list of strings
    """

    def __init__(self, stemmer):

        self.stemmer = stemmer

    def stem(self, text):
        """
        Removes stop words and punctuation and stems and tokenizes text.

        Args:
            text (string): Input text to tokenize

        Returns:
            list of stemmed/lemmatized tokens
        """
        punct_trans_table = str.maketrans(' ', ' ', string.punctuation)
        stop_words = stopwords.words('english')

        tokenized_text = word_tokenize(text)

        text_wo_punctuation = [
            token.translate(punct_trans_table) for token in tokenized_text]

        filtered_text = [
            self.stemmer.stem(token) for token in text_wo_punctuation
            if token.lower() not in stop_words and
            token != ' ' and token
        ]

        return filtered_text

def preprocess_text(texts):
    """
    Apply preprocessing to list of strings

    Args:
        texts (list): List of strings

    Returns:
        List of processed strings
    """

    texts = [regex_substitutions(replace_emojis(text)) for text in texts]

    return texts
