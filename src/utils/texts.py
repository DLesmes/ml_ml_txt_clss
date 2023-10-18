""" module to manage text """

import re
import unicodedata
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


class Cleaner:
    """
    A simple text cleaner class that performs various text cleaning operations.

    Methods:
    --------
    run(x: str) -> str:
        Cleans the input text by normalizing Unicode characters, removing non-alphanumeric
        characters, converting to lowercase, tokenizing, and removing stopwords in both
        Spanish and English.

    Attributes:
    -----------
    None

    Usage:
    ------
    >>> my_cleaner = cleaner()
    >>> cleaned_text = my_cleaner.run("Your input text here.")
    """

    def __init__(self):
        pass

    def run(self, x: str) -> str:
        """
        Cleans the input text using a series of text cleaning operations.

        Parameters:
        -----------
        x : str
            The input text to be cleaned.

        Returns:
        --------
        str
            The cleaned text.

        Example:
        --------
        >>> my_cleaner = cleaner()
        >>> cleaned_text = my_cleaner.run("Your input text here.")
        """
        x = unicodedata.normalize('NFKD', str(x)).encode('ASCII', 'ignore').decode('ascii')
        x = re.sub('[^A-Za-z0-9\s\n]+', '', str(x).lower())
        x = [word for word in word_tokenize(str(x)) if not word in stopwords.words('spanish')+stopwords.words('english')]
        x = " ".join(x)
        return x
