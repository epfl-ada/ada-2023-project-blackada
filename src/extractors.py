"""
SpaCy Document Feature Extraction Module

This module offers classes for extracting linguistic features from spaCy Doc objects, ideal for text analysis tasks like beer review analysis. Each class inherits from the abstract base class ExtractorBase and implements a unique feature extraction method.

Classes:
- ExtractorBase: Defines the feature extraction interface.
- LemmaExtractor: Retrieves lemmatized word forms from documents.
- AdjectiveExtractor: Isolates adjectives from documents.
- DummyExtractor: Extracts the original text of documents.

These classes enable the transformation of spaCy Docs into analyzable data forms like lemmas, adjectives, or raw text.
"""


from abc import abstractmethod, ABC
from spacy.tokens import Doc
from spacy.lang.en.stop_words import STOP_WORDS
from typing import List


class ExtractorBase(ABC):
    """
    Abstract base class for feature extraction from spaCy documents.

    This class should be subclassed to create specific feature extractors.
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def transform(self, docs: List[Doc]) -> List[str]:
        """
        Abstract method to transform a list of spaCy Docs into extracted features.

        Parameters:
        docs (List[Doc]): A list of spaCy Doc objects.

        Returns:
        List[str]: A list of extracted features in string format.
        """
        pass


class LemmaExtractor(ExtractorBase):
    """
    Extractor that transforms spaCy documents into lists of lemmas.
    """

    def __init__(self):
        super().__init__()

    def transform(self, docs: List[Doc]) -> List[str]:
        """
        Transforms documents into a list of lemmatized word strings.

        Parameters:
        docs (List[Doc]): A list of spaCy Doc objects.

        Returns:
        List[str]: A list of strings, each string containing the lemmatized words of a document.
        """
        return [" ".join([token.lemma_ for token in doc]) for doc in docs]

    @property
    def name(self) -> str:
        return "LemmaExtractor"

class AdjectiveExtractor(ExtractorBase):
    """
    Extractor that retrieves adjectives from spaCy documents.
    """

    def __init__(self):
        super().__init__()

    def transform(self, docs: List[Doc]) -> List[str]:
        """
        Transforms documents into a list of strings containing adjectives.

        Parameters:
        docs (List[Doc]): A list of spaCy Doc objects.

        Returns:
        List[str]: A list of strings, each containing the adjectives of a document.
        """
        return [
            " ".join([token.text for token in doc if token.pos_.startswith("ADJ")])
            for doc in docs
        ]

    @property
    def name(self) -> str:
        return "AdjectiveExtractor"


class StopwordExtractor(ExtractorBase):
    """
    Extractor that removes stopwords from spaCy documents.
    """

    def __init__(self):
        super().__init__()

        # Add custom stopwords
        self.STOPWORDS = STOP_WORDS
        additonal_stopwords = [
            "beer",
        ]
        self.STOPWORDS.update(additonal_stopwords)

    def transform(self, docs: List[Doc]) -> List[str]:
        """
        Removes stopwords from documents.

        Parameters:
        docs (List[Doc]): A list of spaCy Doc objects.

        Returns:
        List[str]: A list of strings, each containing the non-stopwords of a document.
        """
        return [
            " ".join([token.text for token in doc if not token.is_stop and token.text not in self.STOPWORDS])
            for doc in docs
        ]

    @property
    def name(self) -> str:
        return "StopwordExtractor"

class DummyExtractor(ExtractorBase):
    """
    Extractor that returns the original text of the spaCy documents.
    """

    def __init__(self):
        super().__init__()

    def transform(self, docs: List[Doc]) -> List[str]:
        """
        Returns the original text of each document.

        Parameters:
        docs (List[Doc]): A list of spaCy Doc objects.

        Returns:
        List[str]: A list containing the original text of each document.
        """
        return [doc.text for doc in docs]

    @property
    def name(self) -> str:
        return "DummyExtractor"