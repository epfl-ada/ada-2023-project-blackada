"""This module contains the ExtractorBase class and its subclasses.
These classes are used to extract features from a list of spaCy documents
corresponding to the beer reviews.
"""


from abc import abstractmethod
from spacy.tokens import Doc


class ExtractorBase:
    def __init__(self):
        pass

    @abstractmethod
    def transform(self, docs: [Doc]) -> [str]:
        pass


class LemmaExtractor(ExtractorBase):
    def __init__(self):
        super().__init__()

    def transform(self, docs: [Doc]) -> [str]:
        return [" ".join([token.lemma_ for token in doc]) for doc in docs]


class AdjectiveExtractor(ExtractorBase):
    def __init__(self):
        super().__init__()

    def transform(self, docs: [Doc]) -> [str]:
        return [
            " ".join([token.text for token in doc if token.pos_.startswith("ADJ")])
            for doc in docs
        ]
