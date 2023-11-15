"""
Module containing classes and functionality used to embed lists of reviews.

Classes
-------
- EmbedderBase: Abstract base class for all embedders.
- CountEmbedder: Uses a count vectorizer from sklearn.
- TfidfEmbedder: Uses a Tfidf vectorizer from sklearn.
- BertEmbedder: Uses 'bert-base-uncased' from HF.
- SentenceTransformerEmbedder: Uses 'all-MiniLM-L6-v2' from sentence-transformers.

Functions
---------
- _get_torch_device(): Returns the device to be used for PyTorch operations.
"""


from abc import abstractmethod, ABC
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


class EmbedderBase(ABC):
    """Abstract class for all embedders which provides an interface for embedding reviews."""
    
    def __init__(self) -> None:
        super().__init__()

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the embedder."""
        pass

    @abstractmethod
    def transform(self, reviews: list[str]) -> np.ndarray:
        """
        Embed a list of reviews.

        Parameters
        ----------
        reviews : list[str]
            List of reviews to be embedded.
        
        Returns
        -------
        np.ndarray
            Array of embeddings.
        """
        pass


class CountEmbedder(EmbedderBase):
    """Embedder which uses a count vectorizer to transform reviews to counts."""

    def __init__(self) -> None:
        super().__init__()
        self.vectorizer = CountVectorizer()

    @property
    def name(self) -> str:
        return "CountEmbedder"

    def transform(self, reviews: list[str]) -> np.ndarray:
        """
        Embed a list of reviews by counting occurence of words.

        Parameters
        ----------
        reviews : list[str]
            List of reviews to be converted to counts.

        Returns
        -------
        np.ndarray
            Array of counts.
        """
        return self.vectorizer.fit_transform(reviews).toarray()


class TfidfEmbedder(EmbedderBase):
    """Embedder which uses a Tfidf vectorizer to transform reviews to Tfidf representations."""

    def __init__(self) -> None:
        super().__init__()
        self.transformer = TfidfVectorizer()

    @property
    def name(self) -> str:
        return "TfidfEmbedder"

    def transform(self, reviews: list[str]) -> np.ndarray:
        """
        Embed a list of reviews using Tfidf transformation.

        Parameters
        ----------
        reviews : list[str]
            List of reviews to be converted to transformed.

        Returns
        -------
        np.ndarray
            Array of Tfidf embeddings.
        """
        return self.transformer.fit_transform(reviews).toarray()
    

class BertEmbedder(EmbedderBase):
    """Embedder which uses Bert to convert reviews to Bert embeddings."""

    def __init__(self) -> None:
        super().__init__()

        # Get GPU if available
        self.device = _get_torch_device()

        # Load pre-trained model & tokenizer
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.model = BertModel.from_pretrained("bert-base-uncased").to(self.device)
        self.model.eval()

    @property
    def name(self) -> str:
        return "BertEmbedder"

    def transform(self, reviews: list[str]) -> np.ndarray:
        """
        Embed a list of reviews using Bert.

        Parameters
        ----------
        reviews : list[str]
            List of reviews to be converted to Bert embeddings.

        Returns
        -------
        np.ndarray
            Array of Bert embeddings.
        """
        # Convert to list if not already (e.g. for Pandas Series or one review is passed)
        reviews = list(reviews)

        # Get tokenized input
        tokenized_reviews = self.tokenizer(reviews, padding=True, truncation=True, return_tensors="pt").to(self.device)

        with torch.no_grad():
            # Get Bert embeddings for all tokens and hidden layers
            output = self.model(**tokenized_reviews, output_hidden_states=True)

        # Return the average embedding of all tokens for each input in the second last hidden layer of the transformer
        # see https://mccormickml.com/2019/05/14/Bert-word-embeddings-tutorial/
        avg_embedding = output.hidden_states[-2].mean(dim=1)

        return avg_embedding.cpu().numpy()
    

class SentenceTransformerEmbedder(EmbedderBase):
    """Embedder which uses sentence-transformers to convert reviews to sentence-transformer embeddings."""

    def __init__(self) -> None:
        super().__init__()
        self.device = _get_torch_device()
        self.model = SentenceTransformer("all-MiniLM-L6-v2").to(self.device)

    @property
    def name(self) -> str:
        return "SentenceTransformerEmbedder"

    def transform(self, reviews: list[str]) -> np.ndarray:
        """
        Embed a list of reviews using sentence-transformers.

        Parameters
        ----------
        reviews : list[str]
            List of reviews to be converted to sentence-transformer embeddings.

        Returns
        -------
        np.ndarray
            Array of sentence-transformer embeddings.
        """
        return self.model.encode(reviews, convert_to_numpy=True)


def _get_torch_device() -> torch.device:
    """
    Returns the device to be used for PyTorch operations.

    Returns
    -------
        torch.device: Device to be used for PyTorch operations.
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")
