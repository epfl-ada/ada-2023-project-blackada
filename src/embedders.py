from abc import abstractmethod

from abc import abstractmethod, ABC
from enum import Enum
import numpy as np
import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

class EmbeddorBase(ABC):
    """Abstract class for all embedders which provides an interface for embedding reviews."""
    def __init__(self) -> None:
        pass

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





class CountEmbeddor(EmbeddorBase):
    """Embedder which uses a count vectorizer to transform reviews to counts."""

    def __init__(self) -> None:
        super().__init__()
        self.vectorizer = CountVectorizer()

    @property
    def name(self) -> str:
        return "CountEmbeddor"

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


class TFIDFEmbeddor(EmbeddorBase):
    """Embedder which uses a TFIDF vectorizer to transform reviews to TFIDF representations."""
    def __init__(self) -> None:
        super().__init__()
        self.transformer = TfidfVectorizer()

    @property
    def name(self) -> str:
        return "TFIDFEmbeddor"

    def transform(self, reviews: list[str]) -> np.ndarray:
        """
        Embed a list of reviews using TFIDF transformation.

        Parameters
        ----------
        reviews : list[str]
            List of reviews to be converted to transformed.

        Returns
        -------
        np.ndarray
            Array of TFIDF embeddings.
        """
        return self.transformer.fit_transform(reviews).toarray()
    

class BERTEmbeddor(EmbeddorBase):
    """Embedder which uses BERT to convert reviews to BERT embeddings."""
    def __init__(self) -> None:
        super().__init__()

        # Get GPU if available
        self.device = _get_torch_device()
        # Check if GPU is available # device = torch.device("mps" if torch.cuda.is_available() else "cpu") device = torch.device("mps")

        # Load pre-trained model & tokenizer
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.model = BertModel.from_pretrained("bert-base-uncased").to(self.device)
        self.model.eval()

    @property
    def name(self) -> str:
        return "BERTEmbeddor"

    def transform(self, reviews: list[str]) -> np.ndarray:
        """
        Embed a list of reviews using BERT.

        Parameters
        ----------
        reviews : list[str]
            List of reviews to be converted to BERT embeddings.

        Returns
        -------
        np.ndarray
            Array of BERT embeddings.
        """
        reviews = list(reviews)
        # Get tokenized input
        tokenized_reviews = self.tokenizer(reviews, padding=True, return_tensors="pt").to(self.device)

        with torch.no_grad():
            # Get BERT embeddings for all tokens and hidden layers
            output = self.model(**tokenized_reviews, output_hidden_states=True)

        # Return the average embedding of all tokens for each input in the second last hidden layer of the transformer
        # see https://mccormickml.com/2019/05/14/BERT-word-embeddings-tutorial/
        avg_embedding = output.hidden_states[-2].mean(dim=1)

        return avg_embedding.cpu().numpy()
    
class SentenceTransformerEmbeddor(EmbeddorBase):
    """Embedder which uses sentence-transformers to convert reviews to sentence-transformer embeddings."""""
    def __init__(self) -> None:
        super().__init__()
        self.device = _get_torch_device()
        self.model = SentenceTransformer("all-MiniLM-L6-v2").to(self.device)

    @property
    def name(self) -> str:
        return "SentenceTransformerEmbeddor"

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

    Returns:
        torch.device: Device to be used for PyTorch operations.
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")