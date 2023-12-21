from abc import abstractmethod, ABC
from typing import Optional

import numpy as np
import torch

from transformers import BertTokenizer, BertModel
from sentence_transformers import SentenceTransformer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from src.utils import get_torch_device


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

        Args:
            reviews (list[str]): List of reviews to be embedded.

        Returns:
            np.ndarray: Array of embeddings.
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
        Embed a list of reviews using count vectorization.

        Args:
            reviews (list[str]): List of reviews to be converted to counts.

        Returns:
            np.ndarray: Array of counts.
        """
        return self.vectorizer.fit_transform(reviews).toarray()


class TfidfEmbedder(EmbedderBase):
    """Embedder which uses a Tfidf vectorizer to transform reviews to Tfidf representations."""

    def __init__(self, sparse_output: Optional[bool] = True) -> None:
        super().__init__()
        self.transformer = TfidfVectorizer()
        self.sparse_output = sparse_output

    @property
    def name(self) -> str:
        return "TfidfEmbedder"

    def transform(self, reviews: list[str]) -> np.ndarray:
        """
        Embed a list of reviews using Tfidf vectorization.

        Args:
            reviews (list[str]): List of reviews to be converted to Tfidf representations.

        Returns:
            np.ndarray: Array of Tfidf representations.
        """
        return (
            self.transformer.fit_transform(reviews).toarray()
            if not self.sparse_output
            else self.transformer.fit_transform(reviews)
        )

    def get_feature_names(self) -> list[str]:
        """
        Get the feature names of the Tfidf vectorizer.

        Returns:
            list[str]: List of feature names.
        """
        return self.transformer.get_feature_names_out()


class BertEmbedder(EmbedderBase):
    """Embedder which uses Bert to convert reviews to Bert embeddings."""

    def __init__(self) -> None:
        super().__init__()

        # Get GPU if available
        self.device = get_torch_device()

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

        Args:
            reviews (list[str]): List of reviews to be converted to Bert embeddings.

        Returns:
            np.ndarray: Array of Bert embeddings.
        """

        # Convert to list if not already (e.g. for Pandas Series or one review is passed)
        reviews = list(reviews)

        # Get tokenized input
        tokenized_reviews = self.tokenizer(
            reviews, padding=True, truncation=True, return_tensors="pt"
        ).to(self.device)

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
        self.device = get_torch_device()
        self.model = SentenceTransformer("all-MiniLM-L6-v2").to(self.device)

    @property
    def name(self) -> str:
        return "SentenceTransformerEmbedder"

    def transform(self, reviews: list[str]) -> np.ndarray:
        """
        Embed a list of reviews using sentence-transformers.

        Args:
            reviews (list[str]): List of reviews to be converted to sentence-transformer embeddings.

        Returns:
            np.ndarray: Array of sentence-transformer embeddings.
        """
        return self.model.encode(reviews, convert_to_numpy=True)
