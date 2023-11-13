"""
Text Embedding Strategies

This module offers classes for converting text to numerical embeddings. It includes 
implementations for various embedding techniques like Count Vectorization, TF-IDF, 
BERT, and Sentence Transformers. An abstract base class defines the common interface, 
while concrete classes provide specific implementations.
"""
from abc import abstractmethod, ABC
from typing import List
import numpy as np
from transformers import BertTokenizer, BertModel
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import torch


class EmbeddorBase(ABC):
    """
    Abstract base class for text embedding strategies.

    Subclasses should implement the transform method to convert a list of text
    documents into numerical embeddings.
    """

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def transform(self, reviews: List[str]) -> np.ndarray:
        pass


class CountEmbeddor(EmbeddorBase):
    """
    Abstract method to transform text documents into embeddings.

    Parameters:
    reviews ([str]): A list of text documents.

    Returns:
    np.ndarray: An array of embeddings.
    """

    def __init__(self) -> None:
        super().__init__()

    def transform(self, reviews: List[str]) -> np.ndarray:
        """
        Transforms text documents into count vector embeddings.

        Parameters:
        reviews ([str]): A list of text documents.

        Returns:
        np.ndarray: A count vectorized representation of the documents.
        """
        vectorizer = CountVectorizer()
        counts = vectorizer.fit_transform(reviews)
        return counts.toarray()


class TfidfEmbeddor(EmbeddorBase):
    """
    Concrete implementation of EmbeddorBase using TF-IDF Vectorization.

    This class converts text documents into a matrix of TF-IDF features.
    """

    def __init__(self) -> None:
        super().__init__()

    def transform(self, reviews: List[str]) -> np.ndarray:
        """
        Transforms text documents into TF-IDF vector embeddings.

        Parameters:
        reviews ([str]): A list of text documents.

        Returns:
        np.ndarray: A TF-IDF vectorized representation of the documents.
        """
        vectorizer = TfidfVectorizer()
        tfidf = vectorizer.fit_transform(reviews)
        return tfidf.toarray()


class BertEmbeddor(EmbeddorBase):
    """
    Concrete implementation of EmbeddorBase using BERT model embeddings.

    Utilizes BERT (Bidirectional Encoder Representations from Transformers) to
    generate embeddings for the given text.
    """

    def __init__(self, device: str = "cpu") -> None:
        """
        Initializes BertEmbeddor with the 'bert-base-uncased' model on the specified device.

        Parameters:
        device (str, optional): The computation device ('cpu' or 'cuda'). Defaults to 'cpu'.
        """
        super().__init__()
        self.device = device
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.model = BertModel.from_pretrained("bert-base-uncased").to(device)
        self.model.eval()

    def transform(self, reviews: List[str]) -> np.ndarray:
        """
        Transforms text documents into BERT embeddings.

        Parameters:
        reviews ([str]): A list of text documents.

        Returns:
        np.ndarray: BERT embeddings of the documents.
        """
        with torch.no_grad():
            encoded_inputs = self.tokenizer(
                reviews, padding=True, truncation=True, return_tensors="pt"
            ).to(self.device)
            outputs = self.model(**encoded_inputs, output_hidden_states = True)
            avg_embeddings = outputs.hidden_states[-2].mean(dim=1)
            return avg_embeddings.detach().cpu().numpy()


class SentenceTransformerEmbeddor(EmbeddorBase):
    """
    Concrete implementation of EmbeddorBase using Sentence Transformers.

    This class uses pre-trained models from the Sentence Transformers library to
    generate embeddings for text.
    """

    def __init__(self, device: str = "cpu") -> None:
        """
        Initializes SentenceTransformerEmbeddor with 'all-MiniLM-L6-v2' model on the specified device.

        Parameters:
        device (str, optional): The computation device ('cpu' or 'cuda'). Defaults to 'cpu'.
        """
        super().__init__()
        self.device = device
        self.model = SentenceTransformer("all-MiniLM-L6-v2").to(device)

    def transform(self, reviews: List[str]) -> np.ndarray:
        """
        Transforms text documents into embeddings using a Sentence Transformer model.

        This method utilizes a pre-trained Sentence Transformer model to convert
        a list of text strings into corresponding numerical embeddings.

        Parameters:
        reviews (List[str]): A list of text documents for embedding.

        Returns:
        np.ndarray: An array of embeddings, where each row corresponds to
                    the embedding of a text document in the input list.
        """
        embeddings = self.model.encode(reviews, convert_to_tensor=True, device=self.device)
        return embeddings.detach().cpu().numpy()
