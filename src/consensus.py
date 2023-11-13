"""
Module for computing consensus scores from embeddings.

Provides an abstract base class, ConsensusBase, for defining consensus calculation methods,
and a concrete implementation, CosineSimilarityConsensus, using cosine similarity.
"""

from abc import abstractmethod, ABC
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class ConsensusBase(ABC):
    """
    Abstract base class for computing consensus from embeddings.

    This class serves as a foundation for different consensus strategies.
    Subclasses should implement the transform method.
    """

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def transform(self, embeddings: np.ndarray) -> float:
        """
        Abstract method to transform embeddings into a consensus score.

        Parameters:
        embeddings (np.ndarray): A numpy array of embeddings.

        Returns:
        float: The calculated consensus score.
        """
        pass


class CosineSimilarity(ConsensusBase):
    """
    Concrete implementation of ConsensusBase using cosine similarity.

    This class calculates the consensus as the average cosine similarity
    between all pairs of embeddings.
    """

    def __init__(self) -> None:
        super().__init__()

    def transform(self, embeddings: np.ndarray) -> float:
        """
        Calculates the average cosine similarity from the given embeddings.

        Parameters:
        embeddings (np.ndarray): A numpy array of embeddings.

        Returns:
        float: The average cosine similarity score.
        """
        sim_matrix = cosine_similarity(embeddings)
        upper_triangle_indices = np.triu_indices_from(sim_matrix, k=1)
        average_similarity = np.mean(sim_matrix[upper_triangle_indices])
        return average_similarity
