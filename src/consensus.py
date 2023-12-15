"""
Module for computing consensus scores from embeddings.

Provides an abstract base class, ConsensusBase, for defining consensus calculation methods,
and a concrete implementation, CosineSimilarityConsensus, using cosine similarity.
"""

from abc import abstractmethod, ABC
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd


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

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Calculates the cosine similarity between all pairs of embeddings.

        Parameters:
            embeddings (np.ndarray): A numpy array of embeddings.

        Returns:
            consensus_matrix (np.ndarray): A numpy array of cosine similarities.
        """
        return cosine_similarity(embeddings)


class ConsensusLevel:
    """
    Helper class for easily retrieving consensus distributions within
    groups at some level of aggregation, e.g. retrieve consensus across
    IPAs, Pale Ales, ... for aggregation level "Beer Style".
    """

    def __init__(
        self, aggregator: str | None, consensus: np.ndarray, reviews: pd.DataFrame
    ) -> None:
        self.aggregator = aggregator
        self.consensus_matrix = consensus
        self.reviews = reviews

        self.groups = self._get_groups()

    def get_consensus_matrix(self, group: str) -> np.ndarray:
        assert group in self.groups, f"{group} is not a valid group."

        indices = self._get_group_indices(group)
        member_similarities = self._subset_matrix(self.consensus_matrix, indices)

        return member_similarities

    def get_consensus_dist(self, group: str) -> float:
        assert group in self.groups, f"{group} is not a valid group."

        group_consensus_matrix = self.get_consensus_matrix(group)
        group_consensus_dist = self._get_upper_triangle(group_consensus_matrix)
        return group_consensus_dist

    def _get_groups(self) -> list[str]:
        if self.aggregator is None:
            return ["All"]
        return self.reviews[self.aggregator].unique().tolist()

    def _get_group_indices(self, group: str) -> list[int]:
        if self.aggregator is None:
            return self.reviews.index.tolist()
        is_group = self.reviews[self.aggregator] == group
        return self.reviews[is_group].index.tolist()

    def _subset_matrix(self, matrix: np.ndarray, indices: list[int]) -> np.ndarray:
        return matrix[indices][:, indices]

    def _get_upper_triangle(self, matrix: np.ndarray) -> np.ndarray:
        indices = np.triu_indices_from(matrix, k=1)
        return matrix[indices]

    def __str__(self) -> str:
        return f"ConsensusInGroup(group={self.group})"
