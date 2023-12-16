"""
Module for computing consensus scores from embeddings.

Provides an abstract base class, ConsensusBase, for defining consensus calculation methods,
and a concrete implementation, CosineSimilarityConsensus, using cosine similarity.
"""

from abc import abstractmethod, ABC
from sklearn.metrics.pairwise import cosine_similarity, paired_distances
import numpy as np
import pandas as pd
from scipy.special import rel_entr
from scipy.spatial.distance import pdist, squareform
from scipy.spatial.distance import jensenshannon


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


class KullbackLeiblerDivergence(ConsensusBase):
    """
    Concrete implementation of ConsensusBase using Kullback-Leibler divergence.

    This class calculates the consensus as the average Kullback-Leibler divergence
    between all pairs of embeddings.
    """

    def __init__(self, epsilon=1e-10) -> None:
        super().__init__()
        self.epsilon = epsilon

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Calculates the Kullback-Leibler divergence between all pairs of embeddings.

        Parameters:
            embeddings (np.ndarray): A numpy array of embeddings.

        Returns:
            consensus_matrix (np.ndarray): A numpy array of Kullback-Leibler divergences.
        """
        normalized_embeddings = self._normalize_embeddings(embeddings)

        divergence_matrix = squareform(
            pdist(normalized_embeddings, metric=self._kullback_leibler_divergence)
        )
        return divergence_matrix

    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        # Add epsilon and renormalize embeddings
        embeddings += self.epsilon
        return embeddings / embeddings.sum(axis=1, keepdims=True)

    def _kullback_leibler_divergence(self, p: np.ndarray, q: np.ndarray) -> np.ndarray:
        # Compute KL divergence in a vectorized manner
        return np.sqrt(rel_entr(p,q).sum())


class JensenShannonDivergence(KullbackLeiblerDivergence):
    """
    Concrete implementation of ConsensusBase using Jensen-Shannon divergence.

    This class calculates the consensus as the average Jensen-Shannon divergence
    between all pairs of embeddings.
    """

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Calculates the Jensen-Shannon divergence between all pairs of embeddings.

        Parameters:
            embeddings (np.ndarray): A numpy array of embeddings.

        Returns:
            consensus_matrix (np.ndarray): A numpy array of Jensen-Shannon divergences.
        """
        normalized_embeddings = self._normalize_embeddings(embeddings)
        divergence_matrix = squareform(pdist(normalized_embeddings, 'jensenshannon'))
        return divergence_matrix



class Correlation(ConsensusBase):
    """
    Concrete implementation of ConsensusBase using Pearson correlation.

    This class calculates the consensus as the average Pearson correlation
    between all pairs of embeddings.
    """

    def __init__(self) -> None:
        super().__init__()

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Calculates the Pearson correlation between all pairs of embeddings.

        Parameters:
            embeddings (np.ndarray): A numpy array of embeddings.

        Returns:
            consensus_matrix (np.ndarray): A numpy array of Pearson correlations.
        """
        normalized_embeddings = (
            embeddings - embeddings.mean(axis=1, keepdims=True)
        ) / embeddings.std(axis=1, keepdims=True)
        return np.corrcoef(normalized_embeddings)


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
