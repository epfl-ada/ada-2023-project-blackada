from abc import abstractmethod, ABC
from tqdm import tqdm

from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity

import numpy as np

import scipy
from scipy.special import rel_entr
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
    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Abstract method to transform embeddings into a consensus score.

        Args:
            embeddings (np.ndarray): A numpy array of embeddings.

        Returns:
            consensus_matrix (np.ndarray): A numpy array of consensus scores.
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

        Args:
            embeddings (np.ndarray): A numpy array of embeddings.

        Returns:
            consensus_matrix (np.ndarray): A numpy array of cosine similarities.
        """
        return cosine_similarity(embeddings)


class AverageCosineSimilarity(ConsensusBase):
    """
    Concrete implementation of ConsensusBase using average cosine similarity.

    This class calculates the consensus as the average cosine similarity
    between all pairs of embeddings.
    """

    def __init__(self) -> None:
        super().__init__()

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Calculates the cosine similarity between all pairs of embeddings.

        Args:
            embeddings (np.ndarray): A numpy array of embeddings.

        Returns:
            consensus_matrix (np.ndarray): A numpy array of cosine similarities.
        """
        n = embeddings.shape[0]
        total_similarity = 0.0
        for i in tqdm(range(n), desc="Calculating average cosine similarity", total=n):
            similarities = cosine_similarity(embeddings[i, :], embeddings)[0]
            total_similarity += (np.sum(similarities) - 1.0) / (n - 1)
        return total_similarity / n  # type: ignore


class KullbackLeiblerDivergence(ConsensusBase):
    """
    Concrete implementation of ConsensusBase using Kullback-Leibler divergence.

    This class calculates the consensus as the average Kullback-Leibler divergence
    between all pairs of embeddings.
    """

    def __init__(self, epsilon=1e-10, n_jobs=-1) -> None:
        super().__init__()
        self.epsilon = epsilon
        self.n_jobs = n_jobs

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Calculates the Kullback-Leibler divergence between all pairs of embeddings.

        Args:
            embeddings (np.ndarray): A numpy array of embeddings.

        Returns:
            consensus_matrix (np.ndarray): A numpy array of Kullback-Leibler divergences.
        """
        normalized_embeddings = self._normalize_embeddings(embeddings)

        divergence_matrix = pairwise_distances(
            normalized_embeddings,
            metric=self._kullback_leibler_divergence,
            n_jobs=self.n_jobs,
        )
        return divergence_matrix

    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Normalizes embeddings by adding epsilon and renormalizing.

        Args:
            embeddings: A numpy array of embeddings.

        Returns:
            A numpy array of normalized embeddings.
        """
        # Add epsilon and renormalize embeddings
        # if sparse matrix, convert to dense
        if isinstance(embeddings, scipy.sparse.csr_matrix):
            embeddings = embeddings.toarray()

        embeddings += self.epsilon
        return embeddings / np.sum(embeddings, axis=1, keepdims=True)

    def _kullback_leibler_divergence(self, p: np.ndarray, q: np.ndarray) -> np.ndarray:
        """
        Calculates the Kullback-Leibler divergence between two embeddings.

        Args:
            p: A numpy array of embeddings.
            q: A numpy array of embeddings.

        Returns:
            A numpy array of Kullback-Leibler divergences.
        """
        return rel_entr(p, q).sum()


class JensenShannonDivergence(KullbackLeiblerDivergence):
    """
    Concrete implementation of ConsensusBase using Jensen-Shannon divergence.

    This class calculates the consensus as the average Jensen-Shannon divergence
    between all pairs of embeddings.
    """

    def __init__(self, epsilon=1e-10, n_jobs=-1) -> None:
        super().__init__(epsilon=epsilon, n_jobs=n_jobs)

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Calculates the Jensen-Shannon divergence between all pairs of embeddings.

        Args:
            embeddings (np.ndarray): A numpy array of embeddings.

        Returns:
            consensus_matrix (np.ndarray): A numpy array of Jensen-Shannon divergences.
        """
        normalized_embeddings = self._normalize_embeddings(embeddings)
        divergence_matrix = pairwise_distances(
            normalized_embeddings, metric=jensenshannon, n_jobs=self.n_jobs
        )
        return divergence_matrix


class Correlation(ConsensusBase):
    """
    Concrete implementation of ConsensusBase using Pearson correlation.

    This class calculates the consensus as the average Pearson correlation
    between all pairs of embeddings.
    """

    def __init__(self, epsilon=1e-10, n_jobs=-1) -> None:
        super().__init__()
        self.epsilon = epsilon
        self.n_jobs = n_jobs

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Calculates the Pearson correlation between all pairs of embeddings.

        Args:
            embeddings (np.ndarray): A numpy array of embeddings.

        Returns:
            consensus_matrix (np.ndarray): A numpy array of Pearson correlations.
        """
        # if sparse
        if isinstance(embeddings, scipy.sparse.csr_matrix):
            embeddings = embeddings.toarray()
        correlation_matrix = pairwise_distances(
            embeddings, metric="correlation", n_jobs=self.n_jobs
        )
        return correlation_matrix
