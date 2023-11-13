from abc import abstractmethod
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class ConsensusBase:
    def __init__(self) -> None:
        pass

    @abstractmethod
    def transform(self, embeddings: np.ndarray) -> float:
        pass


class CosineSimilarity(ConsensusBase):
    def __init__(self)-> None:
        super().__init__()

    def transform(self, embeddings: np.ndarray) -> float:
        sim_matrix = cosine_similarity(embeddings)
        upper_triangle_indices = np.triu_indices_from(sim_matrix, k=1)
        average_similarity = np.mean(sim_matrix[upper_triangle_indices])
        return average_similarity