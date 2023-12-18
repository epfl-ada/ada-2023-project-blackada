from src.consensus import ConsensusBase


import numpy as np
import pandas as pd
from tqdm import tqdm


from typing import Optional


class EmbeddingAggregator:
    """
    This class aggregates TF-IDF embeddings at a specified level of detail using review data.
    It supports operations such as computing group-wise and overall average embeddings and consensus matrices.
    """

    def __init__(
        self,
        embeddings: np.ndarray,
        reviews: pd.DataFrame,
        consensus: ConsensusBase,
        aggregator_column: Optional[tuple[str, str]] = None,
        sub_groups: Optional[list[str]] = None,
    ) -> None:
        """
        Initializes the EmbeddingAggregator with data and configuration.

        :param embeddings: A numpy array of embeddings.
        :param reviews: A pandas DataFrame containing review data.
        :param consensus: An instance of a consensus algorithm class.
        :param aggregator_column: The column name to be used for grouping the data.
        :param sub_groups: A list of specific subgroups to consider within the aggregator column.
        """
        self.embeddings = embeddings
        self.reviews = reviews
        self.consensus = consensus
        self.aggregator_column = aggregator_column
        self.sub_groups = sub_groups if sub_groups else []
        self.groups = self._identify_groups()

    def _identify_groups(self) -> list[str]:
        """
        Identifies groups based on the aggregator column and subgroups, if provided.

        :return: A list of groups.
        """
        if not self.aggregator_column:
            return ["All"]
        all_groups = self.reviews[self.aggregator_column].unique().tolist()
        return [
            group
            for group in all_groups
            if not self.sub_groups or group in self.sub_groups
        ]

    def get_embeddings_by_group(self, group: str) -> np.ndarray:
        """
        Retrieves embeddings for a specific group.

        :param group: The group name.
        :return: A numpy array of embeddings for the specified group.
        """
        if not self.aggregator_column or group == "All":
            return self.embeddings
        group_filter = self.reviews[self.aggregator_column] == group
        return self.embeddings[group_filter]

    def get_average_embeddings(self) -> dict[str, np.ndarray]:
        """
        Calculates the average embeddings for each group.

        :return: A dictionary with group names as keys and their average embeddings as values.
        """
        return {
            group: self.get_embeddings_by_group(group).mean(axis=0)
            for group in tqdm(
                self.groups,
                desc="Calculating average embeddings",
                total=len(self.groups),
            )
        }

    def get_consensus_matrix(self, group: str, max_samples: int = 10000) -> np.ndarray:
        """
        Generates a consensus matrix for a given group.

        :param group: The group name.
        :param max_samples: Maximum number of samples to consider for the consensus matrix.
        :return: A consensus matrix as a numpy array.
        """
        embeddings = self.get_embeddings_by_group(group)
        sample_size = min(embeddings.shape[0], max_samples)
        sampled_embeddings = embeddings[
            np.random.choice(embeddings.shape[0], sample_size, replace=False)
        ]
        return self.consensus.transform(sampled_embeddings)

    def get_consensus_distribution(
        self, group: str, upper: bool = True, max_samples: int = 10000
    ) -> np.ndarray:
        """
        Computes the consensus distribution for a given group.

        :param group: The group name.
        :param upper: If True, uses the upper triangle of the matrix; otherwise, the lower triangle.
        :param max_samples: Maximum number of samples to consider.
        :return: A numpy array representing the consensus distribution.
        """
        consensus_matrix = self.get_consensus_matrix(group, max_samples)
        indices = (
            np.triu_indices_from(consensus_matrix, k=1)
            if upper
            else np.tril_indices_from(consensus_matrix, k=-1)
        )
        return consensus_matrix[indices]

    def get_overall_average_embedding(self) -> np.ndarray:
        """
        Calculates the average embedding across all groups or the entire dataset if no groups are defined.

        :return: The overall average embedding as a numpy array.
        """
        group_embeddings = self.get_average_embeddings().values()
        return np.vstack(list(group_embeddings))

    def get_overall_consensus_matrix(self) -> np.ndarray:
        """
        Calculates the consensus matrix for the entire dataset or across all defined groups.

        :return: The overall consensus matrix as a numpy array.
        """
        overall_embeddings = self.get_overall_average_embedding()
        overall_embeddings = np.asarray(overall_embeddings)
        return self.consensus.transform(overall_embeddings)

    def get_overall_consensus_distribution(self, upper: bool = True) -> np.ndarray:
        """
        Computes the overall consensus distribution for the dataset.

        :param upper: If True, uses the upper triangle of the matrix; otherwise, the lower triangle.
        :return: A numpy array representing the overall consensus distribution.
        """
        consensus_matrix = self.get_overall_consensus_matrix()
        indices = (
            np.triu_indices_from(consensus_matrix, k=1)
            if upper
            else np.tril_indices_from(consensus_matrix, k=-1)
        )
        return consensus_matrix[indices]