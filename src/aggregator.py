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

        Args:
            embeddings: A numpy array of TF-IDF embeddings.
            reviews: A dataframe of review data.
            consensus: A consensus object.
            aggregator_column: A tuple of column name and column value to aggregate by.
            sub_groups: A list of subgroups to consider.
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

        Returns:
            A list of group names.
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

        Args:
            group: The group name.

        Returns:
            A numpy array of embeddings for the group.
        """

        if not self.aggregator_column or group == "All":
            return self.embeddings

        group_filter = self.reviews[self.aggregator_column] == group

        return self.embeddings[group_filter]

    def get_average_embeddings(self) -> dict[str, np.ndarray]:
        """
        Calculates the average embeddings for each group.

        Returns:
            A dictionary of group names and average embeddings.
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

        Args:
            group: The group name.
            max_samples: Maximum number of samples to consider.

        Returns:
            A numpy array representing the consensus matrix.
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

        Args:
            group: The group name.
            upper: If True, uses the upper triangle of the matrix; otherwise, the lower triangle.
            max_samples: Maximum number of samples to consider.

        Returns:
            A numpy array representing the consensus distribution.
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

        Returns:
            A numpy array representing the average embedding.
        """
        group_embeddings = self.get_average_embeddings().values()
        return np.vstack(list(group_embeddings))

    def get_overall_consensus_matrix(self) -> np.ndarray:
        """
        Calculates the consensus matrix for the entire dataset or across all defined groups.

        Returns:
            A numpy array representing the consensus matrix.
        """
        overall_embeddings = self.get_overall_average_embedding()
        overall_embeddings = np.asarray(overall_embeddings)
        return self.consensus.transform(overall_embeddings)

    def get_overall_consensus_distribution(self, upper: bool = True) -> np.ndarray:
        """
        Computes the overall consensus distribution for the dataset.

        Args:
            upper: If True, uses the upper triangle of the matrix; otherwise, the lower triangle.

        Returns:
            A numpy array representing the consensus distribution.
        """
        consensus_matrix = self.get_overall_consensus_matrix()
        indices = (
            np.triu_indices_from(consensus_matrix, k=1)
            if upper
            else np.tril_indices_from(consensus_matrix, k=-1)
        )
        return consensus_matrix[indices]
