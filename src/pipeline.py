"""
Text Analysis Workflow Module

This module introduces the TextAnalysis class, which provides a streamlined workflow for 
processing text documents. It integrates feature extraction, embedding, and metric computation 
in a cohesive manner. The class is designed to be flexible, accepting any implementations 
of extractor, embedder, and metric calculator that conform to a standard interface.
"""

import numpy as np
from typing import List, Any


class TextAnalysis:
    """
    A class that integrates text extraction, embedding, and metric computation.

    This class is designed to work with any extractor, embedder, and metric
    computation method, as long as they adhere to a standard interface.
    """

    def __init__(self, extractor: Any, embedder: Any, metric: Any) -> None:
        """
        Initializes the TextAnalysis with specified extractor, embedder, and metric.

        Parameters:
        extractor (Any): An object that implements a transform method for text extraction.
        embedder (Any): An object that implements a transform method for embedding the extracted text.
        metric (Any): An object that implements a transform method for computing a metric on the embeddings.
        """
        self.extractor = extractor
        self.embedder = embedder
        self.metric = metric

    def transform(self, docs: List[str]) -> np.ndarray:
        """
        Processes the given documents through extraction, embedding, and metric computation.

        Parameters:
        docs (List[str]): A list of documents to be processed.

        Returns:
        np.ndarray: The resulting similarity scores or metrics after processing the documents.
        """
        # Extraction
        extracted_texts = self.extractor.transform(docs)
        # Embedding
        embeddings = self.embedder.transform(extracted_texts)
        # Similarity Calculation
        self.similarity = self.metric.transform(embeddings)
        return self.similarity
