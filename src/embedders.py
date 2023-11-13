from abc import abstractmethod
import numpy as np
from transformers import BertTokenizer, BertModel
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


class EmbeddorBase:
    def __init__(self) -> None:
        pass

    @abstractmethod
    def transform(self, reviews: [str]) -> np.ndarray:
        pass


class CountEmbeddor(EmbeddorBase):
    def transform(self, reviews: [str]) -> np.ndarray:
        vectorizer = CountVectorizer()
        counts = vectorizer.fit_transform(reviews)
        return counts.toarray()


class TfidfEmbeddor(EmbeddorBase):
    def transform(self, reviews: [str]) -> np.ndarray:
        vectorizer = TfidfVectorizer()
        tfidf = vectorizer.fit_transform(reviews)
        return tfidf.toarray()


class BertEmbeddor(EmbeddorBase):
    def __init__(self) -> None:
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.model = BertModel.from_pretrained("bert-base-uncased")
        self.model.eval()

    def transform(self, reviews: [str]) -> np.ndarray:
        encoded_inputs = self.tokenizer(
            reviews, padding=True, truncation=True, return_tensors="pt"
        )
        outputs = self.model(**encoded_inputs, output_hidden_states=True)
        avg_embeddings = outputs.hidden_states[-2].mean(dim=1)
        return avg_embeddings.detach().numpy()


class SentenceTransformerEmbeddor(EmbeddorBase):
    def __init__(self) -> None:
        super().__init__()
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def transform(self, reviews: [str]) -> np.ndarray:
        embeddings = self.model.encode(reviews, convert_to_tensor=True)
        return embeddings.detach().numpy()
