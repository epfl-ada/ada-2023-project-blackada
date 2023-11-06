import torch
import pandas as pd
from enum import Enum
from transformers import BertTokenizer, BertModel
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

class ModelType(Enum):
    SentenceTransformer = 1
    BERT = 2
    Count = 3
    TFIDF = 4

def embed_inputs(
        input_col: pd.Series,
        model: ModelType = ModelType.Count
    ) -> pd.Series:
    if model == ModelType.Count:
        return to_counts(input_col)
    elif model == ModelType.TFIDF:
        return to_tfidf(input_col)
    elif model == ModelType.BERT:
        return input_col.apply(lambda x: embed_inputs_BERT(x))
    elif model == ModelType.SentenceTransformer:
        return input_col.apply(lambda x: embed_inputs_sentence_transformers(x))
    else:
        raise ValueError("Model type not supported")
    
def to_counts(texts: pd.Series) -> pd.Series:
    """Convert a column of text to a column of counts."""
    vectorizer = CountVectorizer()
    counts = vectorizer.fit_transform(texts)
    return pd.Series(counts.toarray().tolist())

def to_tfidf(texts: pd.Series) -> pd.Series:
    """Convert a column of counts to a column of tfidf."""
    transformer = TfidfVectorizer()
    tfidf = transformer.fit_transform(texts)
    return pd.Series(tfidf.toarray().tolist())

# Embed a sentence
def embed_inputs_BERT(
        input_tokens: tuple[str, list[str]]
    ) -> torch.Tensor:
    # Load pre-trained model & tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained("bert-base-uncased")
    model.eval()
    
    encoded_input = tokenizer(input_tokens, padding=True, return_tensors='pt')
    output = model(**encoded_input, output_hidden_states=True)

    # Return the average embedding of all tokens for each input in the second last hidden layer of the transformer
    # see https://mccormickml.com/2019/05/14/BERT-word-embeddings-tutorial/
    avg_embedding = output.hidden_states[-2].mean(dim=1)
    return avg_embedding.squeeze().tolist()

def embed_inputs_sentence_transformers(
        input_tokens: tuple[str, list[str]]
    ) -> torch.Tensor:
    model = SentenceTransformer('all-MiniLM-L6-v2')
    outputs = model.encode(input_tokens, convert_to_tensor=True)
    return outputs.squeeze().tolist()