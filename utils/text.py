import spacy
import pandas as pd
nlp = spacy.load("en_core_web_sm")

def process_text(text):
    # Runs preprocessing and tokenization
    doc = nlp(text)

    # here we can decide what we want to do with the preprocessed text
    lemmas = ' '.join([token.lemma_ for token in doc])
    adjectives = ' '.join([token.text for token in doc if token.pos_.startswith('ADJ')])

    return pd.Series({'lemmas': lemmas, 'adjectives': adjectives})