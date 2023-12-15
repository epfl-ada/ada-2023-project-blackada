"""
Module containing utility files used throughout the project.

Included functions:
- download_data(): Downloads and extracts the zipped `BeerAdvocate` data from a Google Drive URL
- load_data(): Loads the extracted beer data from the specified data directory
"""

import gzip
import os
import shutil
import subprocess
import tarfile
from collections import Counter

import gdown
import numpy as np
import pandas as pd
import spacy
from numpy.linalg import norm
from spacy.language import Language
from spacy.tokens import DocBin
from tqdm import tqdm


def download_data(url: str, data_dir: str) -> None:
    """
    Downloads and extracts the zipped `BeerAdvocate` data from a Google Drive URL
    into the specified data directory and deletes all unused files as well as the
    extracted .tar files.

    Args:
        url (str): Download data from URL.
        data_dir (str): Path of directory to store downloaded data.

    Returns:
        None
    """
    # Download zipped data from Google Drive
    zipped_data = os.path.join(data_dir, "beer_advocate.tar.gz")
    gdown.download(url, zipped_data)

    # Extract zipped data and remove zipped file
    tarfile.open(zipped_data, "r:gz").extractall(data_dir)
    os.remove(zipped_data)

    # Extract the reviews .tar files and remove them
    review_path = os.path.join(data_dir, "reviews.txt.gz")
    _extract_gz(review_path)

    # Remove unused files
    os.remove(os.path.join(data_dir, "ratings.txt.gz"))
    os.remove(os.path.join(data_dir, "reviews.txt.gz"))


def raw_data_exists(data_dir: str) -> bool:
    """
    Check if the raw data exists in the specified data directory.

    Args:
        data_dir (str): Path of directory containing raw data.

    Returns:
        bool: True if raw data exists, else False.
    """
    files = ["reviews.txt", "beers.csv", "breweries.csv", "users.csv"]
    has_path = os.path.isdir(data_dir)
    has_files = all([os.path.isfile(os.path.join(data_dir, file)) for file in files])

    return has_path and has_files


def processed_data_exists(processed_dir: str, num_samples: int | None = None) -> bool:
    """
    Check if the processed data exists in the specified data directory.

    Args:
        processed_dir (str): Path of directory containing raw data.

    Returns:
        bool: True if processed data exists, else False.
    """
    files = ["reviews.feather", "docs.spacy"]
    if num_samples:
        processed_dir = os.path.join(processed_dir, str(num_samples))

    has_path = os.path.isdir(processed_dir)
    has_files = all(
        [os.path.isfile(os.path.join(processed_dir, file)) for file in files]
    )

    return has_path and has_files


def process_data(
    data_dir: str,
    processed_dir: str,
    nlp: Language,
    doc_bin: DocBin,
    num_samples: int | None = None,
) -> None:
    """
    Processes the raw data in the specified data directory. It loads the raw
    reviews and metainfo, merges them, preprocesses them, extracts SpaCy docs
    and saves the processed data as a feather file into the preprocessed
    folder. If number of samples is specified, it only processes a subset
    of the data and saves it into a folder `{preprocessed_dir}/{num_samples}/`

    Uses a SpaCy pipeline to process the reviews defined by the `nlp` object
    and saves the SpaCy docs into a `docs.spacy` file using a the `doc_bin`
    object.

    Args:
        data_dir (str): Path of directory containing raw data.
        processed_dir (str): Path of directory to store processed data.
        nlp (Language): SpaCy language model.
        doc_bin (DocBin): SpaCy DocBin.
        num_samples (int, optional): Subset of data to process. Defaults to processing all samples. (None)

    Returns:
        None
    """
    # Check that raw data exists
    msg = "Raw data does not exist. Call `download_data()` first."
    assert raw_data_exists(data_dir), msg

    # Adjust processed directory if num_samples is specified
    if num_samples:
        processed_dir = os.path.join(processed_dir, str(num_samples))

    # Create processed directory if it does not exist
    os.makedirs(processed_dir, exist_ok=True)

    # Load raw reviews
    reviews = _load_reviews(data_dir)

    # Load metainfo for reviews
    print("Loading metadata...")
    beers, breweries, users = _load_metainfo(data_dir)

    # Merging reviews with metainfo
    print("Merging reviews...")
    reviews = reviews.merge(beers, on="beer_id", how="left")
    reviews = reviews.merge(breweries, on="brewery_id", how="left")
    reviews = reviews.merge(users, on="user_id", how="left")

    # Process reviews
    print("Preprocessing reviews...")
    reviews = _preprocess_reviews(reviews)

    # Limit number of reviews if specified
    if num_samples:
        reviews = reviews.iloc[:num_samples]

    # Save processed reviews as feather
    reviews.to_feather(os.path.join(processed_dir, "reviews.feather"))

    # Extract SpaCy docs with relevant information
    review_texts = reviews.review.text.tolist()
    for doc in tqdm(
        nlp.pipe(review_texts), total=len(review_texts), desc="Processing Spacy docs"
    ):
        doc_bin.add(doc)

    # Save processed reviews as Spacy DocBin
    doc_bin.to_disk(os.path.join(processed_dir, "docs.spacy"))


def load_data(
    processed_dir: str,
    nlp: Language,
    num_samples: int | None = None,
) -> pd.DataFrame:
    """
    Loads the extracted beer data from the specified data directory. It looks for
    a `reviews.feather` and `docs.spacy` file containing the extracted SpaCy object.
    If a limit is specified, it looks in the folder `data_dir/{limit}` for the
    same files which contain a subset of the data.

    Args:
        processed_dir (str): Path of directory containing extracted data.
        num_samples (int, optional): Subset of data to load. Defaults to loading all samples. (None)

    Returns:
        pd.DataFrame: DataFrame containing relevant beer review data and SpaCy docs.
    """
    # Check that processed data exists
    assert processed_data_exists(
        processed_dir, num_samples
    ), "Processed data does not exist. Call `process_data()` first."

    # Adjust processed directory if num_samples is specified
    if num_samples:
        processed_dir = os.path.join(processed_dir, str(num_samples))

    # Load reviews from feather file
    reviews_path = os.path.join(processed_dir, "reviews.feather")
    reviews = pd.read_feather(reviews_path)

    # Laod SpaCy docs from spacy file
    docs_path = os.path.join(processed_dir, "docs.spacy")
    docs = DocBin().from_disk(docs_path).get_docs(nlp.vocab)

    # Add SpaCy docs to reviews DataFrame
    reviews[("review", "doc")] = list(docs)

    return reviews


def _load_reviews(data_dir: str) -> pd.DataFrame:
    """
    Loads the reviews from a `reviews.feather` is it exists. Else,
    it loads the raw reviews from `reviews.txt` and saves the
    processed reviews as a `reviews.feather` file.

    Args:
        data_dir (str): Directory containing raw reviews

    Returns:
        pd.DataFrame: DataFrame containing reviews.

    Notes:
    Note that we ensure correct casting of values before saving.
    However, values of type str are casted to object type since
    string can have different lengths.
    """
    feather_reviews_path = os.path.join(data_dir, "reviews.feather")
    if os.path.isfile(feather_reviews_path):
        print("Loading raw reviews from feather...")
        return pd.read_feather(feather_reviews_path)

    beer_data = []
    current_beer = {}
    file_path = os.path.join(data_dir, "reviews.txt")
    result = subprocess.run(["wc", "-l", file_path], stdout=subprocess.PIPE, text=True)
    line_count = int(result.stdout.strip().split()[0])

    with open(file_path, "r") as file:
        for line in tqdm(file, total=line_count, desc="Loading raw reviews"):
            line = line.strip()
            if not line or ": " not in line:
                if current_beer:
                    beer_data.append(current_beer)
                    current_beer = {}
            else:
                key, value = line.split(": ", 1) if ": " in line else (None, None)
                if key is not None:
                    # Cast values to correct types before saving
                    if key == "date":
                        value = pd.to_datetime(int(value), unit="s")
                    elif value in {"True", "False"}:
                        value = value == "True"
                    elif key in ["beer_id", "brewery_id"]:
                        value = int(value)
                    elif key in [
                        "abv",
                        "appearance",
                        "aroma",
                        "palate",
                        "taste",
                        "overall",
                        "rating",
                    ]:
                        value = float(value)
                    current_beer[key] = value
                else:
                    continue

        if current_beer:
            beer_data.append(current_beer)

    reviews = pd.DataFrame(beer_data)

    # Save reviews as feather file
    reviews.to_feather(feather_reviews_path)

    return reviews


def _load_metainfo(data_dir: str) -> pd.DataFrame:
    """
    Loads the metainfo from the data directory. And performs basic preprocessing:
    - Selects relevant columns
    - Renames columns so that they can be merged with the reviews DataFrame
    - Converts timestamps to datetime

    Args:
        data_dir (str): Path of directory containing extracted data.

    Returns:
        beers (pd.DataFrame): DataFrame containing beer metainfo.
        breweries (pd.DataFrame): DataFrame containing brewery metainfo.
        users (pd.DataFrame): DataFrame containing user metainfo.
    """

    # Load the raw data
    beers = pd.read_csv(os.path.join(data_dir, "beers.csv"))
    breweries = pd.read_csv(os.path.join(data_dir, "breweries.csv"))
    users = pd.read_csv(os.path.join(data_dir, "users.csv"))

    # Define relevant columns
    beers = beers[["beer_id", "nbr_ratings", "nbr_reviews"]].rename(
        {"nbr_ratings": "beer_nbr_ratings", "nbr_reviews": "beer_nbr_reviews"}, axis=1
    )
    breweries = breweries[["id", "location", "nbr_beers"]].rename(
        {"id": "brewery_id", "location": "brewery_location"}, axis=1
    )
    users = users[
        ["nbr_ratings", "nbr_reviews", "user_id", "joined", "location"]
    ].rename(
        {
            "nbr_ratings": "user_nbr_ratings",
            "nbr_reviews": "user_nbr_reviews",
            "location": "user_location",
        },
        axis=1,
    )

    # Convert timestamps to datetime
    users["user_joined"] = pd.to_datetime(users["joined"], unit="s")
    users = users.drop(columns=["joined"])

    return beers, breweries, users


def _preprocess_reviews(reviews: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses the reviews DataFrame. Currently it:
    - Sorts columns
    - Converts columns into multi-index columns
    - Drops any reviews with missing values
    - Resets the index

    Args:
        df (pd.DataFrame): `reviews` DataFrame containing reviews.

    Returns:
        pd.DataFrame: Preprocessed `reviews` DataFrame.
    """
    # Sorted columns and multi-index columns
    columns = [
        "beer_id",
        "beer_name",
        "style",
        "abv",
        "beer_nbr_ratings",
        "beer_nbr_reviews",
        "brewery_id",
        "brewery_name",
        "brewery_location",
        "nbr_beers",
        "user_id",
        "user_name",
        "user_nbr_ratings",
        "user_nbr_reviews",
        "user_joined",
        "user_location",
        "appearance",
        "aroma",
        "palate",
        "taste",
        "overall",
        "rating",
        "text",
        "date",
    ]
    multi_columns = {
        "beer": ["id", "name", "style", "abv", "nbr_ratings", "nbr_reviews"],
        "brewery": ["id", "name", "location", "nbr_beers"],
        "user": ["id", "name", "nbr_ratings", "nbr_reviews", "joined", "location"],
        "review": [
            "appearance",
            "aroma",
            "palate",
            "taste",
            "overall",
            "rating",
            "text",
            "date",
        ],
    }

    # Create multi-index
    multi_columns_tuples = [(k, v) for k, vs in multi_columns.items() for v in vs]
    multi_index = pd.MultiIndex.from_tuples(multi_columns_tuples)

    # Sort columns and create multi-index columns
    reviews = reviews[columns]
    reviews.columns = multi_index

    # Drop any reviews (rows) with missing values
    reviews = reviews.dropna()

    # Reset index
    reviews = reviews.reset_index()

    return reviews


def _extract_gz(file_path: str) -> None:
    """
    Extracts a .gz file into the same directory and removes the .gz file.

    Args:
        file_path (str): Path of .gz file to extract.

    Returns:
        None
    """
    with gzip.open(file_path, "rb") as f_in:
        with open(file_path[:-3], "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)


def cosine_similarity(a, b):
    """
    Calculates the cosine similarity between two vectors.

    Args:
        a (np.ndarray): Vector a.
        b (np.ndarray): Vector b.

    Returns:
        float: Cosine similarity between a and b.
    """
    return a @ b / (norm(a) * norm(b))


def get_word_frequency(text_list):
    """Calculate word frequency given a list of text documents"""
    # Combine all the text documents into one large string
    combined_text = " ".join(text_list)

    # Tokenize the text
    tokens = combined_text.split()

    # Calculate word frequency using Counter
    word_frequency = Counter(tokens)

    # Return Sorted word frequency as Pandas DataFrame
    return pd.DataFrame(word_frequency.most_common(), columns=["word", "frequency"])


def compute_similarity(model, review1: str, review2: str) -> float:
    """
    Computes the similarity between two reviews

    Args:
        model (embedder.EmbedderBase): Embedder model.
        review1 (str): First review.
        review2 (str): Second review.

    Returns:
        float: Cosine similarity between review1 and review2.
    """
    texts = [review1, review2]
    embeddings = model.transform(texts)
    return cosine_similarity(embeddings[0], embeddings[1])
