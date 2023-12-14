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
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Iterable

import gdown
import pandas as pd
import spacy
from numpy.linalg import norm
from tqdm import tqdm

nlp = spacy.load("en_core_web_sm")


def get_line_count(file_path):
    """Quick way of obtaining the number of lines in a (large) file.

    Args:
        file_path (str): Path to file.

    Returns:
        line_count (int): Number of lines in file.
    """
    result = subprocess.run(["wc", "-l", file_path], stdout=subprocess.PIPE, text=True)
    line_count = int(result.stdout.strip().split()[0])
    return line_count


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


def load_data(
    data_dir: str, num_samples: int | None = None, seed: int = 42
) -> pd.DataFrame:
    """
    Loads the extracted beer data from the specified data directory. It looks for
    a `reviews.feather` file which contains the merged data frame including the
    SpaCy object of reach review. If the file doesn't exist, it looks for the
    individual data files, loads them and merges them into a single
    DataFrame and saves it as `data.feather` for future reuse.

    The num_samples parameter can be used to limit the number of samples loaded.
    However, on first load, all samples are always loaded and saved to a .feather file.

    Args:
        data_dir (str): Path of directory containing extracted data.
        num_samples (int, optional): Number of samples to load.
            Defaults to loading all samples. (None)

    Returns:
        pd.DataFrame: DataFrame containing relevant beer data.
    """
    # Look for a .feather file containing the merged data
    reviews_feather_path = os.path.join(data_dir, "reviews.feather")
    if os.path.isfile(reviews_feather_path):
        # Load the .feather file
        reviews = pd.read_feather(reviews_feather_path)

        # Deserialize SpaCy Doc objects
        reviews[("review", "doc")] = reviews.review.doc.apply(
            lambda x: spacy.tokens.Doc(nlp.vocab).from_bytes(x)
        )

        # Limit number of samples if specified
        if num_samples:
            return reviews.sample(n=num_samples, random_state=seed)

        return reviews

    # Load reviews
    print("No .feather file found. Loading raw data...")
    reviews = _load_reviews(data_dir)

    # Load metainfo for reviews
    print("Merging reviews with metainfo...")
    beers, breweries, users = _load_metainfo(data_dir)

    # Merge reviews with metainfo
    reviews = reviews.merge(beers, on="beer_id", how="left")
    reviews = reviews.merge(breweries, on="brewery_id", how="left")
    reviews = reviews.merge(users, on="user_id", how="left")

    # Preprocess reviews
    print("Preprocessing reviews...")
    reviews = _preprocess_reviews(reviews)

    # Serialize SpaCy Doc objects before saving
    reviews[("review", "doc")] = reviews.review.doc.apply(
        lambda doc: doc.to_bytes(),
    )

    # Save to .feather file
    reviews.to_feather(reviews_feather_path)

    # Convert Spacy Doc objects back to Doc objects
    reviews[("review", "doc")] = reviews.review.doc.apply(
        lambda doc: spacy.tokens.Doc(nlp.vocab).from_bytes(doc)
    )

    # Limit number of samples if specified
    if num_samples:
        # Shuffle reviews before limiting number of samples
        reviews = reviews.sample(n=num_samples, random_state=seed)

    return reviews


def _load_reviews(data_dir: str) -> pd.DataFrame:
    """
    Loads the reviews from a `reviews.txt` file in a specified directory.

    Args:
        data_dir (str): Directory containing raw reviews

    Returns:
        pd.DataFrame: DataFrame containing reviews.

    Notes:
    Note that we ensure correct casting of values before saving.
    However, values of type str are casted to object type since
    string can have different lengths.
    """
    beer_data = []
    current_beer = {}
    file_path = os.path.join(data_dir, "reviews.txt")
    total_count = get_line_count(file_path)

    with open(file_path, "r") as file:
        for line in tqdm(file, total=total_count, desc="Loading raw reviews"):
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

    return pd.DataFrame(beer_data)


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


def _preprocess_reviews(reviews: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses the reviews DataFrame. Currently it:
    - Sorts columns
    - Converts columns into multi-index columns
    - Drops any reviews with missing values
    - Adds the SpaCy `Doc` objects to the DataFrame in column ("review", "doc")

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

    # Load SpaCy model and raw review texts
    review_texts = reviews.review.text.tolist()

    # Compute SpaCy Doc objects in parallel
    review_docs = _parallel_map_with_progress(nlp, review_texts)

    # Add SpaCy Doc objects to DataFrame
    reviews[("review", "doc")] = review_docs

    return reviews


def _parallel_map_with_progress(func: Callable, iterable: Iterable) -> list:
    """
    Maps a function to an iterable in parallel and displays a progress bar.

    Args:
        func (Callable): Function to map to iterable.
        iterable (Iterable): Iterable to map function to.

    Returns:
        list: List of results from mapping function to iterable.
    """
    with ThreadPoolExecutor() as executor:
        return list(tqdm(executor.map(func, iterable), total=len(iterable)))


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
