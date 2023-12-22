import gc
import glob
import gzip
import os
import pickle
import shutil
import subprocess
import tarfile
from collections import Counter, defaultdict

import gdown
import numpy as np
import pandas as pd
import scipy
import spacy
from numpy.linalg import norm
from spacy.language import Language
from spacy.tokens import DocBin
import torch
from tqdm import tqdm

# Constants
taxonomy = {
    "Pale Lagers and Pilsners": [
        "Euro Pale Lager",
        "Munich Helles Lager",
        "German Pilsener",
        "Czech Pilsener",
        "Light Lager",
        "Vienna Lager",
    ],
    "Amber and Dark Lagers": [
        "Märzen / Oktoberfest",
        "Schwarzbier",
        "Dunkelweizen",
        "Doppelbock",
    ],
    "Pale Ales": [
        "English Pale Ale",
        "American Pale Ale (APA)",
        "American Blonde Ale",
        "Kölsch",
    ],
    "India Pale Ales (IPAs)": [
        "English India Pale Ale (IPA)",
        "American IPA",
        "American Double / Imperial IPA",
        "Belgian IPA",
    ],
    "Stouts and Porters": [
        "American Stout",
        "Milk / Sweet Stout",
        "Irish Dry Stout",
        "English Stout",
        "Oatmeal Stout",
        "American Porter",
        "Baltic Porter",
        "Foreign / Export Stout",
        "American Double / Imperial Stout",
        "Russian Imperial Stout",
    ],
    "Bitters and English Ales": [
        "English Bitter",
        "Irish Red Ale",
        "English Brown Ale",
        "English Porter",
        "English Dark Mild Ale",
        "English Pale Mild Ale",
        "Extra Special / Strong Bitter (ESB)",
        "English Strong Ale",
        "Old Ale",
    ],
    "Wheat Beers": [
        "American Pale Wheat Ale",
        "Hefeweizen",
        "Witbier",
        "Kristalweizen",
    ],
    "Belgian and French Ales": [
        "Saison / Farmhouse Ale",
        "Belgian Strong Pale Ale",
        "Tripel",
        "Quadrupel (Quad)",
        "Dubbel",
        "Belgian Pale Ale",
    ],
    "Specialty Beers": [
        "Fruit / Vegetable Beer",
        "Herbed / Spiced Beer",
        "Chile Beer",
        "Gose",
        "Rauchbier",
        "Smoked Beer",
        "Rye Beer",
        "Scottish Gruit / Ancient Herbed Ale",
        "Sahti",
    ],
    "Hybrid and Experimental Beers": [
        "California Common / Steam Beer",
        "American Black Ale",
        "American Wild Ale",
        "Winter Warmer",
        "Altbier",
        "Scottish Ale",
        "Black & Tan",
    ],
}

columns = [
    "beer_id",
    "beer_name",
    "style",
    "substyle",
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
    "beer": [
        "id",
        "name",
        "style",
        "substyle",
        "abv",
        "nbr_ratings",
        "nbr_reviews",
    ],
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
        processed_dir (str): Path of directory containing processed data.
        num_samples (int | None): Number of samples to check for, or None for all data.

    Returns:
        bool: True if processed data exists with at least the specified number of samples, else False.
    """
    if num_samples is None:
        # Look for 'all' folder
        return _check_folder(processed_dir, "all")
    else:
        # Check all folders with names that can be converted to integers
        for folder_name in os.listdir(processed_dir):
            if folder_name.isdigit() and int(folder_name) >= num_samples:
                if _check_folder(processed_dir, folder_name):
                    return True
        return False


def _check_folder(base_dir: str, folder_name: str) -> bool:
    """
    Helper function to check if a specific folder contains the necessary files.

    Args:
        base_dir (str): Base directory path.
        folder_name (str): Name of the folder to check.

    Returns:
        bool: True if the folder contains the necessary files, else False.
    """
    folder_path = os.path.join(base_dir, folder_name)
    reviews_exists = os.path.isfile(os.path.join(folder_path, "reviews.feather"))
    docs_exists = any(
        os.path.isfile(os.path.join(folder_path, f))
        for f in os.listdir(folder_path)
        if f.endswith(".spacy")
    )
    return reviews_exists and docs_exists


def process_data(
    data_dir: str,
    processed_dir: str,
    nlp: Language,
    doc_bin: DocBin,
    num_samples: int | None = None,
    batch_size: int = 200_000,
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
    else:
        processed_dir = os.path.join(processed_dir, "all")

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

    # Add substyle to reviews
    substyle2style = defaultdict(lambda: "Other")
    for style in taxonomy.keys():
        for substyle in taxonomy[style]:
            substyle2style[substyle] = style

    # Add style to reviews
    reviews["substyle"] = reviews["style"]
    reviews["style"] = reviews["substyle"].map(substyle2style)

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
    batch_number = 0
    for i, doc in enumerate(
        tqdm(
            nlp.pipe(review_texts, n_process=-1),
            total=len(review_texts),
            desc="Processing Spacy docs",
        )
    ):
        doc_bin.add(doc)
        if (i + 1) % batch_size == 0 or i + 1 == len(review_texts):
            batch_number += 1
            batch_path = os.path.join(processed_dir, f"docs_{batch_number}.spacy")
            doc_bin.to_disk(batch_path)

            # delete from memory
            del doc_bin
            gc.collect()
            doc_bin = DocBin()
            print(f"Saved batch {batch_number} of Spacy docs.")

    print("Done processing Spacy docs.")


def load_data(
    processed_dir: str,
    nlp,
    load_docs: bool = False,
    num_samples: int | None = None,
) -> pd.DataFrame:
    """
    Loads the extracted beer data from the specified data directory. It looks for
    a `reviews.feather` file and multiple `docs_*.spacy` files containing the extracted SpaCy object.
    If a limit is specified, it loads and optionally downsamples the data to match the specified number of samples.

    Args:
        processed_dir (str): Path of directory containing extracted data.
        nlp: SpaCy language model.
        num_samples (int, optional): Subset of data to load. Defaults to loading all samples. (None)

    Returns:
        pd.DataFrame: DataFrame containing relevant beer review data and SpaCy docs.
    """
    # Check that processed data exists
    assert processed_data_exists(
        processed_dir, num_samples
    ), "Processed data does not exist. Call `process_data()` first."

    # Find the appropriate folder
    processed_dir = find_appropriate_folder(processed_dir, num_samples)

    # Load reviews from feather file
    reviews_path = os.path.join(processed_dir, "reviews.feather")
    reviews = pd.read_feather(reviews_path)

    # Sanity Check
    if num_samples and len(reviews) > num_samples:
        raise ValueError(
            "The number of reviews should match the number of num samples specified!"
        )

    # If no loading of docs is specified, return reviews only
    if not load_docs:
        return reviews

    # Get all docs files paths
    docs_files = glob.glob(os.path.join(processed_dir, "docs*.spacy"))
    print(f"Found {len(docs_files)} Spacy docs files.")

    # Sort files numerically
    if len(docs_files) > 1:
        docs_files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))

    # Load the files, and save the docs in a list
    all_docs = []
    for file in tqdm(docs_files, desc="Loading Spacy docs", total=len(docs_files)):
        doc_bin = DocBin().from_disk(file)
        all_docs.extend(list(doc_bin.get_docs(nlp.vocab)))
        del doc_bin

    # Add docs to reviews
    reviews[("review", "doc")] = all_docs

    return reviews


def find_appropriate_folder(base_dir: str, num_samples: int | None) -> str:
    """
    Find the appropriate folder for the specified number of samples.

    Args:
        base_dir (str): Base directory path.
        num_samples (int | None): Number of samples to find.

    Returns:
        str: Path to the appropriate folder.
    """
    if num_samples is None:
        return os.path.join(base_dir, "all")
    else:
        suitable_folders = [
            f for f in os.listdir(base_dir) if f.isdigit() and int(f) >= num_samples
        ]
        if suitable_folders:
            suitable_folders.sort(key=int)  # Sort folders numerically
            return os.path.join(base_dir, suitable_folders[0])
        else:
            raise FileNotFoundError(
                "No suitable folder found for the specified number of samples."
            )


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


def load_embeddings(processed_dir: str, num_samples: int | None = None) -> np.ndarray:
    """
    Loads the embeddings from the specified data directory. It looks for
    a `embeddings.npz` file and loads it.

    Args:
        processed_dir (str): Path of directory containing processed data.
        num_samples (int | None): Subset of data to load. Defaults to loading all samples. (None)

    Returns:
        np.ndarray: Array of embeddings.
    """
    embeddings = scipy.sparse.load_npz(os.path.join(processed_dir, "embeddings.npz"))
    if num_samples:
        embeddings = embeddings[:num_samples]
    return embeddings


def filter_data(
    embeddings: np.ndarray,
    reviews: pd.DataFrame,
    min_nbr_reviews: int | None = 50,
    max_nbr_reviews_per_beer: int | None = None,
    min_words: int | None = 50,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Filters the embeddings and reviews to only contain beers with at least `min_nbr_reviews` reviews.

    Args:
        embeddings (np.ndarray): Array of embeddings.
        reviews (pd.DataFrame): DataFrame containing reviews.
        min_nbr_reviews (int): Minimum number of reviews a beer must have to be included.

    Returns:
        tuple[np.ndarray, pd.DataFrame]: Filtered embeddings and reviews.
    """
    # Filter out beers with less than min_nbr_reviews reviews
    if min_nbr_reviews:
        beer_num_reviews = reviews.groupby(("beer", "name")).size()
        beers_to_keep = beer_num_reviews[beer_num_reviews >= min_nbr_reviews].index
        reviews = reviews[reviews[("beer", "name")].isin(beers_to_keep)]
    # Keep only max_nbr_reviews per beer
    if max_nbr_reviews_per_beer:
        reviews = reviews.groupby(("beer", "name")).sample(
            n=max_nbr_reviews_per_beer, random_state=42
        )

    # Filter out reviews with less than min_words words
    if min_words:
        word_lengths = reviews.review.text.apply(lambda x: len(x.split()))
        reviews = reviews[word_lengths >= min_words]
    # Filter embeddings
    embeddings = embeddings[reviews.index]
    # Reset index
    reviews = reviews.reset_index(drop=True)
    return embeddings, reviews


def get_torch_device() -> torch.device:
    """
    Returns the appropriate torch device.

    Returns:
        torch.device: The appropriate torch device.
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def load_vocab(processed_dir: str) -> np.ndarray:
    """
    Loads vocabulary corresponding to the tfidf matrix column dimension.

    Args:
        processed_dir(str) : path to the vocab file

    Returns:
        vocab(np.ndarray) : 1d array containing the vocabulary
    """
    # Loads a dict where key is the word and value is the index in tfidf matrix
    vocab_path = os.path.join(processed_dir, "vocab.pkl")
    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)

    # Parse into an array
    vocab = sorted(
        [(word, index) for word, index in vocab.items()],
        key=lambda x: x[1],
        reverse=False,
    )
    vocab = np.array([word for word, _ in vocab])

    return vocab
