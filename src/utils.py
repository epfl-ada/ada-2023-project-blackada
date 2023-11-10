"""
Module containing utility files used throughout the project.

Included functions:
- download_data(): Downloads and extracts the zipped `BeerAdvocate` data from a Google Drive URL
- load_data(): Loads the extracted beer data from the specified data directory
"""

import os
import tarfile
import gdown
import pandas as pd
import gzip
import shutil


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


def load_data(data_dir: str, num_samples: int | None = None) -> pd.DataFrame:
    """
    Loads the extracted beer data from the specified data directory. It looks for
    a `reviews.feather` file which contains the merged data frame. If it doesn't exist,
    it looks for the individual data files, loads them and merges them into a single
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
    if os.path.isfile(os.path.join(data_dir, "reviews.feather")):
        # Load the .feather file
        load_path = os.path.join(data_dir, "reviews.feather")
        reviews = pd.read_feather(load_path)
        return reviews[:num_samples]

    # TODO: Merge relevant parts of this data into the `reviews` DataFrame
    # Load individual data files
    # beers = pd.read_csv(os.path.join(data_dir, "beers.csv"))
    # breweries = pd.read_csv(os.path.join(data_dir, "breweries.csv"))
    # users = pd.read_csv(os.path.join(data_dir, "users.csv"))

    # Load and preprocess reviews
    reviews = _load_reviews(os.path.join(data_dir, "reviews.txt"))
    reviews = _preprocess_reviews(reviews)

    # Save to .feather file if num_samples is None
    save_path = os.path.join(data_dir, "reviews.feather")
    reviews.to_feather(save_path)

    # Limit number of samples if specified
    if num_samples:
        reviews = reviews.head(num_samples)

    return reviews


def _load_reviews(file_path: str) -> pd.DataFrame:
    """
    Loads the reviews from a `.txt` file at a specified path.

    Args:
        file_path (str): Path of `.txt` file containing reviews.

    Returns:
        pd.DataFrame: DataFrame containing reviews.
    """
    beer_data = []
    current_beer = {}

    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()
            if not line or ": " not in line:
                if current_beer:
                    beer_data.append(current_beer)
                    current_beer = {}
            else:
                key, value = line.split(": ", 1) if ": " in line else (None, None)
                if key is not None:
                    if key == "date":
                        value = pd.to_datetime(int(value), unit="s")
                    elif value in {"True", "False"}:
                        value = value == "True"
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
        "brewery_id",
        "brewery_name",
        "user_id",
        "user_name",
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
        "beer": ["id", "name", "style", "abv"],
        "brewery": ["id", "name"],
        "user": ["id", "name"],
        "rating": ["appearance", "aroma", "palate", "taste", "overall", "rating"],
        "review": ["text", "date"],
    }

    # Create multi-index
    multi_columns_tuples = [(k, v) for k, vs in multi_columns.items() for v in vs]
    multi_index = pd.MultiIndex.from_tuples(multi_columns_tuples)

    # Sort columns and create multi-index columns
    reviews = reviews[columns]
    reviews.columns = multi_index

    return reviews
