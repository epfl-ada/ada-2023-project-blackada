import src.utils as utils
import os
import spacy
from spacy.tokens import DocBin
import argparse

# Globals
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
NLP = spacy.load("en_core_web_sm")
DOC_BIN = DocBin(attrs=["LEMMA", "POS"])


def main():
    # Get CLI arguments
    parser = argparse.ArgumentParser(description="Process data")
    parser.add_argument(
        "--num-samples", type=int, default=None, help="Number of reviews to process"
    )
    args = parser.parse_args()

    # Processing data
    utils.process_data(
        DATA_DIR, PROCESSED_DIR, nlp=NLP, doc_bin=DOC_BIN, num_samples=args.num_samples
    )


if __name__ == "__main__":
    main()
