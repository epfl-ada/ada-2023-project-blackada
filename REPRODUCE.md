## Installation

To reproduce all results, this notebook should be run with the correct **Python version** inside the specified **virtual environment** to use all packages with the correct version.

We use [Poetry](https://python-poetry.org/) for package management and versioning. This project was developed for Python `3.10` and Poetry `1.2`. We recommend installing Python via [pyenv](https://github.com/pyenv/pyenv) and Poetry via [pipx](https://pypa.github.io/pipx/).

```bash
pyenv install 3.10
```

Then, install Poetry via `pipx`

```bash
pipx install poetry==1.2.0
```

The project has a `.python-version` in the root directory, which will automatically activate the correct Python version when you enter the project directory. You can check that the correct Python version is used by running `python --version` (should be any minor release of Python `3.10`) and that `poetry --version` is `1.2.0`.

Next, we install all dependencies via Poetry:

```bash
poetry install
```

In addition, you need to load `spaCy` language models for English. This can be done by running the following commands:

```bash
poetry shell
python -m spacy download en_core_web_sm
```

You can now run the project by using the virtual environment created by Poetry. In VSCode activate the `venv` environment that is created in the `/.venv` in the root directory. You can also use the environment via the command line by running `poetry shell`. Here, you should use the correct Python executable and have all dependencies installed. Exit the environment via `exit`.

## Getting the Data (Optional)

Given the large size of the dataset, it is advisable to download the raw as well as the processed data, so you can rather focus on our analysis. The below command will download the data to the `data` directory in the root directory. It takes around 17 min.

```bash
huggingface-cli download ludekcizinsky/blackada --local-dir data --repo-type dataset
```