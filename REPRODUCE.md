# Installation

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

You can now run the project by using the virtual environment created by Poetry. In VSCode activate the `venv` environment that is created in the `/.venv` in the root directory. You can also use the environment via the command line by running `poetry shell`. Here, you should use the correct Python executable and have all dependencies installed. Exit the environment via `exit`.
