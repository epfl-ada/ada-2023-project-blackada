# Unlocking the Palate: Evaluating Taste Consensus Among Beer Reviewers 🍺

## 🔗 Shortcuts

Here is a list of things that you likely want to do:

- 🍿 Check out our [datastory](https://ada-2023-project-blackada-webpage.vercel.app/).
- 🧪 Take a look at our detailed [analysis](main.ipynb) 

## 💡 Abstract

Navigating the world of beer reviews can be a daunting task for non-experts. Beer aficionados often describe brews as having nuanced flavors such as "grassy notes" and "biscuity/ crackery malt," with hints of "hay." But do these descriptions reflect the actual tasting experience? Following a "wisdom-of-the-crowd" approach, a descriptor can be considered meaningful if many, independent reviewers use similar descriptors for a beer's taste. To quantify consensus, we use natural language processing techniques to extract descriptors of a beer's taste and numerically represent these descriptors to compute similarity or consensus scores. The consensus scores between beer reviews will unveil whether there is a shared understanding of taste among beer geeks.

## 🔎 Research Questions

Our primary research question is:

**Do beer descriptors in reviews relate to the taste experience of a beer?**

We try to answer this question by investigating the following:

**Is there a consensus among reviewers when using specific descriptors for certain beer types and specific beers?**

This naturally led us to additional interesting sub-questions:

- *Are there patterns of language used across reviews of the same beers, suggesting a common vocabulary for describing certain beer characteristics?*
- *How does the consensus vary by the country of the reviewer or over time?*

## 🔮 Methods

### EDA

We first explore the dataset to investigate the features relevant to the project and filter the data in various ways. We look at the following:

- Features of interest (e.g. beer style, beer name, reviewer location, etc)
- The number of missing values
- The length of textual reviews in words and characters
- The number of reviews grouped by beer style, breweries and beers

### Extractors

Before embedding the reviews, we implemented different options for pre-processing the text and observe its effect on our analysis. The following methods are explored

- Dummy Extraction: Use the review as-is, without any pre-processing.
- Lemmatization: Words were lemmatized to reduce inflected words to their base form.
- Adjective Extraction: We extract only the adjectives from the reviews.
- Stopword Removal: We remove common words that do not add meaning to the text.

Further details on the methods are discussed in [main.ipynb](main.ipynb) and [playground/extractors.ipynb](playground/extractors.ipynb) notebooks.

### Textual Embeddings

We investigate four different potential text embedding models.

- Count Vectorisation: Counts the occurrences of each word.
- TFIDF Vectorisation: Like count vectorisation but also considers the importance of terms within individual documents and across the entire dataset.
- BERT: Embeds review using the pre-trained `bert-base-uncased` model. BERT is a bidirectional encoder-only transformer trained to be effective at various NLP tasks.
- SentenceTransformers: Embeds review using the pre-trained `all-MiniLM-L6-v2` model. The model uses BERT as a backbone, and undergoes further training using Siamese networks.

Further details on the methods are discussed in [main.ipynb](main.ipynb) and [playground/embedders.ipynb](playground/embedders.ipynb) notebooks.

### Consensus Calculation

To quantify the similarity between beer reviews, we use the mean pair-wise cosine similarity within a set of embeddings. 

### Analysis

We can calculate and compare the consensus scores for different subsets of the beer reviews. The following variables will be investigated: Beer style (IPA, Stout etc), Brewery, and specific beers.

### Interpretation

A difference in consensus groups and a subgroup of reviews indicates that the language used in the subgroup is more specific. Given that we group by different categories of beer reviews, this could mean that the language has meaning related to the groups. A simple example might be that, if the word 'clear' is used more often in IPA beer reviews than in the whole dataset, then 'clear' is a meaningful word.

Using CountVectorizer or TFIDF for our final analyses, we can recover the words that correspond to each feature, and therefore find the words which contribute the greatest difference between groups. This way, we can indicate which words are used more often when describing specific kinds of beers, and create a lexicographical guide to beer vocabulary.


## 👥 Team Organisation

| Name   | Contributions                          |
| ------ | -------------------------------------- |
| Pierre | Embeddor Models, README, Interpretation|
| Ludek  | Extraction Models, EDA, README, DataViz|
| Mika   | Pipeline, EDA, Main Notebook           |
| Peter  | Conensus calculation, Data preparation |
| Chris  | EDA, Visualisations                    |

In addition, all members will contribute their relevant section(s) to the final website, as well as help with all other sections where required.
