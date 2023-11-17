# Unlocking the Palate: Evaluating Taste Consensus Among Beer Reviewers 🍺

## Abstract

Navigating the world of beer reviews can be a daunting task for non-experts. Beer aficionados often describe brews as having nuanced flavors such as "grassy notes" and "biscuity/ crackery malt," with hints of "hay." But do these descriptions reflect the actual tasting experience? Following a "wisdom-of-the-crowd" approach, a descriptor can be considered meaningful if many, independent reviewers use similar descriptors for a beer's taste. To quantify consensus, we use natural language processing techniques to extract descriptors of a beer's taste and numerically represent these descriptors to compute similarity or consensus scores. The consensus scores between beer reviews will unveil whether there is a shared understanding of taste among beer geeks.

## 🔎 Research Questions

- **Primary**: Do beer descriptors in reviews relate to the taste experience of a beer?
- Is there a consensus among reviewers when using specific descriptors for certain beer types and specific beers?
- Are there patterns of language use across reviews of the same beers, suggesting a common vocabulary for describing certain beer characteristics?
- How does the consensus in language use vary by the country of the reviewe or over time?
- (Optional) Is the language use drastically different between the beer community and beer critics?

## 📚 Additional Datasets

Our main project does not require any additional data and will rely solely on the BeerAdvocate dataset. However, we have identified a potential additional dataset on the [BeerCritic](http://www.thebeercritic.com/) website that might be used, time permitting. The website contains expert reviews by selected beer critics which we can use to question how useful critics are to the average person.

## 🔮 Methods

### EDA 

We first explore the dataset to investigate the features relevant to the project, and filter the data in various ways. We will look at the following:

- The data we have available
- The length of textual reviews in words and characters
- The number of reviews grouped by beer style, breweries and beers
- Any missing data

### Text Pre-Processing

Prior to embedding, we will implemented differnet options for pre-processing the text and see its effect on our analysis. The following are explored: 

- Removal of Stopwords: Common English stopwords were removed to focus on meaningful content.

- Lemmatization: Words were lemmatized to reduce inflected words to their base form.

- Adjective Extraction: Optionally, we can retain only adjectives for embedding if non-adjectives create too much noise


### Textual Embeddings
We investigate 4 different potential text embedding models: 
- [CountVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html)

The CountVectorizer simply counts the occurences of each word.

- [TFIDFVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)

The Term Frequency-Inverse Document Frequency (TF-IDF) transformation uses CountVectorization but also considers the importance of terms within individual documents and across the entire dataset.

- BERT ([HuggingFace](https://huggingface.co/bert-base-uncased))

We use the pre-trained `bert-base-uncased` model. BERT is a bidirectional encoder-only transformer trained to be effective at various NLP tasks.

- SentenceTransformers (from [`sentence-transformers`](https://www.sbert.net/docs/pretrained_models.html))

We use the pre-trained `all-MiniLM-L6-v2` model. The model uses BERT as a backbone, and undergoes further training using [Siamese networks](https://towardsdatascience.com/a-friendly-introduction-to-siamese-networks-85ab17522942).

Further details are discussed in `main.ipynb`.

### Consensus Calculation

To quantify the similarity between beer reviews, we use the mean pair-wise cosine similarity within a set of embeddings.

### Analysis 

We can calculate consensus scores for different subsets of the beer reviews and compare. The following variables will be investigated:

- Beer style (IPA, Stout etc)
- Specific beer
- Location of review
- Size of brewery
- Quantative metrics of the beers (Rating, Overall, Appearance, Aroma, Palate, Taste, ABV)
- The time of the review 

### Interpretation

A difference in consensus between group and a subgroup of reviews indicates the the language used in the subgroup is more speciic. Given that we group by different categories of beer reviews, this could mean that the language has meaning related to the groups. A simple example might be that, if the word 'clear' is used more often in IPA beer reviews than in the whole dataset, then 'clear' is a meaningful word.

Using CountVectorizer or TFIDF for our final analyses, we can recover the words which correspond to each feature, and therefore find the words which contribute the greatest difference between groups. This way, we can indicate which words are used more often when describing specific kinds of beers, and create a lexicographical guide to beer vocabulary.

### Comparison to The Critics (Optional)

Time-permitting, we will scrape the aforementioned website and can compare the language used by critics for the most popular beers. Using word counts or tf-idf features, we will be able to see if the most popular unique descriptors for beers are shared by the critics' reviews.

### Datastory

We present our findings as a static website including explanations of our method and creative visualisation of language use. We will end with describing what language to use for different beers if one wishes to be a beer aficianado!

## 📆 Proposed Timeline

##### **Week 10 (20.11-26.11)**
- Conduct initial analysis of consensus scores and language patterns within smaller beer groups
- Investigate combinations of pre-processing steps, embedding models and consesus calculations
- (time permitting): Conduct webscrape on BeerCritic website

**Week 11 (27.11-03.12)**
- Compare language patterns and consensus scores between different groupings of reviews: including exact beers, beer type and reviewer location
- (time-permitting): Attempt to conduct similar analysis on critics' reviews

**Week 12 (04.12-10.12):**
- Begin writing up findings and collate our best results into visualisations 
- Create simple static website using github.io with the skeleton of our data story

**Week 13 (11.12-17.12)**
- Complete the first draft of the website
- Conduct a thorough review of both the code and website

**Week 14: (18.12-22.12)**
- Conduct final checks
- Submission of P3 by the deadline

**🔴 Final Deadline**: 22.12.2023 (23:59 CET)

## 👥 Team Organisation

| Name   | Contributions |
| ------ | ------------- |
| Pierre | Embeddor Models, README, Webscrape       |
| Ludek  | Extraction Models, EDA     |
| Mika   | Pipeline, EDA, Main Notebook      |
| Peter  | Conensus calculation, Data preparation     |
| Chris  | EDA, Visualisations       |

In addition, all members will contribute their relevant section(s) the final website, as well as help with all other sections where required.
