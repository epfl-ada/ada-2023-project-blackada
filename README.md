# Unlocking the Palate: Evaluating Taste Consensus Among Beer Reviewers 🍺

## Abstract

Navigating the world of beer reviews can be a daunting task for non-experts. Beer aficionados often describe brews as having nuanced flavors such as "grassy notes" and "biscuity/ crackery malt," with hints of "hay." But do these descriptions reflect the actual tasting experience? Following a "wisdom-of-the-crowd" approach, a descriptor can be considered meaningful if many, independent reviewers use similar descriptors for a beer's taste. To quantify consensus, we use natural language processing techniques to extract descriptors of a beer's taste and numerically represent these descriptors to compute similarity or consensus scores. The consensus scores between beer reviews will unveil whether there is a shared understanding of taste among beer geeks.

## 🔎 Research Questions

- **Overall**: To what extent do beer descriptors in reviews accurately convey the taste experience of a beer?
- Is there a consensus among reviewers when using specific descriptors for certain beer types, and does this consensus vary across different beer styles?
- Are there patterns of language use that are consistent across reviews of the same beer type, suggesting a common vocabulary for describing certain beer characteristics?
- Is there a difference in language use between beers of different qualities? For example, those that have a good aroma but taste worse, or those that are very visually appealing but contain little alcohol. This would suggest that the language use is correlated with the subjective experience of the beer, and therefore the language is meaningful. 
- How does the level of consensus in language use vary between well-established beer types (e.g., IPA, Stout) and more niche or experimental varieties?
- How does the consensus in language use vary by the country of the reviewer? Can we uncover a difference in vocabulary dependent on the location where the beer is drank?
- Does the level of consensus in language use increase or decrease over time for specific beer types, indicating potential shifts in consumer preferences or evolving trends in beer flavors?
- (Optional) Is the language used in beer reviews by the general public drastically different compared to the language used by beer critics? This might suggest that critics are not necessarily very helpful in conveying interpretable reviews of beers.


## 📚 Additional Datasets

Our main project does not require any additional data and will rely solely on the BeerAdvocate dataset. However, we have identified a potential additional dataset on the [BeerCritic](http://www.thebeercritic.com/) website that we might use further down the road, time permitting. The website contains expert reviews by selected beer critics. We could use these reviews to investigate if there is a similar consensus between 'lay-reviewers' and critics in order to question how useful critics are to the average person.

## 🔮 Methods

### EDA 

We first explore the dataset to investigate the features relevant to the project. We look at the following:

- The length of textual reviews
- The location of users
- The dsitribution of different beer types
- 
### Text Pre-processing

`//TODO:` @Ludek fill in the relevant options with brief detail

e.g. removal of reviews below a given threshold, lowercasing reviews etc

Prior to embedding, we carry out the following steps to clean the textual reviews: 

- Removal of Stopwords: Common English stopwords were removed to focus on meaningful content.

- Lemmatization: Words were lemmatized to reduce inflected words to their base form.

- Lowercasing: All text was converted to lowercase for consistency.

- Adjective Extraction: Optionally, we can retain only adjectives for embedding if non-adjectives create too much noise


### Textual Embeddings
We investigate 4 different potential text embedding models: 
- [CountVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html)

The count vectoriser from `sklearn` simply counts the occurences of each word and places the result into a vector - where each entry corresponds to the count of a different word.

- [TFIDFVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)

The Term Frequency-Inverse Document Frequency (TF-IDF) transformation uses CountVectorization but also considers the importance of terms within individual documents and across the entire dataset.

- BERT ([from HuggingFace](https://huggingface.co/bert-base-uncased))

We use the pre-trained `bert-base-uncased` model. BERT is a bidirectional encoder-only transformer trained to be adaptable on various NLP tasks.

- SentenceTransformers ([from `sentence-transformers`](https://www.sbert.net/docs/pretrained_models.html))

We use the pre-trained `all-MiniLM-L6-v2` model. The `sentence-transformer` models use BERT as a backbone, and conduct further training using [Siamese networks](https://towardsdatascience.com/a-friendly-introduction-to-siamese-networks-85ab17522942).

Further details and potential pros/cons of each model are discussed in `main.ipynb`.

### Consensus Calculation

Given a list of embeddings of reviews, a consensus score can be calculated using 2 different methods: 

- Pair-wise Cosine Similarity

We can measure the cosine similarity between each pair of reviews. This metric measures the cosine of the angle between two vectors and ranges from 0 (no similarity) to 1 (complete similarity).

- Pair-wise Euclidean distance

Alternatively, we can use the euclidean distance, which may also take into account the 'strength' (i.e. magnitude) of the embedding. 

### Grouping Analysis 

We can calculate consensus scores for different subsets of the beer reviews and compare. The following variables will be investigated:

- Beer type (IPA, Stout etc)
- Specific beer
- Location of review
- Size of brewery
- Quantative metrics of the beers (Rating, Overall, Appearance, Aroma, Palate, Taste, ABV)

### Interpretation

A significant enough difference in consensus indicates that the language use is different. Given that we group by different categories of beer reviews, this could mean that the language has meaning related to the groups. A simple example might be that, if the word 'clear' is used more often in IPA beer reviews, then 'clear' is a meaningful word. 

If we use either CountVectorizer or TFIDF for our final analyses, then we can recover the words which correspond to each feature, and therefore find the words which contribute the greatest difference between groups. This way, we can indicate which words are used more often when describing specific kinds of beers, and create a lexicographical guide to beer vocabulary.

### Comparison to The Critics (Optional)
Time-permitting, we will scrape the aforementioned website and can compare the language used by critics for the most popular beers. Using word counts or tf-idf features, we will be able to see if the most popular unique descriptors for beers are shared by the critics' reviews. This will be a predominantly qualitative analysis.

### Datastory

We present our findings as a static website hosted on `github.io`, including eye-catching explanations of our method, creative visualisations of the differences in language use and conclusions on what language to use if one wishes to be a beer aficianado!

## 📆 Proposed Timeline

##### **Week 10 (20.11-26.11)**
- Initial analysis of consensus scores and language patterns within smaller beer groups
- Investigate use of different embedding models in combination with text extraction methods and consesus calculations and investigate findings
- (time permitting): Conduct webscrape on BeerCritic website and build dataset

**Week 11 (27.11-03.12)**
- Compare language patterns and consensus scores between different groupings of reviews: including exact beers, beer type and reviewer location
- Seek feedback from TA on methodology and results thus far
- (time-permitting): Attempt to conduct similar analysis on beer critic reviews and compare consensus scores

**Week 12 (04.12-10.12):**
- Begin writing up findings and collate our best results into visualisations 
- Create simple static website using github.io with the skeleton of our data story

**Week 13 (11.12-17.12)**
- Complete the first draft of the website, including all sections
- Conduct a thorough review of the project for clarity, coherence, and consistency on the website and in the code
- Seek feedback from TA colleagues and mentors

**Week 14: (18.12-22.12)**

- Make revisions from feedback and conduct final checks
- Submission of P3 by the deadline

**🔴 Final Deadline**: 22.12.2023 (23:59 CET)

## 👥 Team Organisation

| Name   | Contributions |
| ------ | ------------- |
| Pierre | *TODO*        |
| Ludek  | *TODO*        |
| Mika   | *TODO*        |
| Peter  | *TODO*        |
| Chris  | *TODO*        |

## 📚 References

*TODO.*
