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

*TODO.*

## 📆 Proposed Timeline

```txt
.
├── Week 8 (30.10-12.11):
│   - Setup GitHub
│   - Automatic data downloading and loading
│   - Write and analyse modules for extraction of beer descriptors, embedding, and consensus scores
│   - Tie together modules in `main.ipynb`
│   - Write `README.md`
|
├── Week 9 (13.11-19.11):
│   - Complete initial analyses for P2
│   - Complete `README.md` to show project feasability in line with instructions for P2
├── 17.11.2023: 🔴 P2 Deadline (23:59 CET)
│
├── Week 10 (20.11-26.11):
│   - Conduct initial analysis of consensus scores and language patterns within smaller beer groups.
|   - Investigate use of different embedding models in combination with text extraction methods and consesus calculations and investigate findings.
│   - (time-permitting): Conduct webscrape on BeerCritic website and build dataset
|
├── Week 11 (27.11-03.12):
│   - Compare language patterns and consensus scores between different groupings of reviews: including exact beers, beer type and reviewer location
|   - Seek feedback from TA on methodology and results thus far
│   - (time-permitting): Attempt to conduct similar analysis on beer critic reviews and compare consensus scores
|
├── Week 12 (04.12-10.12):
│   - Begin writing up findings and collating our best results into visualisations 
│   - Create simple static website using github.io with the skeleton of our data story
|
├── Week 13 (11.12-17.12):
│   - Complete the first draft of the website, including all sections.
│   - Conduct a thorough review of the paper for clarity, coherence, and consistency.
│   - Seek feedback from TA colleagues and mentors.
|
├── Week 14: (18.12-22.12):
│   - Final checks, make revision from feedback
│   - Submission of P3 by the deadline
|
├── 22.12.2023: 🔴 Final Deadline (23:59 CET)
.
```

## 🗿 Internal Milestones

```txt
TODO
```

## 👥 Contributions

| Name   | Contributions |
| ------ | ------------- |
| Pierre | *TODO*        |
| Ludek  | *TODO*        |
| Mika   | *TODO*        |
| Peter  | *TODO*        |
| Chris  | *TODO*        |

## 📚 References

*TODO.*
