## TODOs

For official description of deliverables, see this [section](https://epfl-ada.github.io/teaching/fall2023/cs401/projects/#p2-project-proposal-and-initial-analyses) on ada website.

Below, we describe todos for each individual step of the pipeline, note that result of each step should be saved in a binary file to be loaded in the next step.

- [ ] For now, we will use very naive way of versioning the dataset, which is that we can simply upload the most recent data for the given pipeline step to google drive, so then others can download it locally. But of course, if this becomes pain, we will figure something else out - e.g. git lfs, or hugging face datasets.

**Getting Cleang data**
The objective of this step is to get a clean dataframe with all the needed columns, we should be able to load this dataframe from a binary file in the future.

- [ ] We need to figure out which columns we want in addition to the text, beer brand, beer name, ratings etc 
- [ ] We need to figure how the subsampling, because the subsequent steps need to be run only on a smaller dataset since the extraction and embeddings will take some time. We should take into consideration where the users are from, what beer brands we are getting and so on.

**EDA**

- [ ] We should get information about the users who wrote the reviews: where are they from, average number of reviews --> essentially show what biases come into our dataset

**Extracting the text***

The objective of this step is to preprocess the textual reviews and map them to a new text. For instance, we extract adjectives only from the text.

- [ ] We have implemented the adjectives extraction, now we need to implement the other methods that we are interested - see the text.py for more details on how we use spaCy to do this.

**Getting the embeddings**

- [ ] Pier has implemented several methods for getting the embeddings, which is a good start, now we need to run them on subsample and save the results in a binary file.

**Group agreement**

- [ ] We need to implement group consensus method using the embeddings, for start, we can use the cosine similarity between the embeddings of the reviews.
