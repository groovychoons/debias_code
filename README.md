# Quantifying Intersectional Biases in Static Word Embeddings

This project trains a Word2Vec model, incorporating phrases to represent multi-word identities, such as "black_woman" or "mexican_woman". 

We evaluate whether more bias is detected using these phrases or with names (e.g. European American and African American names) using IBD and EIBD as evaluation metrics.

We also build on Guo and Caliskan by looking at Asian American and Middle Eastern American stereotypes.

This repository provides:
- Code to ...

## Key Papers

- [Distributional techniques for philosophical enquiry (Herbelot et. al, 2012)](https://aclanthology.org/W12-1008.pdf)
More info about this paper - first to look at phrases over multiplicative model - before SWE were a thing but compares using phrase 'black_woman' built into a word distribution, rather than taking the sum of its parts (black x woman)

- [Greenwald]()
Has black, european, japanese, korean names

- [Detecting Emergent Intersectional Biases: Contextualized Word Embeddings Contain a Distribution of Human-like Biases (Guo and Caliskan, 2021)](https://arxiv.org/abs/2006.03955)
Provides EIBD and IBD, and Mexican names

- [An Intersectional Analysis of Gender and Ethnic Stereotypes: Testing Three Hypotheses (Ghavami and Peplau, 2013)](https://journals.sagepub.com/doi/epub/10.1177/0361684312464203)
Provides the stereotypes used for IBD and EIBD


## Corpus

- WMT News Crawl
    - Source: https://data.statmt.org/news-crawl/en/
    - Description: Text extracted from online newspapers
    - Extraction: 2007-2021 combined. Cleaned, tokenized, lemmatized.
    - Version: One sentence per line (add link?)
    - Size: 314m sentences

- Add Wikipedia and UMBC, Common Crawl?

## Installation and setup

Clone the repo somewhere.

- To create an environment with all necessary dependencies use `conda env create ./environment.yml`
- Activate the environment with `conda activate debias`


## Running the code

```bash
python debias
```

## What are we doing?

This code trains a Word2Vec model on the WMT news dataset of 314 million sentences. 
The training includes terms for identifying the race subspace, such as "black_woman", "african_american" and "black_teenager". [load_model file](debias/load_model_script.py)

### Training the model

Once the model is trained, we find both the race and gender subspaces using the difference between various pairs, such as he -> she and white_man -> black_man, and then computing their principle components. [find_space file](debias/find_space_kv.py)

We also calculate the race subspace by using the difference between name pairs (e.g. adam -> tyrone) to compare to the phrasal model of calculating the race subspace. [find_space file](debias/find_space_kv.py)

We then see how occupations score within these subspaces and plot it on a graph. [graph_plot](Graph_plot.ipynb)

At this point, we have shown that the phrasal model performs better at showing stereotypes than the use of names (Caliskan et. al, 2016), and that there are indeed stereotypes within this model. [Currently, I'm just going off my own belief and ideas of what that bias looks like -  need to show this using IBD/EIBD]

## Why are we doing it?

Research questions:
- Can we better represent bias for intersectional identities using phrases (instead of single words/names) within word embedding models?
- Can we better understand intersectional stereotypes through the use of these phrases?

Novel contributions:
- Using phrases, allows us to look at identities previously not able to

## Success / evaluation metrics

- The three key papers I'm following on from don't actually have any solid evaluation, just 'it looks less biased'
- IBD/EIBD (Guo and Caliskan paper)
- show that occupations are not scoring as highly in the race/gender subspaces, therefore there is less biased
