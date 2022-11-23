# Debiasing the Internet

Created by Zara Siddique for an as-of-yet nameless paper.


## Installation and setup

Clone the repo somewhere.

- To create an environment with all necessary dependencies use `conda env create ./environment.yml`
- Activate the environment with `conda activate debias`


## Running the code

```bash
python debias

## What are we doing?

This code trains a Word2Vec model on the WMT news dataset of 314 million sentences.
The training includes terms for identifying the race subspace, such as "black_woman", "african_american" and "black_teenager"

Once the model is trained, we find both the race and gender subspaces using the difference between various pairs, such as he -> she and white_man -> black_man, and then computing their principle components.

We also calculate the race subspace by using the difference between name pairs (e.g. adam -> tyrone) to compare to the phrasal model of calculating the race subspace.

We then see how occupations score within these subspaces and plot it on a graph.

At this point, we have shown that the phrasal model performs better at showing stereotypes than the use of names (Caliskan et. al, 2016), and that there are indeed stereotypes within this model. [Currently, I'm just going off my own belief and ideas of what that bias looks like - I may need a solid scientific way of showing these are common stereotypes.]

The next steps will be to debias both race and gender using hard/soft debiasing from the Bolukbasi paper and adversarial debiasing.

## Why are we doing it?

Research questions:
- Can we better represent bias for intersectional identities using phrases (instead of single words/names) within word embedding models?
- Can we better understand intersectional stereotypes through the use of these phrases?

- How does debiasing a model by gender only affect race and vice versa?
- Can we effectively debias this model in both race and gender at the same time?

Novel contributions:
- Using phrases, allows us to look at identities previously not able to
- Exploratory measures of how debiasing along one axis affects another
- Debiasing for multiple axes / multiclass debiasing

## Success / evaluation metrics

This is part of where I'm struggling, some options are:
- crowd-worker evaluation of stereotypes (so we can tell which are represented and then which have been successfully debiased)
- find some other way of evaluation that's already been done (the three key papers I'm following on from don't actually have any solid evaluation, just 'it looks less biased')
- show that occupations are not scoring as highly in the race/gender subspaces, therefore there is less biased