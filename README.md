# Debiasing the Internet

Created by Zara Siddique for an as-of-yet nameless paper.


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
The training includes terms for identifying the race subspace, such as "black_woman", "african_american" and "black_teenager". [code/load_model_script.py](load_model)

Once the model is trained, we find both the race and gender subspaces using the difference between various pairs, such as he -> she and white_man -> black_man, and then computing their principle components. [code/find_space_kv.py](find_space)

We also calculate the race subspace by using the difference between name pairs (e.g. adam -> tyrone) to compare to the phrasal model of calculating the race subspace. [code/find_space_kv.py](find_space)

We then see how occupations score within these subspaces and plot it on a graph. [Graph_plot.ipynb](graph_plot)

At this point, we have shown that the phrasal model performs better at showing stereotypes than the use of names (Caliskan et. al, 2016), and that there are indeed stereotypes within this model. [Currently, I'm just going off my own belief and ideas of what that bias looks like - I may need a solid scientific way of showing these are common stereotypes.]

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
