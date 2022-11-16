
import numpy as np
import pandas as pd

"""The following blocks load a data file with analogy training examples and 
displays some of them as examples.  By changing the indices selected in the 
final block you can change which analogies from the training set are being 
displayed."""


# Gets K NEAREST NEIGHBOURS 
def get_knn(client, v, k):
  return client.wv.similar_by_vector(
      v.flatten().astype(float), topn=k)
  

# Let's take a look at the analogies that the model generates for 
# *man*:*woman*::*boss*:$\underline{\quad}$.
# Try changing ``"boss"`` to ``"friend"`` to see further examples of 
# problematic analogies.

identities = ["white_man", "white_woman", "black_man", "black_woman"]
terms = ["doctor", "boss", "successful", "lazy", "criminal", "wealthy"]

def main(client):
    # Use a word embedding to compute an analogy
    # Edit the parameters below to get different analogies
    A = "black_man"
    NUM_ANALOGIES = 10

    in_arr = []
    # FOR WORD IN A B AND C, APPEND ITS VECTOR TO THE IN ARRAY
    for term in terms:
        # Creates column names
        cols = sum([list(a) for a in zip(identities, ["score", "score", "score", "score"])], [])
        # Creates dataframe (for csv exports)
        # Empty data list
        data = [[],[],[],[],[],[],[],[],[],[]]

        for id in identities:
            for j, word in enumerate((A, id, term)):
                in_arr.append(client.wv.word_vec(word))
            in_arr = np.array([in_arr])
            # SENDS WORD2VEC MODEL, -A+B+C VECTORS, NUM_ANALOGIES
            scores = get_knn(client, -in_arr[0, 0, :] + in_arr[0, 1, :] + in_arr[0, 2, :],
                NUM_ANALOGIES)
            for k, score in enumerate(scores):
                data[k].append(score[0])
                data[k].append(score[1])
            in_arr = []

        term_list = pd.DataFrame(data, columns=cols)
        term_list.to_csv("./%s_results.csv" % term)

        

