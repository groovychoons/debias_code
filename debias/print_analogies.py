
import numpy as np

"""The following blocks load a data file with analogy training examples and 
displays some of them as examples.  By changing the indices selected in the 
final block you can change which analogies from the training set are being 
displayed."""


# PRINTS K NEAREST NEIGHBOURS 
def print_knn(client, v, k):
  print ("%d closest neighbors to A-B+C:" % k)
  # USES GENSIM SIMILAR BY VECTOR
  for neighbor, score in client.wv.similar_by_vector(
      v.flatten().astype(float), topn=k):
    print ("%s : score=%f" % (neighbor, score))

# Let's take a look at the analogies that the model generates for 
# *man*:*woman*::*boss*:$\underline{\quad}$.
# Try changing ``"boss"`` to ``"friend"`` to see further examples of 
# problematic analogies.

def main(client):
    # Use a word embedding to compute an analogy
    # Edit the parameters below to get different analogies
    A = "white_man"
    B = "white_woman"
    C = "boss"
    NUM_ANALOGIES = 10

    in_arr = []
    # FOR WORD IN A B AND C, APPEND ITS VECTOR TO THE IN ARRAY
    for i, word in enumerate((A, B, C)):
        in_arr.append(client.wv.word_vec(word))
    
    in_arr = np.array([in_arr])

    # SENDS WORD2VEC MODEL, -A+B+C VECTORS, NUM_ANALOGIES
    print_knn(client, -in_arr[0, 0, :] + in_arr[0, 1, :] + in_arr[0, 2, :],
            NUM_ANALOGIES)
