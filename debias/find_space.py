
import numpy as np

"""
### Defining the Protected Variable of Embeddings

The description of how to incorporate adversarial networks into machine learned models above is very generic because the technique is generally applicable for any type of systems which can be described in terms of input $X$ being predictive of $Y$ but potentially containing information about a protected variable $Z$.  So long as you can construct the relevant update functions you can apply this technique.  However, that doesn’t tell you much about the nature of $X$, $Y$ and $Z$.  In the case of the word analogies task above, $X = B + C - A$ and $Y = D$.  Figuring out what $Z$ should be is a little bit trickier though.  For that we can look to a paper by [Bulokbasi et. al.](http://papers.nips.cc/paper/6227-man-is-to-computer-programmer-as-woman-is-to-homemaker-debiasing-word-embeddings) where they developed an unsupervised methodology for removing gendered semantics from word embeddings.

The first step is to select pairs of words which are relevant to the type of bias you are trying to remove.  In the case of gender you can choose word pairs like “man”:”woman” and “boy”:girl” which have gender as the only difference in their semantics.  Once you have these word pairs you can compute the difference between their embeddings to produce vectors in the embeddings’ semantic space which are roughly parallel to the semantics of gender.  Performing Principal Components Analysis (PCA) on those vectors then gives you the major components of the semantics of gender as defined by the gendered word pairs provided.
"""


def _np_normalize(v):
    """Returns the input vector, normalized."""
    return v / np.linalg.norm(v)


def load_vectors(client, analogies):
    """Loads and returns analogies and embeddings.

    Args:
      client: the client to query.
      analogies: a list of analogies.

    Returns:
      A tuple with:
      - the embedding matrix itself
      - a dictionary mapping from strings to their corresponding indices
        in the embedding matrix
      - the list of words, in the order they are found in the embedding matrix
    """
    words_unfiltered = set()
    for analogy in analogies:
        words_unfiltered.update(analogy)
    print("found %d unique words" % len(words_unfiltered))

    vecs = []
    words = []
    index_map = {}
    for word in words_unfiltered:
        try:
            vecs.append(_np_normalize(client.word_vec(word)))
            index_map[word] = len(words)
            words.append(word)
        except KeyError:
            print("word not found: %s" % word)
    print("words not filtered out: %d" % len(words))

    return np.array(vecs), index_map, words




def find_gender_direction(embed,
                          indices):
    """Finds and returns a 'gender direction'."""
    pairs = [
        ("woman", "man"),
        ("her", "his"),
        ("she", "he"),
        ("aunt", "uncle"),
        ("niece", "nephew"),
        ("daughters", "sons"),
        ("mother", "father"),
        ("daughter", "son"),
        ("granddaughter", "grandson"),
        ("girl", "boy"),
        ("stepdaughter", "stepson"),
        ("mom", "dad"),
    ]
    m = []
    for wf, wm in pairs:
        m.append(embed[indices[wf]] - embed[indices[wm]])
    m = np.array(m)

    # the next three lines are just a PCA.
    m = np.cov(np.array(m).T)
    evals, evecs = np.linalg.eig(m)
    return _np_normalize(np.real(evecs[:, np.argmax(evals)]))



# """Let's now look at the words with the largest *negative* projection onto the gender dimension."""

# words = set()
# for a in analogies:
#   words.update(a)

# df = pd.DataFrame(data={"word": list(words)})
# df["gender_score"] = df["word"].map(
#     lambda w: client.word_vec(w).dot(gender_direction))
# df.sort_values(by="gender_score", inplace=True)
# print (df.head(10))

# """Let's now look at the words with the largest *positive* projection onto the gender dimension."""

# df.sort_values(by="gender_score", inplace=True, ascending=False)
# print (df.head(10))

def main(client, analogies):
    embed, indices, words = load_vectors(client, analogies)

    embed_dim = len(embed[0].flatten())
    print("word embedding dimension: %d" % embed_dim)


    # Using the embeddings, find the gender vector.
    gender_direction = find_gender_direction(embed, indices)
    print("gender direction: %s" % str(gender_direction.flatten()))

    """Once you have the first principal component of the embedding differences, you can start projecting the embeddings of words onto it.  
    That projection is roughly the degree to which a word is relevant to the latent protected variable defined by the first principle 
    component of the word pairs given.  This projection can then be taken as the protected variable $Z$ which the adversary is attempting 
    to predict on the basis of the predicted value of $Y$.  The code below illustrates how to construct a function which computes $Z$ from $X$ in this way.

    Try editing the WORD param in the next cell to see the projection of other words onto the gender direction.
    """

    WORD = "she"

    word_vec = client.word_vec(WORD)
    print(word_vec.dot(gender_direction))

    return indices, embed, gender_direction