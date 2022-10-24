
import numpy as np
import heapq

"""The following blocks load the list of toxic words and finds an overall toxicity score
 for the identities below."""

def main(client):

    # Import toxic words
    list_ = open("./data/list.txt").read().split()
    print("Number of toxic words: ", len(list_))
    toxic_words = []
    # Remove words not in Word2Vec vocab

    words2 = list(client.wv.key_to_index.keys())
    for a in list_:
        if a in words2 and a not in toxic_words:
            toxic_words.append(a)
    print("Number of toxic words in Word2Vec vocab: ", len(toxic_words))

    toxicity_ids = [['black_woman', 0], ['white_woman', 0], ['black_man', 0], ['white_man', 0]]
    toxicity_scores = []

    for a in toxic_words:
        temp_scores = []
        for id in toxicity_ids:
            score = client.wv.similarity(a, id[0])
            temp_scores.append(score)
        toxicity_scores.append([a, temp_scores])

    for num, id in enumerate(toxicity_ids):
        print(num, id[0])
        data = heapq.nlargest(10, toxicity_scores, key=lambda e:e[1][num])
        for d in data:
            print(d[0])
    #print(toxicity_scores)

# PRINTS K NEAREST NEIGHBOURS 
def print_knn(client, v, k):
  print ("%d closest neighbors to A-B+C:" % k)
  # USES GENSIM SIMILAR BY VECTOR
  for neighbor, score in client.wv.similar_by_vector(
      v.flatten().astype(float), topn=k):
    print ("%s : score=%f" % (neighbor, score))