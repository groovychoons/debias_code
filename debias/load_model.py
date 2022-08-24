
import os
import gensim.downloader as api
from gensim.models import KeyedVectors

WORD2VEC_FILE = os.path.join("data", "GoogleNews-vectors-negative300.bin.gz")

# Initialize the embeddings client if this hasn't been done yet.
# For efficiency we just load the first 2M words, and don't re-initialize the 
# client if it already exists.

def main():
  print ("Loading word embeddings from %s" % WORD2VEC_FILE)
  client = KeyedVectors.load_word2vec_format(WORD2VEC_FILE, binary=True, limit=2000000)
  # client = api.load("glove-wiki-gigaword-50")
  return client