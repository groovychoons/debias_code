
"""# Mitigating Unwanted Biases with Adversarial Learning

Authors: Andrew Zaldivar, Ben Hutchinson, Blake Lemoine, Brian Zhang, Margaret Mitchell

---

## Summary of this Notebook

This notebook is a guide to the paper ([archiv](https://arxiv.org/pdf/1801.07593.pdf))

First, we'll import all the packages that we'll need.
"""

import gensim
from gensim.models import Word2Vec, KeyedVectors
import gzip
import numpy as np
import os
import pandas as pd
import tensorflow as tf

print("checkpoint 1 - imports complete")

WORD2VEC_FILE = os.path.join("../data", "GoogleNews-vectors-negative300.bin.gz")
ANALOGIES_FILE = os.path.join("../data", "questions-words.txt")

# Initialize the embeddings client if this hasn't been done yet.
# For the efficiency of this notebook we just load the first 2M words, and don't
# re-initialize the client if it already exists. You could of course filter the
# word list in other ways.
if not 'client' in vars():
  print ("Loading word embeddings from %s" % WORD2VEC_FILE)
  client = KeyedVectors.load_word2vec_format(WORD2VEC_FILE, binary=True, limit=2000000)

print("checkpoint 2 - word2vec loaded")

"""The following blocks load a data file with analogy training examples and displays some of them as examples.  By changing the indices selected in the final block you can change which analogies from the training set are being displayed."""

def print_knn(client, v, k):
  print ("%d closest neighbors to A-B+C:" % k)
  for neighbor, score in client.similar_by_vector(
      v.flatten().astype(float), topn=k):
    print ("%s : score=%f" % (neighbor, score))

#Lets take a look at the analogies that the model generates for *man*:*woman*::*boss*:$\underline{\quad}$.
#Try changing ``"boss"`` to ``"friend"`` to see further examples of problematic analogies.


# Use a word embedding to compute an analogy
# Edit the parameters below to get different analogies
A = "man"
B = "woman"
C = "boss"
NUM_ANALOGIES = 5

in_arr = []
for i, word in enumerate((A, B, C)):
  in_arr.append(client.word_vec(word))
in_arr = np.array([in_arr])

print("checkpoint 3")

print_knn(client, -in_arr[0, 0, :] + in_arr[0, 1, :] + in_arr[0, 2, :],
          NUM_ANALOGIES)

def load_analogies(filename):
  """Loads analogies.

  Args:
    filename: the file containing the analogies.

  Returns:
    A list containing the analogies.
  """
  analogies = []
  with open(filename, "r") as fast_file:
    for line in fast_file:
      line = line.strip()
      # in the analogy file, comments start with :
      if line[0] == ":":
        continue
      words = line.split()
      # there are no misformatted lines in the analogy file, so this should
      # only happen once we're done reading all analogies.
      if len(words) != 4:
        print ("Invalid line: %s" % line)
        continue
      analogies.append(words)
  print ("loaded %d analogies" % len(analogies))
  return analogies

analogies = load_analogies(ANALOGIES_FILE)
print ("\n".join("%s is to %s as %s is to %s" % tuple(x) for x in analogies[:5]))

"""## Adversarial Networks for Bias Mitigation

The method presented here for removing some of the bias from embeddings is based on the idea that those embeddings are intended to be used to predict some outcome $Y$ based on an input $X$ but that outcome should, in a fair world, be completely unrelated to some protected variable $Z$.  If that were the case then knowing $Y$ would not help you predict $Z$ any better than chance.  This principle can be directly translated into two networks in series as illustrated below.  The first attempts to predict $Y$ using $X$ as input.  The second attempts to use the predicted value of $Y$ to predict $Z$.  See Figure 1 of [the paper](https://arxiv.org/pdf/1801.07593.pdf).

However, simply training the weights in W based on $\nabla_WL_1$ and the weights in $U$ based on $\nabla_UL_2$ won’t actually achieve an unbiased model.  In order to do that you need to incorporate into $W$’s update function the concept that $U$ should be no better than chance at predicting $Z$.  The way that you can achieve that is analogous to how Generative Adversarial Networks (GANs) ([Goodfellow et al. 2014](http://papers.nips.cc/paper/5423-generative-adversarial-nets)) train their generators.

In addition to $\nabla_WL_1$ you incorporate the negation of $\nabla_WL_2$ into $W$’s update function.  However, it’s possible that $\nabla_WL_1$ is changing $W$ in a way which will improve accuracy by using the biased information you are trying to protect.  In order to avoid that you also incorporate a term which removes that component of $\nabla_WL_1$ by projecting it onto $\nabla_WL_2$.  Once you’ve incorporated those two terms, the update function for $W$ becomes:


$\nabla_WL_1-proj_{(\nabla_WL_2)}\nabla_WL_1 - \nabla_WL_2$

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
  print( "found %d unique words" % len(words_unfiltered))

  vecs = []
  words = []
  index_map = {}
  for word in words_unfiltered:
    try:
      vecs.append(_np_normalize(client.word_vec(word)))
      index_map[word] = len(words)
      words.append(word)
    except KeyError:
      print( "word not found: %s" % word)
  print ("words not filtered out: %d" % len(words))

  return np.array(vecs), index_map, words

embed, indices, words = load_vectors(client, analogies)

embed_dim = len(embed[0].flatten())
print ("word embedding dimension: %d" % embed_dim)

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

# Using the embeddings, find the gender vector.
gender_direction = find_gender_direction(embed, indices)
print ("gender direction: %s" % str(gender_direction.flatten()))

"""Once you have the first principal component of the embedding differences, you can start projecting the embeddings of words onto it.  
That projection is roughly the degree to which a word is relevant to the latent protected variable defined by the first principle 
component of the word pairs given.  This projection can then be taken as the protected variable $Z$ which the adversary is attempting 
to predict on the basis of the predicted value of $Y$.  The code below illustrates how to construct a function which computes $Z$ from $X$ in this way.

Try editing the WORD param in the next cell to see the projection of other words onto the gender direction.
"""

WORD = "she"

word_vec = client.word_vec(WORD)
print( word_vec.dot(gender_direction))

"""Let's now look at the words with the largest *negative* projection onto the gender dimension."""

words = set()
for a in analogies:
  words.update(a)

df = pd.DataFrame(data={"word": list(words)})
df["gender_score"] = df["word"].map(
    lambda w: client.word_vec(w).dot(gender_direction))
df.sort_values(by="gender_score", inplace=True)
print (df.head(10))

"""Let's now look at the words with the largest *positive* projection onto the gender dimension."""

df.sort_values(by="gender_score", inplace=True, ascending=False)
print (df.head(10))

"""### Training the model

Training adversarial networks is hard. They are touchy, and if touched the wrong way, they blow up VERY quickly. One must be very careful to train both models slowly enough, so that the parameters in the models do not diverge. In practice, this usually entails significantly lowering the step size of both the classifier and the adversary. It is also probably beneficial to initialize the parameters of the adversary to be extremely small, to ensure that the classifier does not overfit against a particular (sub-optimal) adversary (such overfitting can very quickly cause divergence!).  It is also possible that if the classifier is too good at hiding the protected variable from the adversary then the adversary will impose updates that diverge in an effort to improve its performance.  The solution to that can sometimes be to actually increase the adversary’s learning rate to prevent divergence (something almost unheard of in most learning systems).  Below is a system for learning the debiasing model for word embeddings described above.  

Inspect the terminal output for the kernel to inspect the performance of the model. Try modifying the hyperparameters at the top to see how that impacts the convergence properties of the system.
"""

def tf_normalize(x):
  """Returns the input vector, normalized.

  A small number is added to the norm so that this function does not break when
  dealing with the zero vector (e.g. if the weights are zero-initialized).

  Args:
    x: the tensor to normalize
  """
  return x / (tf.norm(x) + np.finfo(np.float32).tiny)


class AdversarialEmbeddingModel(object):
  """A model for doing adversarial training of embedding models."""

  def __init__(self, client,
               data_p, embed_dim, projection,
               projection_dims, pred):
    """Creates a new AdversarialEmbeddingModel.

    Args:
      client: The (possibly biased) embeddings.
      data_p: Placeholder for the data.
      embed_dim: Number of dimensions used in the embeddings.
      projection: The space onto which we are "projecting".
      projection_dims: Number of dimensions of the projection.
      pred: Prediction layer.
    """
    # load the analogy vectors as well as the embeddings
    self.client = client
    self.data_p = data_p
    self.embed_dim = embed_dim
    self.projection = projection
    self.projection_dims = projection_dims
    self.pred = pred

  def nearest_neighbors(self, sess, in_arr,
                        k):
    """Finds the nearest neighbors to a vector.

    Args:
      sess: Session to use.
      in_arr: Vector to find nearest neighbors to.
      k: Number of nearest neighbors to return
    Returns:
      List of up to k pairs of (word, score).
    """
    v = sess.run(self.pred, feed_dict={self.data_p: in_arr})
    return self.client.similar_by_vector(v.flatten().astype(float), topn=k)

  def write_to_file(self, sess, f):
    """Writes a model to disk."""
    np.savetxt(f, sess.run(self.projection))

  def read_from_file(self, sess, f):
    """Reads a model from disk."""
    loaded_projection = np.loadtxt(f).reshape(
        [self.embed_dim, self.projection_dims])
    sess.run(self.projection.assign(loaded_projection))

  def fit(self,
          sess,
          data,
          data_p,
          labels,
          labels_p,
          protect,
          protect_p,
          gender_direction,
          pred_learning_rate,
          protect_learning_rate,
          protect_loss_weight,
          num_steps,
          batch_size,
          debug_interval=1000):
    """Trains a model.

    Args:
      sess: Session.
      data: Features for the training data.
      data_p: Placeholder for the features for the training data.
      labels: Labels for the training data.
      labels_p: Placeholder for the labels for the training data.
      protect: Protected variables.
      protect_p: Placeholder for the protected variables.
      gender_direction: The vector from find_gender_direction().
      pred_learning_rate: Learning rate for predicting labels.
      protect_learning_rate: Learning rate for protecting variables.
      protect_loss_weight: The constant 'alpha' found in
          debias_word_embeddings.ipynb.
      num_steps: Number of training steps.
      batch_size: Number of training examples in each step.
      debug_interval: Frequency at which to log performance metrics during
          training.
    """
    feed_dict = {
        data_p: data,
        labels_p: labels,
        protect_p: protect,
    }
    # define the prediction loss
    pred_loss = tf.losses.mean_squared_error(labels_p, self.pred)

    # compute the prediction of the protected variable.
    # The "trainable"/"not trainable" designations are for the predictor. The
    # adversary explicitly specifies its own list of weights to train.
    protect_weights = tf.get_variable(
        "protect_weights", [self.embed_dim, 1], trainable=False)
    protect_pred = tf.matmul(self.pred, protect_weights)
    protect_loss = tf.losses.mean_squared_error(protect_p, protect_pred)

    pred_opt = tf.optimizers.Adam(pred_learning_rate)
    protect_opt = tf.optimizers.Adam(protect_learning_rate)

    protect_grad = {v: g for (g, v) in pred_opt.compute_gradients(protect_loss)}
    pred_grad = []

    # applies the gradient expression found in the document linked
    # at the top of this file.
    for (g, v) in pred_opt.compute_gradients(pred_loss):
      unit_protect = tf_normalize(protect_grad[v])
      # the two lines below can be commented out to train without debiasing
      g -= tf.reduce_sum(g * unit_protect) * unit_protect
      g -= protect_loss_weight * protect_grad[v]
      pred_grad.append((g, v))
      pred_min = pred_opt.apply_gradients(pred_grad)

    # compute the loss of the protected variable prediction.
    protect_min = protect_opt.minimize(protect_loss, var_list=[protect_weights])

    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    step = 0
    while step < num_steps:
      # pick samples at random without replacement as a minibatch
      ids = np.random.choice(len(data), batch_size, False)
      data_s, labels_s, protect_s = data[ids], labels[ids], protect[ids]
      sgd_feed_dict = {
          data_p: data_s,
          labels_p: labels_s,
          protect_p: protect_s,
      }

      if not step % debug_interval:
        metrics = [pred_loss, protect_loss, self.projection]
        metrics_o = sess.run(metrics, feed_dict=feed_dict)
        pred_loss_o, protect_loss_o, proj_o = metrics_o
        # log stats every so often: number of steps that have passed,
        # prediction loss, adversary loss
        print("step: %d; pred_loss_o: %f; protect_loss_o: %f" % (step,
                     pred_loss_o, protect_loss_o))
        for i in range(proj_o.shape[1]):
          print("proj_o: %f; dot(proj_o, gender_direction): %f)" %
                       (np.linalg.norm(proj_o[:, i]),
                       np.dot(proj_o[:, i].flatten(), gender_direction)))
      sess.run([pred_min, protect_min], feed_dict=sgd_feed_dict)
      step += 1
      
def filter_analogies(analogies,
                     index_map):
  filtered_analogies = []
  for analogy in analogies:
    if filter(index_map.has_key, analogy) != analogy:
      print ("at least one word missing for analogy: %s" % analogy)
    else:
      filtered_analogies.append(map(index_map.get, analogy))
  return filtered_analogies

def make_data(
    analogies, embed,
    gender_direction):
  """Preps the training data.

  Args:
    analogies: a list of analogies
    embed: the embedding matrix from load_vectors
    gender_direction: the gender direction from find_gender_direction

  Returns:
    Three numpy arrays corresponding respectively to the input, output, and
    protected variables.
  """
  data = []
  labels = []
  protect = []
  for analogy in analogies:
    # the input is just the word embeddings of the first three words
    data.append(embed[analogy[:3]])
    # the output is just the word embeddings of the last word
    labels.append(embed[analogy[3]])
    # the protected variable is the gender component of the output embedding.
    # the extra pair of [] is so that the array has the right shape after
    # it is converted to a numpy array.
    protect.append([np.dot(embed[analogy[3]], gender_direction)])
  # Convert all three to numpy arrays, and return them.
  return tuple(map(np.array, (data, labels, protect)))

"""Edit the training parameters below to experiment with different training runs.

For example, try increasing the number of training steps to 50k.
"""

# Edit the training parameters below to experiment with different training runs.
# For example, try 
pred_learning_rate = 2**-16
protect_learning_rate = 2**-16
protect_loss_weight = 1.0
num_steps = 10000
batch_size = 1000

embed_dim = 300
projection_dims = 1


sess = tf.InteractiveSession()
with tf.variable_scope('var_scope', reuse=tf.AUTO_REUSE):
    analogy_indices = filter_analogies(analogies, indices)

    data, labels, protect = make_data(analogy_indices, embed, gender_direction)
    data_p = tf.placeholder(tf.float32, shape=[None, 3, embed_dim], name="data")
    labels_p = tf.placeholder(tf.float32, shape=[None, embed_dim], name="labels")
    protect_p = tf.placeholder(tf.float32, shape=[None, 1], name="protect")

    # projection is the space onto which we are "projecting". By default, this is
    # one-dimensional, but this can be tuned by projection_dims
    projection = tf.get_variable("projection", [embed_dim, projection_dims])

    # build the prediction layer
    # pred is the simple computation of d = -a + b + c for a : b :: c : d
    pred = -data_p[:, 0, :] + data_p[:, 1, :] + data_p[:, 2, :]
    pred -= tf.matmul(tf.matmul(pred, projection), tf.transpose(projection))

    trained_model = AdversarialEmbeddingModel(
        client, data_p, embed_dim, projection, projection_dims, pred)

    trained_model.fit(sess, data, data_p, labels, labels_p, protect, protect_p, gender_direction,
              pred_learning_rate,
            protect_learning_rate, protect_loss_weight, num_steps, batch_size)

"""### Analogy generation using the embeddings with bias reduced by the adversarial model

Let's see how the model that has been trained to mitigate bias performs on the analogy task.
As before, change "boss" to "friend" to see how those analogies have changed too.

"""

# Parameters
A = "man"
B = "woman"
C = "boss"
NUM_ANALOGIES = 5

# Use a word embedding to compute an analogy
in_arr = []
for i, word in enumerate((A, B, C)):
  in_arr.append(client.word_vec(word))
in_arr = np.array([in_arr])

print_knn(client, sess.run(pred, feed_dict={data_p: in_arr}),
          NUM_ANALOGIES)

"""##Conclusion

The method demonstrated here helps to reduce the amount of bias in word embeddings and, although not demonstrated here,
 generalizes quite well to other domains and tasks.  By trying to hide a protected variable from an adversary, a machine
  learned system can reduce the amount of biased information about that protected variable implicit in the system.  In 
  addition to the specific method demonstrated here there are many variations on this theme which can be used to achieve
   different degrees and types of debiasing.  For example, you could debias with respect to more than one principle 
   component of the protected variable by having the adverary predict multiple projections.  Many other elaborations on 
   this basic idea are possible and hopefully this relatively simple system can serve as the basis for more complex and 
   sophisticated systems capable of achieving subtle types of bias mitigation in many applications.
"""