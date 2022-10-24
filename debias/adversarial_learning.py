import numpy as np
import tensorflow as tf

print("checkpoint 1 - imports complete")

"""The following blocks load a data file with analogy training examples and 
displays some of them as examples.  By changing the indices selected in the 
final block you can change which analogies from the training set are being 
displayed."""

# PRINTS K NEAREST NEIGHBOURS

def print_knn(client, v, k):
    print("%d closest neighbors to A-B+C:" % k)
    # USES GENSIM SIMILAR BY VECTOR
    for neighbor, score in client.wv.similar_by_vector(
            v.flatten().astype(float), topn=k):
        print("%s : score=%f" % (neighbor, score))


"""## Adversarial Networks for Bias Mitigation

The method presented here for removing some of the bias from embeddings is based on the idea that those embeddings are intended to be used to predict some outcome $Y$ based on an input $X$ but that outcome should, in a fair world, be completely unrelated to some protected variable $Z$.  If that were the case then knowing $Y$ would not help you predict $Z$ any better than chance.  This principle can be directly translated into two networks in series as illustrated below.  The first attempts to predict $Y$ using $X$ as input.  The second attempts to use the predicted value of $Y$ to predict $Z$.  See Figure 1 of [the paper](https://arxiv.org/pdf/1801.07593.pdf).

However, simply training the weights in W based on $\nabla_WL_1$ and the weights in $U$ based on $\nabla_UL_2$ won’t actually achieve an unbiased model.  In order to do that you need to incorporate into $W$’s update function the concept that $U$ should be no better than chance at predicting $Z$.  The way that you can achieve that is analogous to how Generative Adversarial Networks (GANs) ([Goodfellow et al. 2014](http://papers.nips.cc/paper/5423-generative-adversarial-nets)) train their generators.

In addition to $\nabla_WL_1$ you incorporate the negation of $\nabla_WL_2$ into $W$’s update function.  However, it’s possible that $\nabla_WL_1$ is changing $W$ in a way which will improve accuracy by using the biased information you are trying to protect.  In order to avoid that you also incorporate a term which removes that component of $\nabla_WL_1$ by projecting it onto $\nabla_WL_2$.  Once you’ve incorporated those two terms, the update function for $W$ becomes:


$\nabla_WL_1-proj_{(\nabla_WL_2)}\nabla_WL_1 - \nabla_WL_2$
"""
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
        protect_weights = tf.compat.v1.get_variable(
            "protect_weights", [self.embed_dim, 1], trainable=False)
        protect_pred = tf.matmul(self.pred, protect_weights)
        protect_loss = tf.losses.mean_squared_error(protect_p, protect_pred)

        pred_opt = tf.compat.v1.train.AdamOptimizer(pred_learning_rate)
        protect_opt = tf.compat.v1.train.AdamOptimizer(protect_learning_rate)

        print(pred_opt)
        protect_grad = {
            v: g for (g, v) in pred_opt.compute_gradients(protect_loss)}
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
        protect_min = protect_opt.minimize(
            protect_loss, var_list=[protect_weights])

        sess.run(tf.compat.v1.global_variables_initializer())
        sess.run(tf.compat.v1.local_variables_initializer())
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
                print("Step", step)
                print("pred loss o", pred_loss_o)
                print("protect loss o", protect_loss_o)
                # print("step: %d; pred_loss_o: %f; protect_loss_o: %f" % (step,
                #             pred_loss_o, protect_loss_o))
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
        isInDict = True
        for word in analogy:
            if word not in index_map:
                isInDict = False
                # print("at least one word missing for analogy: %s" % analogy)
        if isInDict:
            filtered_analogies.append(list(map(index_map.get, analogy)))
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

def main(client, analogies, indices, embed, gender_direction):
  # Edit the training parameters below to experiment with different training runs.
  # For example, try
  pred_learning_rate = 2**-16
  protect_learning_rate = 2**-16
  protect_loss_weight = 1.0
  num_steps = 10000
  batch_size = 1000

  embed_dim = 100
  projection_dims = 1

  tf.compat.v1.disable_eager_execution()

  sess = tf.compat.v1.InteractiveSession()
  with tf.compat.v1.variable_scope('var_scope', reuse=tf.compat.v1.AUTO_REUSE):
      analogy_indices = filter_analogies(analogies, indices)

      data, labels, protect = make_data(analogy_indices, embed, gender_direction)
      data_p = tf.compat.v1.placeholder(
          tf.float32, shape=[None, 3, embed_dim], name="data")
      labels_p = tf.compat.v1.placeholder(
          tf.float32, shape=[None, embed_dim], name="labels")
      protect_p = tf.compat.v1.placeholder(
          tf.float32, shape=[None, 1], name="protect")

      # projection is the space onto which we are "projecting". By default, this is
      # one-dimensional, but this can be tuned by projection_dims
      projection = tf.compat.v1.get_variable(
          "projection", [embed_dim, projection_dims])

      # build the prediction layer
      # pred is the simple computation of d = -a + b + c for a : b :: c : d
      pred = -data_p[:, 0, :] + data_p[:, 1, :] + data_p[:, 2, :]
      pred -= tf.matmul(tf.matmul(pred, projection), tf.transpose(projection))

      trained_model = AdversarialEmbeddingModel(
          client, data_p, embed_dim, projection, projection_dims, pred)

      trained_model.fit(sess, data, data_p, labels, labels_p, protect, protect_p, gender_direction,
                        pred_learning_rate,
                        protect_learning_rate, protect_loss_weight, num_steps, batch_size)
      
  # Parameters
  A = "he"
  B = "she"
  C = "doctor"
  NUM_ANALOGIES = 10

  # Use a word embedding to compute an analogy
  in_arr = []
  for i, word in enumerate((A, B, C)):
      in_arr.append(client.wv.word_vec(word))
  in_arr = np.array([in_arr])

  print_knn(client, sess.run(pred, feed_dict={data_p: in_arr}),
            NUM_ANALOGIES)

  return(client)

"""### Analogy generation using the embeddings with bias reduced by the adversarial model

Let's see how the model that has been trained to mitigate bias performs on the analogy task.
As before, change "boss" to "friend" to see how those analogies have changed too.

"""

# # Parameters
# A = "he"
# B = "she"
# C = "doctor"
# NUM_ANALOGIES = 10

# # Use a word embedding to compute an analogy
# in_arr = []
# for i, word in enumerate((A, B, C)):
#     in_arr.append(client.word_vec(word))
# in_arr = np.array([in_arr])

# print_knn(client, sess.run(pred, feed_dict={data_p: in_arr}),
#           NUM_ANALOGIES)

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
