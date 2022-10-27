import numpy as np
import tensorflow as tf
import pandas as pd

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

occupations = [["accountant", 0.0, 0.4], ["acquaintance", 0.0, 0.0], ["actor", 0.8, 0.0], ["actress", -1.0, 0.0], ["adjunct_professor", 0.0, 0.5], ["administrator", 0.0, 0.2], ["adventurer", 0.0, 0.5], ["advocate", 0.0, -0.1], ["aide", 0.0, -0.2], ["alderman", 0.7, 0.2], ["alter_ego", 0.0, 0.0], ["ambassador", 0.0, 0.7], ["analyst", 0.0, 0.4], ["anthropologist", 0.0, 0.4], ["archaeologist", 0.0, 0.6], ["archbishop", 0.4, 0.5], ["architect", 0.1, 0.6], ["artist", 0.0, -0.2], ["artiste", -0.1, -0.2], ["assassin", 0.1, 0.8], ["assistant_professor", 0.1, 0.4], ["associate_dean", 0.0, 0.4], ["associate_professor", 0.0, 0.4], ["astronaut", 0.1, 0.8], ["astronomer", 0.1, 0.5], ["athlete", 0.0, 0.7], ["athletic_director", 0.1, 0.7], ["attorney", 0.0, 0.3], ["author", 0.0, 0.1], ["baker", 0.0, -0.1], ["ballerina", -0.5, -0.5], ["ballplayer", 0.2, 0.8], ["banker", 0.0, 0.6], ["barber", 0.5, 0.5], ["baron", 0.6, 0.3], ["barrister", 0.1, 0.4], ["bartender", 0.0, 0.3], ["biologist", 0.0, 0.1], ["bishop", 0.6, 0.4], ["bodyguard", 0.1, 0.9], ["bookkeeper", 0.0, -0.4], ["boss", 0.0, 0.7], ["boxer", 0.1, 0.9], ["broadcaster", -0.1, 0.4], ["broker", 0.1, 0.5], ["bureaucrat", 0.1, 0.5], ["businessman", 0.8, 0.2], ["businesswoman", -0.9, -0.1], ["butcher", 0.1, 0.9], ["butler", 0.5, 0.5], ["cab_driver", 0.1, 0.8], ["cabbie", 0.1, 0.6], ["cameraman", 0.8, 0.1], ["campaigner", 0.0, 0.2], ["captain", 0.1, 0.6], ["cardiologist", 0.1, 0.5], ["caretaker", 0.0, -0.9], ["carpenter", 0.1, 0.8], ["cartoonist", 0.0, 0.5], ["cellist", -0.1, 0.0], ["chancellor", 0.1, 0.6], ["chaplain", 0.1, 0.6], ["character", 0.0, 0.0], ["chef", 0.0, 0.5], ["chemist", 0.0, 0.2], ["choreographer", -0.2, -0.2], ["cinematographer", 0.0, 0.5], ["citizen", 0.0, 0.0], ["civil_servant", 0.0, 0.2], ["cleric", 0.3, 0.3], ["clerk", 0.0, -0.5], ["coach", 0.1, 0.8], ["collector", 0.0, 0.4], ["colonel", 0.1, 0.8], ["columnist", 0.0, 0.2], ["comedian", 0.0, 0.3], ["comic", 0.1, 0.1], ["commander", 0.1, 0.8], ["commentator", 0.0, 0.4], ["commissioner", 0.0, 0.8], ["composer", 0.1, 0.4], ["conductor", 0.1, 0.6], ["confesses", 0.0, 0.0], ["congressman", 0.7, 0.3], ["constable", 0.2, 0.6], ["consultant", 0.0, 0.1], ["cop", 0.2, 0.6], ["correspondent", 0.0, 0.0], ["councilman", 0.8, 0.1], ["councilor", -0.1, -0.1], ["counselor", 0.0, -0.1], ["critic", 0.1, 0.4], ["crooner", 0.2, 0.2], ["crusader", 0.1, 0.7], ["curator", -0.1, 0.2], ["custodian", 0.1, 0.9], ["dad", 1.0, 0.0], ["dancer", -0.1, -0.9], ["dean", 0.2, 0.7], ["dentist", 0.0, 0.7], ["deputy", 0.1, 0.7], ["dermatologist", 0.0, -0.3], ["detective", 0.1, 0.5], ["diplomat", 0.0, 0.5], ["director", 0.1, 0.6], ["disc_jockey", 0.2, 0.6], ["doctor", 0.0, 0.7], ["doctoral_student", 0.0, 0.3], ["drug_addict", 0.0, 0.0], ["drummer", 0.0, 0.9], ["economics_professor", 0.1, 0.6], ["economist", 0.1, 0.5], ["editor", 0.1, 0.4], ["educator", 0.0, -0.5], ["electrician", 0.1, 0.8], ["employee", 0.0, 0.0], ["entertainer", 0.0, 0.0], ["entrepreneur", 0.0, 0.5], ["environmentalist", 0.0, -0.4], ["envoy", 0.1, 0.2], ["epidemiologist", 0.0, 0.0], ["evangelist", 0.1, 0.4], ["farmer", 0.1, 0.8], ["fashion_designer", -0.2, -0.4], ["fighter_pilot", 0.2, 0.7], ["filmmaker", 0.1, 0.3], ["financier", 0.1, 0.5], ["firebrand", 0.0, 0.1], ["firefighter", 0.1, 0.7], ["fireman", 0.8, 0.2], ["fisherman", 0.9, 0.1], ["footballer", 0.4, 0.5], ["foreman", 0.5, 0.4], ["freelance_writer", 0.0, 0.0], ["gangster", 0.2, 0.7], ["gardener", -0.1, 0.0], ["geologist", 0.0, 0.4], ["goalkeeper", 0.1, 0.5], ["graphic_designer", 0.0, 0.2], ["guidance_counselor", 0.0, 0.0], ["guitarist", 0.1, 0.5], ["hairdresser", -0.2, -0.8], ["handyman", 0.8, 0.2], ["headmaster", 0.4, 0.2], ["historian", 0.0, 0.5], ["hitman", 0.8, 0.2], ["homemaker", -0.1, -0.9], ["hooker", -0.2, -0.8], ["housekeeper", -0.2, -0.8], ["housewife", -1.0, 0.0], ["illustrator", 0.0, 0.2], ["industrialist", 0.1, 0.7], ["infielder", 0.1, 0.5], ["inspector", 0.1, 0.5], ["instructor", 0.0, -0.3], ["interior_designer", -0.2, -0.6], ["inventor", 0.1, 0.5], ["investigator", 0.1, 0.5], ["investment_banker", 0.1, 0.7], ["janitor", 0.1, 0.9], ["jeweler", 0.1, 0.3], ["journalist", -0.1, 0.3], ["judge", 0.0, 0.7], ["jurist", 0.0, 0.0], ["laborer", 0.1, 0.9], ["landlord", 0.1, 0.4], ["lawmaker", 0.0, 0.7], ["lawyer", 0.1, 0.5], ["lecturer", 0.0, 0.2], ["legislator", 0.1, 0.7], ["librarian", -0.1, -0.9], ["lieutenant", 0.1, 0.7], ["lifeguard", 0.0, 0.6], ["lyricist", 0.0, -0.2], ["maestro", 0.1, 0.5], ["magician", 0.1, 0.7], ["magistrate", 0.0, 0.8], ["maid", -0.4, -0.6], ["major_leaguer", 0.2, 0.7], ["manager", 0.0, 0.6], ["marksman", 0.6, 0.4], ["marshal", 0.1, 0.7], ["mathematician", 0.0, 0.8], ["mechanic", 0.3, 0.6], ["mediator", 0.0, -0.2], ["medic", 0.1, 0.4], ["midfielder", 0.3, 0.5], ["minister", 0.1, 0.8], ["missionary", 0.0, 0.3], ["mobster", 0.1, 0.9], ["monk", 0.8, 0.1], ["musician", 0.0, 0.0], ["nanny", -0.3, -0.7], ["narrator", 0.0, 0.2], ["naturalist", 0.0, -0.2], ["negotiator", 0.0, 0.3], ["neurologist", 0.0, 0.6], ["neurosurgeon", 0.0, 0.7], ["novelist", 0.0, 0.0], ["nun", -0.8, -0.1], ["nurse", -0.1, -0.9], ["observer", 0.0, -0.1], ["officer", 0.1, 0.8], ["organist", -0.2, -0.3], ["painter", 0.0, 0.2], ["paralegal", -0.1, -0.4], ["parishioner", 0.0, 0.1], ["parliamentarian", 0.0, 0.6], ["pastor", 0.3, 0.7], ["pathologist", 0.0, 0.3], ["patrolman", 1.0, 0.0], ["pediatrician", 0.0, -0.2], ["performer", 0.0, -0.2], ["pharmacist", 0.0, 0.3], ["philanthropist", 0.0, 0.3], ["philosopher", 0.0, 0.8], ["photographer", 0.0, -0.1], ["photojournalist", 0.0, 0.1], ["physician", 0.0, 0.6], ["physicist", 0.1, 0.7], ["pianist", 0.0, -0.1], ["planner", 0.0, -0.3], ["plastic_surgeon", 0.2, 0.4], ["playwright", 0.0, 0.5], ["plumber", 0.1, 0.8], ["poet", 0.0, -0.1], ["policeman", 0.8, 0.2], ["politician", 0.0, 0.5], ["pollster", 0.0, 0.3], ["preacher", 0.2, 0.7], ["president", 0.1, 0.9], ["priest", 0.7, 0.3], ["principal", 0.0, 0.3], ["prisoner", 0.1, 0.6], ["professor", 0.1, 0.4], ["professor_emeritus", 0.0, 0.5], ["programmer", 0.2, 0.6], ["promoter", 0.0, 0.3], ["proprietor", 0.1, 0.4], ["prosecutor", -0.1, 0.3], ["protagonist", 0.0, 0.1], ["protege", 0.0, 0.2], ["protester", -0.1, 0.0], ["provost", 0.0, 0.4], ["psychiatrist", 0.0, -0.2], ["psychologist", 0.0, 0.0], ["publicist", -0.1, -0.2], ["pundit", 0.0, 0.2], ["rabbi", 0.2, 0.6], ["radiologist", 0.0, -0.3], ["ranger", 0.2, 0.7], ["realtor", -0.2, -0.2], ["receptionist", -0.3, -0.7], ["registered_nurse", -0.1, -0.9], ["researcher", 0.0, 0.1], ["restaurateur", 0.0, 0.2], ["sailor", 0.1, 0.8], ["saint", 0.2, 0.3], ["salesman", 0.8, 0.2], ["saxophonist", 0.1, 0.5], ["scholar", 0.0, 0.6], ["scientist", 0.0, 0.5], ["screenwriter", 0.1, 0.4], ["sculptor", 0.0, 0.5], ["secretary", -0.2, -0.8], ["senator", 0.1, 0.7], ["sergeant", 0.1, 0.7], ["servant", 0.0, 0.1], ["serviceman", 0.7, 0.3], ["sheriff_deputy", 0.1, 0.8], ["shopkeeper", 0.0, 0.5], ["singer", 0.0, -0.2], ["singer_songwriter", 0.0, -0.3], ["skipper", 0.1, 0.7], ["socialite", -0.4, -0.3], ["sociologist", 0.0, -0.2], ["soft_spoken", -0.1, -0.9], ["soldier", 0.3, 0.6], ["solicitor", 0.1, 0.3], ["solicitor_general", 0.0, 0.5], ["soloist", -0.1, -0.3], ["sportsman", 0.9, 0.1], ["sportswriter", 0.1, 0.9], ["statesman", 0.6, 0.4], ["steward", 0.4, -0.1], ["stockbroker", 0.1, 0.5], ["strategist", 0.0, 0.3], ["student", 0.0, 0.0], ["stylist", -0.2, -0.7], ["substitute", -0.1, -0.1], ["superintendent", 0.0, 0.9], ["surgeon", 0.1, 0.7], ["surveyor", 0.0, 0.5], ["swimmer", 0.0, 0.0], ["taxi_driver", 0.1, 0.9], ["teacher", 0.0, -0.8], ["technician", 0.1, 0.6], ["teenager", 0.0, -0.1], ["therapist", -0.1, -0.4], ["trader", 0.1, 0.6], ["treasurer", 0.0, -0.3], ["trooper", 0.2, 0.5], ["trucker", 0.2, 0.7], ["trumpeter", 0.0, 0.2], ["tutor", 0.0, -0.3], ["tycoon", 0.1, 0.7], ["undersecretary", 0.0, -0.3], ["understudy", 0.0, 0.0], ["valedictorian", 0.0, 0.0], ["vice_chancellor", 0.0, 0.6], ["violinist", -0.1, -0.3], ["vocalist", 0.0, -0.3], ["waiter", 1.0, 0.0], ["waitress", -0.9, -0.1], ["warden", 0.1, 0.9], ["warrior", 0.1, 0.9], ["welder", 0.3, 0.6], ["worker", 0.0, 0.3], ["wrestler", 0.2, 0.6], ["writer", 0.0, 0.0]]

def race_scores(client, race_direction, analogies):
    """Let's now look at the words with the largest *negative* projection onto the race dimension."""
    words = set()

    words2 = list(client.wv.key_to_index.keys())
    for a in occupations:
        if a[0] in words2:
            words.add(a[0])

    df = pd.DataFrame(data={"word": list(words)})
    df["race_score"] = df["word"].map(
        lambda w: client.wv.get_vector(w).dot(race_direction))
    df.sort_values(by="race_score", inplace=True)
    print (df.word.head(10))

    """Let's now look at the words with the largest *positive* projection onto the race dimension."""

    df.sort_values(by="race_score", inplace=True, ascending=False)
    print (df.word.head(10))

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
                #print("Step", step)
                #print("pred loss o", pred_loss_o)
                #print("protect loss o", protect_loss_o)
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
  A = "white_man"
  B = "black_woman"
  C = "boss"
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
