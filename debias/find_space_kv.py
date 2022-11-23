
import numpy as np
import pandas as pd
import heapq
"""
### Defining the Protected Variable of Embeddings

The description of how to incorporate adversarial networks into machine learned models above is very generic because the technique is generally applicable for any type of systems which can be described in terms of input $X$ being predictive of $Y$ but potentially containing information about a protected variable $Z$.  So long as you can construct the relevant update functions you can apply this technique.  However, that doesn’t tell you much about the nature of $X$, $Y$ and $Z$.  In the case of the word analogies task above, $X = B + C - A$ and $Y = D$.  Figuring out what $Z$ should be is a little bit trickier though.  For that we can look to a paper by [Bulokbasi et. al.](http://papers.nips.cc/paper/6227-man-is-to-computer-programmer-as-woman-is-to-homemaker-debiasing-word-embeddings) where they developed an unsupervised methodology for removing gendered semantics from word embeddings.

The first step is to select pairs of words which are relevant to the type of bias you are trying to remove.  In the case of gender you can choose word pairs like “man”:”woman” and “boy”:girl” which have gender as the only difference in their semantics.  Once you have these word pairs you can compute the difference between their embeddings to produce vectors in the embeddings’ semantic space which are roughly parallel to the semantics of gender.  Performing Principal Components Analysis (PCA) on those vectors then gives you the major components of the semantics of gender as defined by the gendered word pairs provided.
"""

occupations = [["accountant", 0.0, 0.4], ["acquaintance", 0.0, 0.0], ["actor", 0.8, 0.0], ["actress", -1.0, 0.0], ["adjunct_professor", 0.0, 0.5], ["administrator", 0.0, 0.2], ["adventurer", 0.0, 0.5], ["advocate", 0.0, -0.1], ["aide", 0.0, -0.2], ["alderman", 0.7, 0.2], ["alter_ego", 0.0, 0.0], ["ambassador", 0.0, 0.7], ["analyst", 0.0, 0.4], ["anthropologist", 0.0, 0.4], ["archaeologist", 0.0, 0.6], ["archbishop", 0.4, 0.5], ["architect", 0.1, 0.6], ["artist", 0.0, -0.2], ["artiste", -0.1, -0.2], ["assassin", 0.1, 0.8], ["assistant_professor", 0.1, 0.4], ["associate_dean", 0.0, 0.4], ["associate_professor", 0.0, 0.4], ["astronaut", 0.1, 0.8], ["astronomer", 0.1, 0.5], ["athlete", 0.0, 0.7], ["athletic_director", 0.1, 0.7], ["attorney", 0.0, 0.3], ["author", 0.0, 0.1], ["baker", 0.0, -0.1], ["ballerina", -0.5, -0.5], ["ballplayer", 0.2, 0.8], ["banker", 0.0, 0.6], ["barber", 0.5, 0.5], ["baron", 0.6, 0.3], ["barrister", 0.1, 0.4], ["bartender", 0.0, 0.3], ["biologist", 0.0, 0.1], ["bishop", 0.6, 0.4], ["bodyguard", 0.1, 0.9], ["bookkeeper", 0.0, -0.4], ["boss", 0.0, 0.7], ["boxer", 0.1, 0.9], ["broadcaster", -0.1, 0.4], ["broker", 0.1, 0.5], ["bureaucrat", 0.1, 0.5], ["businessman", 0.8, 0.2], ["businesswoman", -0.9, -0.1], ["butcher", 0.1, 0.9], ["butler", 0.5, 0.5], ["cab_driver", 0.1, 0.8], ["cabbie", 0.1, 0.6], ["cameraman", 0.8, 0.1], ["campaigner", 0.0, 0.2], ["captain", 0.1, 0.6], ["cardiologist", 0.1, 0.5], ["caretaker", 0.0, -0.9], ["carpenter", 0.1, 0.8], ["cartoonist", 0.0, 0.5], ["cellist", -0.1, 0.0], ["chancellor", 0.1, 0.6], ["chaplain", 0.1, 0.6], ["character", 0.0, 0.0], ["chef", 0.0, 0.5], ["chemist", 0.0, 0.2], ["choreographer", -0.2, -0.2], ["cinematographer", 0.0, 0.5], ["citizen", 0.0, 0.0], ["civil_servant", 0.0, 0.2], ["cleric", 0.3, 0.3], ["clerk", 0.0, -0.5], ["coach", 0.1, 0.8], ["collector", 0.0, 0.4], ["colonel", 0.1, 0.8], ["columnist", 0.0, 0.2], ["comedian", 0.0, 0.3], ["comic", 0.1, 0.1], ["commander", 0.1, 0.8], ["commentator", 0.0, 0.4], ["commissioner", 0.0, 0.8], ["composer", 0.1, 0.4], ["conductor", 0.1, 0.6], ["confesses", 0.0, 0.0], ["congressman", 0.7, 0.3], ["constable", 0.2, 0.6], ["consultant", 0.0, 0.1], ["cop", 0.2, 0.6], ["correspondent", 0.0, 0.0], ["councilman", 0.8, 0.1], ["councilor", -0.1, -0.1], ["counselor", 0.0, -0.1], ["critic", 0.1, 0.4], ["crooner", 0.2, 0.2], ["crusader", 0.1, 0.7], ["curator", -0.1, 0.2], ["custodian", 0.1, 0.9], ["dad", 1.0, 0.0], ["dancer", -0.1, -0.9], ["dean", 0.2, 0.7], ["dentist", 0.0, 0.7], ["deputy", 0.1, 0.7], ["dermatologist", 0.0, -0.3], ["detective", 0.1, 0.5], ["diplomat", 0.0, 0.5], ["director", 0.1, 0.6], ["disc_jockey", 0.2, 0.6], ["doctor", 0.0, 0.7], ["doctoral_student", 0.0, 0.3], ["drug_addict", 0.0, 0.0], ["drummer", 0.0, 0.9], ["economics_professor", 0.1, 0.6], ["economist", 0.1, 0.5], ["editor", 0.1, 0.4], ["educator", 0.0, -0.5], ["electrician", 0.1, 0.8], ["employee", 0.0, 0.0], ["entertainer", 0.0, 0.0], ["entrepreneur", 0.0, 0.5], ["environmentalist", 0.0, -0.4], ["envoy", 0.1, 0.2], ["epidemiologist", 0.0, 0.0], ["evangelist", 0.1, 0.4], ["farmer", 0.1, 0.8], ["fashion_designer", -0.2, -0.4], ["fighter_pilot", 0.2, 0.7], ["filmmaker", 0.1, 0.3], ["financier", 0.1, 0.5], ["firebrand", 0.0, 0.1], ["firefighter", 0.1, 0.7], ["fireman", 0.8, 0.2], ["fisherman", 0.9, 0.1], ["footballer", 0.4, 0.5], ["foreman", 0.5, 0.4], ["freelance_writer", 0.0, 0.0], ["gangster", 0.2, 0.7], ["gardener", -0.1, 0.0], ["geologist", 0.0, 0.4], ["goalkeeper", 0.1, 0.5], ["graphic_designer", 0.0, 0.2], ["guidance_counselor", 0.0, 0.0], ["guitarist", 0.1, 0.5], ["hairdresser", -0.2, -0.8], ["handyman", 0.8, 0.2], ["headmaster", 0.4, 0.2], ["historian", 0.0, 0.5], ["hitman", 0.8, 0.2], ["homemaker", -0.1, -0.9], ["hooker", -0.2, -0.8], ["housekeeper", -0.2, -0.8], ["housewife", -1.0, 0.0], ["illustrator", 0.0, 0.2], ["industrialist", 0.1, 0.7], ["infielder", 0.1, 0.5], ["inspector", 0.1, 0.5], ["instructor", 0.0, -0.3], ["interior_designer", -0.2, -0.6], ["inventor", 0.1, 0.5], ["investigator", 0.1, 0.5], ["investment_banker", 0.1, 0.7], ["janitor", 0.1, 0.9], ["jeweler", 0.1, 0.3], ["journalist", -0.1, 0.3], ["judge", 0.0, 0.7], ["jurist", 0.0, 0.0], ["laborer", 0.1, 0.9], ["landlord", 0.1, 0.4], ["lawmaker", 0.0, 0.7], ["lawyer", 0.1, 0.5], ["lecturer", 0.0, 0.2], ["legislator", 0.1, 0.7], ["librarian", -0.1, -0.9], ["lieutenant", 0.1, 0.7], ["lifeguard", 0.0, 0.6], ["lyricist", 0.0, -0.2], ["maestro", 0.1, 0.5], ["magician", 0.1, 0.7], ["magistrate", 0.0, 0.8], ["maid", -0.4, -0.6], ["major_leaguer", 0.2, 0.7], ["manager", 0.0, 0.6], ["marksman", 0.6, 0.4], ["marshal", 0.1, 0.7], ["mathematician", 0.0, 0.8], ["mechanic", 0.3, 0.6], ["mediator", 0.0, -0.2], ["medic", 0.1, 0.4], ["midfielder", 0.3, 0.5], ["minister", 0.1, 0.8], ["missionary", 0.0, 0.3], ["mobster", 0.1, 0.9], ["monk", 0.8, 0.1], ["musician", 0.0, 0.0], ["nanny", -0.3, -0.7], ["narrator", 0.0, 0.2], ["naturalist", 0.0, -0.2], ["negotiator", 0.0, 0.3], ["neurologist", 0.0, 0.6], ["neurosurgeon", 0.0, 0.7], ["novelist", 0.0, 0.0], ["nun", -0.8, -0.1], ["nurse", -0.1, -0.9], ["observer", 0.0, -0.1], ["officer", 0.1, 0.8], ["organist", -0.2, -0.3], ["painter", 0.0, 0.2], ["paralegal", -0.1, -0.4], ["parishioner", 0.0, 0.1], ["parliamentarian", 0.0, 0.6], ["pastor", 0.3, 0.7], ["pathologist", 0.0, 0.3], ["patrolman", 1.0, 0.0], ["pediatrician", 0.0, -0.2], ["performer", 0.0, -0.2], ["pharmacist", 0.0, 0.3], ["philanthropist", 0.0, 0.3], ["philosopher", 0.0, 0.8], ["photographer", 0.0, -0.1], ["photojournalist", 0.0, 0.1], ["physician", 0.0, 0.6], ["physicist", 0.1, 0.7], ["pianist", 0.0, -0.1], ["planner", 0.0, -0.3], ["plastic_surgeon", 0.2, 0.4], ["playwright", 0.0, 0.5], ["plumber", 0.1, 0.8], ["poet", 0.0, -0.1], ["policeman", 0.8, 0.2], ["politician", 0.0, 0.5], ["pollster", 0.0, 0.3], ["preacher", 0.2, 0.7], ["president", 0.1, 0.9], ["priest", 0.7, 0.3], ["principal", 0.0, 0.3], ["prisoner", 0.1, 0.6], ["professor", 0.1, 0.4], ["professor_emeritus", 0.0, 0.5], ["programmer", 0.2, 0.6], ["promoter", 0.0, 0.3], ["proprietor", 0.1, 0.4], ["prosecutor", -0.1, 0.3], ["protagonist", 0.0, 0.1], ["protege", 0.0, 0.2], ["protester", -0.1, 0.0], ["provost", 0.0, 0.4], ["psychiatrist", 0.0, -0.2], ["psychologist", 0.0, 0.0], ["publicist", -0.1, -0.2], ["pundit", 0.0, 0.2], ["rabbi", 0.2, 0.6], ["radiologist", 0.0, -0.3], ["ranger", 0.2, 0.7], ["realtor", -0.2, -0.2], ["receptionist", -0.3, -0.7], ["registered_nurse", -0.1, -0.9], ["researcher", 0.0, 0.1], ["restaurateur", 0.0, 0.2], ["sailor", 0.1, 0.8], ["saint", 0.2, 0.3], ["salesman", 0.8, 0.2], ["saxophonist", 0.1, 0.5], ["scholar", 0.0, 0.6], ["scientist", 0.0, 0.5], ["screenwriter", 0.1, 0.4], ["sculptor", 0.0, 0.5], ["secretary", -0.2, -0.8], ["senator", 0.1, 0.7], ["sergeant", 0.1, 0.7], ["servant", 0.0, 0.1], ["serviceman", 0.7, 0.3], ["sheriff_deputy", 0.1, 0.8], ["shopkeeper", 0.0, 0.5], ["singer", 0.0, -0.2], ["singer_songwriter", 0.0, -0.3], ["skipper", 0.1, 0.7], ["socialite", -0.4, -0.3], ["sociologist", 0.0, -0.2], ["soft_spoken", -0.1, -0.9], ["soldier", 0.3, 0.6], ["solicitor", 0.1, 0.3], ["solicitor_general", 0.0, 0.5], ["soloist", -0.1, -0.3], ["sportsman", 0.9, 0.1], ["sportswriter", 0.1, 0.9], ["statesman", 0.6, 0.4], ["steward", 0.4, -0.1], ["stockbroker", 0.1, 0.5], ["strategist", 0.0, 0.3], ["student", 0.0, 0.0], ["stylist", -0.2, -0.7], ["substitute", -0.1, -0.1], ["superintendent", 0.0, 0.9], ["surgeon", 0.1, 0.7], ["surveyor", 0.0, 0.5], ["swimmer", 0.0, 0.0], ["taxi_driver", 0.1, 0.9], ["teacher", 0.0, -0.8], ["technician", 0.1, 0.6], ["teenager", 0.0, -0.1], ["therapist", -0.1, -0.4], ["trader", 0.1, 0.6], ["treasurer", 0.0, -0.3], ["trooper", 0.2, 0.5], ["trucker", 0.2, 0.7], ["trumpeter", 0.0, 0.2], ["tutor", 0.0, -0.3], ["tycoon", 0.1, 0.7], ["undersecretary", 0.0, -0.3], ["understudy", 0.0, 0.0], ["valedictorian", 0.0, 0.0], ["vice_chancellor", 0.0, 0.6], ["violinist", -0.1, -0.3], ["vocalist", 0.0, -0.3], ["waiter", 1.0, 0.0], ["waitress", -0.9, -0.1], ["warden", 0.1, 0.9], ["warrior", 0.1, 0.9], ["welder", 0.3, 0.6], ["worker", 0.0, 0.3], ["wrestler", 0.2, 0.6], ["writer", 0.0, 0.0]]
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
            vecs.append(_np_normalize(client.wv.get_vector(word)))
            index_map[word] = len(words)
            words.append(word)
        except KeyError:
            #print("word not found: %s" % word)
            pass
    print("words not filtered out: %d" % len(words))

    return np.array(vecs), index_map, words

        # ("white_man", "black_man"),
        # ("white_woman", "black_woman"),
        # ("white_teenager", "black_teenager"),
        # ("white_student", "black_student"),
        # ("white_people", "black_people"),
        # ("white_men", "black_men"),
        # ("whites", "blacks"),
        # ("white_women", "black_women"),
        # ("white_girl", "black_girl"),
        # ("white_boy", "black_boy"),
        # ("white_youth", "black_youth"),
        # ("white_family", "black_family"),

def find_gender_direction(embed, indices, client):
    """Finds and returns a 'gender direction'."""
    pairs = [
        ("white_woman", "white_man"),
        ("black_woman", "black_man"),
        ("woman", "man"),
        ("her", "his"),
        ("she", "he"),
        ("aunt", "uncle"),
        ("niece", "nephew"),
        ("mother", "father"),
        ("daughter", "son"),
        ("granddaughter", "grandson"),
        ("girl", "boy"),
        ("stepdaughter", "stepson"),
        ("mom", "dad"),
    ]

    # Creates a numpy array which is just n (no.of pairs) normalised vectors
    # of he-she (or equivalent)
    m = []
    for wof, wom in pairs:
        m.append((_np_normalize(client.wv.get_vector(wof))) - (_np_normalize(client.wv.get_vector(wom))))
    m = np.array(m)

    # the next three lines are just a PCA
    # Creates a covariance matrix of M transpose
    m = np.cov(m.T)
    # eigenvalues and eigenvectors of M
    evals, evecs = np.linalg.eig(m)
    # 
    return _np_normalize(np.real(evecs[:, np.argmax(evals)]))

wm = ["adam", "harry", "josh", "roger", "alan", "frank", "justin", "ryan", "andrew", "jack", "matthew", "stephen", "brad", "greg", "paul", " jonathan", "peter"]
ww = ["amanda", "courtney", "heather", "melanie", "katie", "betsy", "kristin", "nancy", "stephanie", "ellen", "lauren", "colleen", "emily", "megan", "rachel"]
bm = ["alonzo", "jamel", "theo", "alphonse", "jerome", "leroy", "torrance", "darnell", "lamar", "lionel", "tvree", "deion", "lamont", "malik", "terrence", "tyrone", "tavon", "marcellus", "wardell"]
bw = ["nichelle", "shereen", "ebony", "latisha", "shaniqua", "jasmine", "tanisha", "tia", "lakisha", "latoya", "yolanda", "malika", "yvette"]

wm1 = ['adam', 'harry', 'josh', 'roger', 'alan', 'frank', 'justin', 'ryan', 'andrew', 'jack', 'matthew', 'peter']
bm1 = ['alonzo', 'theo', 'jerome', 'leroy', 'torrance', 'darnell', 'lamar', 'lionel', 'lamont', 'malik', 'terrence', 'tyrone']

def find_avg_vector(client, names):
    words = []
    words2 = list(client.wv.key_to_index.keys())
    for a in names:
        if a in words2:
            words.append(a)
    print(words)
    master_vec = [client.wv.get_vector(words[0])]
    for item in words[1:]:
        master_vec += client.wv.get_vector(item)
    return master_vec / len(words)

def find_race_direction_with_names(client):
    wm_avg = find_avg_vector(client, wm)
    bm_avg = find_avg_vector(client, bm)
    ww_avg = find_avg_vector(client, ww)
    bw_avg = find_avg_vector(client, bw)
    """Finds and returns a 'race direction'."""
    # pairs = [
    #     (wm_avg, bm_avg),
    #     (ww_avg, bw_avg)
    # ]
    pairs = zip(wm1, bm1)

    # Creates a numpy array which is just n (no.of pairs) normalised vectors
    # of he-she (or equivalent)
    m = []
    for wof, wom in pairs:
        #m.append((_np_normalize(wof[0])) - (_np_normalize(wom[0])))
        m.append((_np_normalize(client.wv.get_vector(wof))) - (_np_normalize(client.wv.get_vector(wom))))

    m = np.array(m)
    # the next three lines are just a PCA
    # Creates a covariance matrix of M transpose
    m = np.cov(m.T)
    # eigenvalues and eigenvectors of M
    evals, evecs = np.linalg.eig(m)

    return _np_normalize(np.real(evecs[:, np.argmax(evals)]))


def find_race_direction(embed, indices, client):
    """Finds and returns a 'race direction'."""
    pairs = [
        ("white_man", "black_man"),
        ("white_woman", "black_woman"),
        ("white_teenager", "black_teenager"),
        #("white_student", "black_student"),
        ("white_people", "black_people"),
        ("caucasian", "african-american"),
        #("white_men", "black_men"),
        #("whites", "blacks"),
    ]

    # Creates a numpy array which is just n (no.of pairs) normalised vectors
    # of he-she (or equivalent)
    m = []
    for wof, wom in pairs:
        m.append((_np_normalize(client.wv.get_vector(wof))) - (_np_normalize(client.wv.get_vector(wom))))
    m = np.array(m)

    # the next three lines are just a PCA
    # Creates a covariance matrix of M transpose
    m = np.cov(m.T)
    # eigenvalues and eigenvectors of M
    evals, evecs = np.linalg.eig(m)

    return _np_normalize(np.real(evecs[:, np.argmax(evals)]))

def race_scores(client, race_direction, gender_direction):
    """Let's now look at the words with the largest *negative* projection onto the race dimension."""
    words = set()

    words2 = list(client.wv.key_to_index.keys())
    for a in occupations:
        if a[0] in words2:
            words.add(a[0])

    df = pd.DataFrame(data={"word": list(words)})
    df["race_score"] = df["word"].map(
        lambda w: (client.wv.get_vector(w).dot(race_direction)))
    df.sort_values(by="race_score", inplace=True)

    df["gender_score"] = df["word"].map(
        lambda w: (client.wv.get_vector(w).dot(gender_direction)))

    df.sort_values(by="race_score", inplace=True)

    print (df.head(10))
    """Let's now look at the words with the largest *positive* projection onto the race dimension."""

    df.sort_values(by="race_score", inplace=True, ascending=False)
    print (df.head(10))
    df.to_csv("./results.csv")


def identity_occupations(client):
    "Plain ol word similarity to different identities"
    words = []

    words2 = list(client.wv.key_to_index.keys())
    for a in occupations:
        if a[0] in words2:
            words.append(a[0])
    
    s_ids = [['black_woman', 0], ['white_woman', 0], ['black_man', 0], ['white_man', 0]]
    occupation_scores = []

    for a in words:
        temp_scores = []
        for id in s_ids:
            score = client.wv.similarity(a, id[0])
            temp_scores.append(score)
        occupation_scores.append([a, temp_scores])

    for num, id in enumerate(s_ids):
        print(num, id[0])
        data = heapq.nlargest(10, occupation_scores, key=lambda e:e[1][num])
        for d in data:
            print(d[0])


def main(client, analogies):
    embed, indices, words = load_vectors(client, analogies)

    embed_dim = len(embed[0].flatten())
    print("word embedding dimension: %d" % embed_dim)

    # Using the embeddings, find the gender vector.
    gender_direction = find_gender_direction(embed, indices, client)
    print("gender direction: %s" % str(gender_direction.flatten()))

    rd_names = find_race_direction_with_names(client)
    print("race direction: %s" % str(rd_names.flatten()))

    # Using the embeddings, find the race vector.
    race_direction = find_race_direction(embed, indices, client)
    #print("race direction: %s" % str(race_direction.flatten()))

    race_scores(client, race_direction, gender_direction)
    return indices, embed, gender_direction, race_direction

    """Once you have the first principal component of the embedding differences, you can start projecting the embeddings of words onto it.  
    That projection is roughly the degree to which a word is relevant to the latent protected variable defined by the first principle 
    component of the word pairs given.  This projection can then be taken as the protected variable $Z$ which the adversary is attempting 
    to predict on the basis of the predicted value of $Y$.  The code below illustrates how to construct a function which computes $Z$ from $X$ in this way.

    Try editing the WORD param in the next cell to see the projection of other words onto the gender direction.

    WORD = "he"
    word_vec = client.wv.get_vector(WORD)
    print(word_vec.dot(gender_direction))

    WORD = "brick"
    word_vec = client.wv.get_vector(WORD)
    print(word_vec.dot(gender_direction))

    WORD = "she"
    word_vec = client.wv.get_vector(WORD)
    print(word_vec.dot(gender_direction))

    """