
import nltk
import string as s
from gensim.models import Word2Vec
from nltk.corpus import stopwords
import os
import gensim.downloader as api
from gensim.models import KeyedVectors

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

lemmatizer = nltk.stem.WordNetLemmatizer()

replace_list = ["white woman", "black woman", "white man", "black man", "white women", "white men", 
                "black women", "black men", "african american", "dual heritage", "ethnic minority",  
                "person of colour",  "woman of colour",  "black girl", "black boy",  "black teenager", 
                "black student", "black youth", "black family", "black people", "white girl", "white boy", 
                "white teenager", "white student", "white youth", "white family", "white people"]
no_lemmatize = ["blacks", "whites"]

# Tokenises a string and adds lowercase tokens to list
# Adds phrases using underscore
# Stops lemmatization of blacks and whites
def get_sent_tokens(sentence):
    list_tokens = []
    sentence = sentence.lower()
    list_tokens_sentence = nltk.tokenize.word_tokenize(sentence)
    for token in list_tokens_sentence:
        if token not in no_lemmatize:
            list_tokens.append(lemmatizer.lemmatize(token))
        else:
            list_tokens.append(token)

    sentence_joined = " ".join(list_tokens)

    # join list of tokens as one string then do replacement then split
    for j, word in enumerate(replace_list):
        word = word.lower()
        if word.lower() in sentence_joined:
            sentence_joined = sentence_joined.replace(word, word.replace(" ", "_"))
    sentence_split = sentence_joined.split(" ")
    return sentence_split

# Tokenises a string and adds lowercase tokens to list
# Adds phrases using underscore
# Stops lemmatization of blacks and whites
def get_sent_tokens_single(sentence):
    list_tokens = []
    sentence = sentence.lower()
    list_tokens_sentence = nltk.tokenize.word_tokenize(sentence)
    for token in list_tokens_sentence:
        if token not in no_lemmatize:
            list_tokens.append(lemmatizer.lemmatize(token))
        else:
            list_tokens.append(token)

    return list_tokens

# Removes punctuation from list of tokens
def remove_punctuations(lst):
    new_lst = []
    for i in lst:
        for j in s.punctuation:
            if j == "-" or j == "_":
                pass
            else:
                i = i.replace(j, '')
        new_lst.append(i)
    return new_lst


def clean_article(data, phrasal):
    stopwords = set(nltk.corpus.stopwords.words('english'))
    stopwords.update(["", "ha", "said", "wa", "nt", "would", "also", "could"])
    not_stopwords = ["he", "she", "she's", "he's", "herself", "himself", "her", "his", "hers", "him"]
    stopwords.difference_update(not_stopwords)
    final_article = []

    for sentence in data:
        sentence_tokens = []
        if phrasal:
            sentence_tokens = get_sent_tokens(sentence)
        else:
            sentence_tokens = get_sent_tokens_single(sentence)
        sentence_tokens = remove_punctuations(sentence_tokens)
        final_sentence = []
        for word in sentence_tokens:
            if word in stopwords:
                continue
            final_sentence.append(word)
        final_article.append(final_sentence)

    return final_article


def create_news_model(phrasal):
    print("Running script")

    rawdata = []
    with open("./data") as infile:
        print("Data loaded")
        for line in infile:
            rawdata.append(line)

    print("No. of sentences: ", len(rawdata))

    tokenized_articles = clean_article(rawdata, phrasal)
    print("Articles tokenized")

    # train model
    model = Word2Vec(tokenized_articles)
    # summarize the loaded model
    if not os.path.exists('models'):
        print("Creating a models directory")
        os.mkdir('models')
    
    if phrasal:
        model.save("./models/word2vec_phrasal.model")
        model.save('./models/vectors_phrasal.kv')
    else:
        model.save("./models/word2vec_single.model")
        model.save('./models/vectors_single.kv')

def load_news_model():
    client = KeyedVectors.load("./models/vectors_phrasal.kv")
    return client


create_news_model(True)
create_news_model(False)
