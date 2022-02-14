import nltk
from elasticsearch import Elasticsearch
import re
import ast
from nltk import word_tokenize
from gensim.models import Word2Vec
import tensorflow
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.decomposition import PCA
from matplotlib import pyplot
elasticsearch = Elasticsearch(host="localhost", port=9200)


# Question 3
def basic_search(field, value):
    try:
        res = elasticsearch.search(index="bx-books", from_=0, size=10000, query={
            "match": {
              f"{field}": f"{value}"
            }},
            sort=[
                {
                    "_score": {
                        "order": "desc"
                    }
                }
            ]
        )
    except:
        print("There has been an error in retrieving the data!")
        return

    if (res["hits"]["total"]["value"] == 0):
        return

    return res


# Finds the rating that a specific uid has given to a book of specific isbn returning it as an integer, if it doesnt find any rating then it returns None
def user_rating_search(uid, isbn):
    try:
        res = elasticsearch.search(index="bx-book-ratings", query={
            "bool": {
              "must": [
                {"match": {
                  "isbn": f"{isbn}"
                  }
                },
                {
                  "match": {
                    "uid": f"{uid}"
                  }
                }

              ]
            }
          })
    except:
        print("There has been an error in retrieving the data!")

    if (res["hits"]["total"]["value"] == 0):
        return
    else:
        hit_rating = res["hits"]["hits"][0]["_source"]["rating"]
        return float(hit_rating)


def tokenize_sentence(sentence):

    sentence_strip_commas = sentence.replace(",", "")
    sentence_strip_stops = sentence_strip_commas.replace(".", "")
    sentence_strip_marks = sentence_strip_stops.replace("?", "")
    sentence_strip_stars = sentence_strip_marks.replace("*", "")

    sentence_strip_numbers = re.sub(r"[0-9]+", "", sentence_strip_stars)
    final_sentence = re.sub(r"!|@|#|$|%|^|&|:|;|'|<|>|/|-|=", "", sentence_strip_numbers)

    split_sentence = re.split('\s+', final_sentence)
    new_sentence = [word_tokenize(word) for word in split_sentence]
    return new_sentence


def find_rated_books(uid):

    res = elasticsearch.search(index="bx-book-ratings", from_=0, size=10000, query={
        "bool": {
          "must": [
            {
              "match": {
                "uid": f"{uid}"
              }
            }

          ]
        }
      })

    if (res["hits"]["total"]["value"] == 0):
        print("No data has been returned!")
        return

    return res


# Testing
# isbn: 0671789422
# uid: 168047
# Possessing the Secret of Joy
if __name__ == "__main__":

    training_data_for_user = {}

    print("Searching on index: bx-books.")
    print("You can search any field and value. Program ends with user_id input 'END'\n")

    while True:
        user_id = input("Give me a user_id(uid): ")
        if (user_id == 'END'):
            print("Program exiting.")
            break

        rated_books = find_rated_books(user_id)
        for book in rated_books["hits"]["hits"]:

            book_search = basic_search("isbn", book["_source"]["isbn"])
            if (book_search is None):
                continue
            summary = book_search["hits"]["hits"][0]["_source"]["summary"]
            book_rating = user_rating_search(user_id, book["_source"]["isbn"])
            tokenized_sentence = tokenize_sentence(summary)

            training_data_for_user[str(tokenized_sentence)] = book_rating

        print(len(training_data_for_user))

        for i in training_data_for_user:
            print(i)
            print(training_data_for_user[i])
            print('\n')






    # Making the model for this sentence
    # model = Word2Vec(tokenized_sentence, min_count=1)
    # tensorflow_model = keras.Sequential()
    # tensorflow_model.add(layers.Dense(23, activation="relu"))
    # tensorflow_model.add(layers.Dense(4, activation="relu"))
    # tensorflow_model.add(layers.Dense(1, activation="relu"))
    #
    # print(tensorflow_model)
    # print(tensorflow_model.layers[0].weights)


    # words = list(model.wv.index_to_key)
    # print(words, "\n")
    #
    #
    # # Word vectors
    # word_vectors = model.wv.vectors
    # # print(word_vectors)
    #
    # #Visualising
    # X = model.wv[model.wv.index_to_key]
    # pca = PCA(n_components=2)
    # result = pca.fit_transform(X)
    # pyplot.scatter(result[:, 0], result[:, 1])
    #
    # for i, word in enumerate(words):
    #     pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
    # pyplot.show()








