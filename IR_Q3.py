from base64 import encode
from elasticsearch import Elasticsearch
import re
import numpy as np

from tensorflow import keras
from tensorflow.keras import layers
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.preprocessing import text_dataset_from_directory
from sklearn.decomposition import PCA
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
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
    final_sentence = re.sub(
        r"!|@|#|$|%|^|&|:|;|'|<|>|/|-|=", "", sentence_strip_numbers)

    split_sentence = re.split('\s+', final_sentence)
    # new_sentence = [word_tokenize(word) for word in split_sentence]
    return split_sentence


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


def form_training_data(user_id):
    training_data_for_user = {}
    rated_books = find_rated_books(user_id)
    for book in rated_books["hits"]["hits"]:
        book_search = basic_search("isbn", book["_source"]["isbn"])
        if (book_search is None):
            continue
        summary = book_search["hits"]["hits"][0]["_source"]["summary"]
        book_rating = user_rating_search(user_id, book["_source"]["isbn"])
        tokenized_sentence = tokenize_sentence(summary)
        training_data_for_user[str(tokenized_sentence)] = book_rating
    return training_data_for_user


# Testing
# isbn: 0671789422
# uid: 168047
# 252071
# Possessing the Secret of Joy
if __name__ == "__main__":

    vocab_size = 10000

    print("Searching on index: bx-books.")
    print("You can search any field and value. Program ends with user_id input 'END'\n")

    while True:
        user_id = input("Give me a user_id(uid): ")
        if (user_id == 'END'):
            print("Program exiting.")
            break

        training_data = form_training_data(user_id=user_id)
        print(len(training_data))
        training_labels = np.array(list(training_data.values())) / 10
        # print(training_data.keys())
        encoded_data = [one_hot(sentence, vocab_size)
                        for sentence in training_data.keys()]
        # print(encoded_data)

        # Variable max length for the padding
        max_len = 0
        for sentence in encoded_data:
            if(len(sentence) > max_len):
                max_len = len(sentence)
        # print(max_len)
        padded_data = pad_sequences(
            encoded_data, maxlen=max_len, padding='post')

        # Now we are ready to do embedding layer

        model = Sequential()
        model.add(Embedding(vocab_size, 32, input_length=max_len))
        model.add(Flatten())
        model.add(Dense(100, activation='sigmoid'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='mse',
                      metrics=['accuracy'])
        # print(model.summary())

        # print(training_labels)

        history = model.fit(padded_data, training_labels,
                            epochs=20, verbose=1)
        loss, accuracy = model.evaluate(
            padded_data, training_labels, verbose=0)
        print('Accuracy: %f' % (accuracy*100))
