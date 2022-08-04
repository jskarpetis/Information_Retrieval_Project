from elasticsearch import Elasticsearch
import random
from tqdm import trange
import pandas as pd
import numpy as np
import re

from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.preprocessing import text_dataset_from_directory
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
elasticsearch = Elasticsearch(host="localhost", port=9200)


def all_book_search():
    try:
        res = elasticsearch.search(index="bx-books", query={
            "match_all": {}}, track_total_hits=True)
        # track_total_hits brings all the data back, bad practice i believe, scroll api better but this is easier
    except:
        print("There has been an error in retrieving the data!")
        return

    if (res["hits"]["total"]["value"] == 0):
        return None

    return res


def k_means(n_clusters, dataset, epochs, is_3d=False):
    n = len(dataset)
    centroids = {}
    clusters = {}
    variance_array = []

    # Initial clusters
    for cluster in range(n_clusters):
        if (is_3d is False):
            centroids[cluster] = [random.uniform(-3, 3), random.uniform(-3, 3)]
        else:
            centroids[cluster] = [
                random.uniform(-3, 3), random.uniform(-3, 3), random.uniform(0, 3)]

    # print(centroids)
    ### Algorithm to minimize the total_variance###
    for epoch in trange(epochs):
        for cluster_id in range(n_clusters):
            clusters[cluster_id] = []
        total_variance = 0
        for point in dataset:
            # print("\nData point --> {}\n".format(point))
            stored_norms = []

            for cluster_id in centroids:
                centroid_coord = centroids[cluster_id]
                norm_squared = np.linalg.norm(
                    np.subtract(point, centroid_coord))**2
                # print('Norm squared --> {}'.format(norm_squared))
                stored_norms.append(norm_squared)

            minimum_norm_index = np.argmin(stored_norms)
            # print(minimum_norm_index)

            clusters[minimum_norm_index].append(point.tolist())
            total_variance += stored_norms[minimum_norm_index]
            # print(total_variance)

        for cluster_id in centroids:
            if (clusters[cluster_id] == []):
                continue
            cluster_data = clusters[cluster_id]

            if (is_3d is False):
                best_centroid = np.zeros(shape=(2,))
            else:
                best_centroid = np.zeros(shape=(3,))

            for point in cluster_data:
                best_centroid = np.add(best_centroid, point)
            best_centroid = np.divide(best_centroid, len(cluster_data))
            centroids[cluster_id] = best_centroid.tolist()

        variance_array.append(total_variance)

    return variance_array, clusters, centroids


def pre_process_summary(summary, vocab_size, max_len):
    tokenized_sentence = str(tokenize_sentence(summary))

    print(tokenized_sentence, '\n')
    encoded_data = [one_hot(tokenized_sentence, vocab_size)]
    print(encoded_data, "\n")
    padded_data = pad_sequences(
        encoded_data, maxlen=max_len, padding='post')
    print(padded_data, "\n")
    return padded_data


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


if __name__ == "__main__":
    vocab_size = 10000
    dataset = []
    all_books = all_book_search()
    print(len(all_books))
    max_summary_len = 0
    count = 0
    for hit in all_books["hits"]["hits"]:
        count += 1
        summary = hit["_source"]["summary"]
        if (len(tokenize_sentence(summary)) > max_summary_len):
            max_summary_len = len(tokenize_sentence(summary))

        dataset.append(pre_process_summary(
            summary, vocab_size=vocab_size, max_len=max_summary_len))
    print(len(dataset))
    print(count)
