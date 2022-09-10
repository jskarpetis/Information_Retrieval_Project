from elasticsearch import Elasticsearch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from operator import indexOf
from tqdm import trange
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re
# NOT TO SUBMIT
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences

elasticsearch = Elasticsearch(host="localhost", port=9200)


def all_book_search():
    try:
        res = elasticsearch.search(index="bx-books", query={
            "match_all": {},
        }, size=100, scroll='1m')
        # track_total_hits brings all the data back, bad practice i believe, scroll api better but this is easier
    except:
        print("There has been an error in retrieving the data!")
        return

    if (res["hits"]["total"]["value"] == 0):
        return None

    return res


def plot_data(arrays, pair_list=False, centroids=False, centroid_list=None):
    if (pair_list):
        for matrix in arrays:
            xs = [x[0] for x in matrix]
            ys = [x[1] for x in matrix]
            plt.scatter(xs, ys, label=f"{indexOf(arrays, matrix)} cluster")
            if (centroids):
                xs = [x[0] for x in centroid_list]
                ys = [x[1] for x in centroid_list]
                plt.scatter(xs, ys, s=80, color='Black')
        plt.legend(loc='best')
        plt.show()


def k_means(n_clusters, dataset, epochs):

    centroids = {}
    clusters = {}
    variance_array = []

    # Initial clusters
    for cluster in range(n_clusters):
        centroids[cluster] = np.random.randint(
            0, 100000, np.shape(dataset[0]))

    # print(centroids)
    ### Algorithm to minimize the total_variance###
    for epoch in trange(epochs):
        for cluster_id in range(n_clusters):
            clusters[cluster_id] = []
        total_variance = 0
        for vector in dataset:
            # print("\nData point --> {}\n".format(point))
            stored_cosine_similarity_scores = []

            for cluster_id in centroids:
                centroid_coord = centroids[cluster_id]
                cosine_similarity = np.dot(
                    vector, centroid_coord) / (np.linalg.norm(vector) * np.linalg.norm(centroid_coord))
                # print('Norm squared --> {}'.format(norm_squared))
                stored_cosine_similarity_scores.append(cosine_similarity)

            minimum_norm_index = np.argmin(stored_cosine_similarity_scores)
            # print(minimum_norm_index)

            clusters[minimum_norm_index].append(vector)
            total_variance += abs(
                stored_cosine_similarity_scores[minimum_norm_index])
            # print(total_variance)

        for cluster_id in centroids:
            if (clusters[cluster_id] == []):
                continue
            cluster_data = clusters[cluster_id]

            best_centroid = np.zeros(shape=(np.shape(dataset[0])))

            for vector in cluster_data:
                best_centroid = np.add(best_centroid, vector)
            best_centroid = np.divide(best_centroid, len(cluster_data))
            centroids[cluster_id] = best_centroid.tolist()

        variance_array.append(total_variance)

    return variance_array, clusters, centroids


def pre_process_summary(summary, vocab_size, max_len):
    tokenized_sentence = str(tokenize_sentence(summary))

    # print(tokenized_sentence, '\n')
    encoded_data = [one_hot(tokenized_sentence, vocab_size)]
    # print(encoded_data, "\n")
    padded_data = pad_sequences(
        encoded_data, maxlen=max_len, padding='post')
    # print(padded_data, "\n")
    return padded_data


def tokenize_sentence(sentence):

    sentence_strip_numbers = re.sub(r"[0-9]+", "", sentence)
    final_sentence = re.sub(
        r"!|@|#|$|%|^|&|:|;|'|<|>|/|-|=|(|)|", "", sentence_strip_numbers)
    final_sentence = re.sub('[\W\_]', ' ', final_sentence)

    split_sentence = re.split('\s+', final_sentence)
    return split_sentence


def pca_data(n_components, data):

    for cluster_id in data.keys():
        scaled_data = StandardScaler().fit_transform(data[cluster_id])
        pca = PCA(n_components=n_components)
        principal_components = pca.fit_transform(scaled_data)
        data[cluster_id] = principal_components
    return data


def pca_centroids(n_components, centroids):

    for cluster_id in centroids.keys():
        temp_data = [centroids[cluster_id]]
        print(temp_data)
        scaled_data = StandardScaler().fit_transform(temp_data)
        pca = PCA(n_components=n_components)
        principal_components = pca.fit_transform(scaled_data)
        centroids[cluster_id] = principal_components
    return centroids


if __name__ == "__main__":
    batch_count = 0
    vocab_size = 100000
    max_summary_length = 0
    all_books_response = all_book_search()

    scroll_dataset = {}

    old_scroll_id = all_books_response['_scroll_id']
    results = all_books_response['hits']['hits']

    while len(results):
        # print('Batch with No --> {}'.format(batch_count))
        for i, r in enumerate(results):
            # Minor preprocessing
            book_summary = r['_source']['summary']
            tokenized_summary = tokenize_sentence(book_summary)

            # Storing the book_isbn(key) and summary(value) to a dictionary
            scroll_dataset[r['_source']['isbn']] = tokenized_summary

            # Finding max len to use at the conversion of the words to embeddings
            if (max_summary_length < len(tokenized_summary)):
                max_summary_length = len(tokenized_summary)

        # Finding the next batch of data
        result = elasticsearch.scroll(scroll_id=old_scroll_id, scroll='1m')

        # Storing the new scroll_id to use
        if old_scroll_id != result['_scroll_id']:
            old_scroll_id = result['_scroll_id']

        # Storing the next batch of data to results
        results = result['hits']['hits']
        batch_count += 1
        break

    print('################################################################################################## ONE HOT ENCODE ##################################################################################################')
    for key in scroll_dataset.keys():
        one_hot_encoded_summary = pre_process_summary(
            summary=str(scroll_dataset[key]), vocab_size=vocab_size, max_len=max_summary_length)

        scroll_dataset[key] = one_hot_encoded_summary[0].tolist()

        # print('Length of sentence -> {}\t\tLength of encoded sentence -> {}'.format(
        #     len(scroll_dataset[key]), len(one_hot_encoded_summary[0].tolist())))

    dataset_clustering = list(scroll_dataset.values())

    variance_array, clusters, centroids = k_means(30, dataset_clustering, 150)
    # print(variance_array, "\n", list(centroids.values()))

    # print(clusters)

    for key in clusters.keys():
        print(clusters[key], '\n')

    # Data is now ready to be clustered
