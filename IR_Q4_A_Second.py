import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from elasticsearch import Elasticsearch
from sklearn.cluster import KMeans


elasticsearch = Elasticsearch(host="localhost", port=9200)


def all_book_search():
    try:
        res = elasticsearch.search(index="bx-books", query={
            "match_all": {},
        }, size=100, scroll='1m')
    except:
        print("There has been an error in retrieving the data!")
        return

    if (res["hits"]["total"]["value"] == 0):
        return None

    return res


if __name__ == "__main__":
    batch_count = 0
    vocab_size = 50000
    all_books_response = all_book_search()

    old_scroll_id = all_books_response['_scroll_id']
    results = all_books_response['hits']['hits']

    dataframe = pd.DataFrame(columns=['isbn', 'summary'])

    while len(results):
        print('Batch with No --> {}'.format(batch_count))
        for i, r in enumerate(results):
            # Minor preprocessing
            isbn = r['_source']['isbn']
            book_summary = r['_source']['summary']

            dataframe.loc[len(dataframe.index)] = [isbn, book_summary]

        # Finding the next batch of data
        result = elasticsearch.scroll(scroll_id=old_scroll_id, scroll='1m')

        # Storing the new scroll_id to use
        if old_scroll_id != result['_scroll_id']:
            old_scroll_id = result['_scroll_id']

        # Storing the next batch of data to results
        results = result['hits']['hits']
        batch_count += 1

    td_vectorizer = TfidfVectorizer(stop_words='english')

    all_summaries = dataframe['summary']
    print(all_summaries)
    features = td_vectorizer.fit_transform(all_summaries)
    k = 30
    model = KMeans(n_clusters=k, init='k-means++',
                   max_iter=30, n_init=1, verbose=1)
    # The normalization of data is done to achieve cosine similarity
    label = model.fit_predict(normalize(features))

    dataframe['cluster_label'] = label

    dataframe.to_csv('./IR Project/Clustered_Books.csv')
