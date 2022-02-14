from elasticsearch import Elasticsearch, helpers
import csv

elasticsearch = Elasticsearch(host="localhost", port=9200)


# Loading the data into elastic search, by creating the indices

with open('./IR Project/BX-Books.csv', encoding="UTF-8") as file:
    reader = csv.DictReader(file)
    helpers.bulk(elasticsearch, reader, index="bx-books")

with open('./IR Project/BX-Book-Ratings.csv', encoding="UTF-8") as file:
    reader = csv.DictReader(file)
    helpers.bulk(elasticsearch, reader, index="bx-book-ratings")

with open('./IR Project/BX-Users.csv', encoding="UTF-8") as file:
    reader = csv.DictReader(file)
    helpers.bulk(elasticsearch, reader, index="bx-users")
